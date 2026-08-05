# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-test virtual address space headroom trace (issue #2381).

The one-shot capture in :mod:`helpers.oom_diagnostics` says what the machine
looked like at the moment of failure. It cannot say which test consumed the
address space, because by then hundreds of tests have run. This module answers
that by sampling once per test and writing the series to a CSV.

Address space is the resource that actually runs out, and none of the pool
attributes report it: ``used_mem_current`` and ``reserved_mem_current`` both
count backing memory, so an uncapped pool that has never allocated a byte reads
as zero on both while holding a device-memory-sized reservation. The only
direct read is to ask the driver for a reservation and see what it grants,
which is what :class:`HeadroomProbe` does.

Enabled by default on this branch; set ``CUDA_CORE_VA_TRACE=0`` to turn it off.
"""

import os
import pathlib
import time

from cuda.bindings import driver

DEFAULT_FILENAME = "cuda_core_va_trace.csv"

# cuMemAddressReserve requires a power-of-two alignment and a size that is a
# multiple of it. 2 MiB matches the granularity the driver uses for pools.
VA_ALIGNMENT = 2 * 1024 * 1024
MIN_PROBE_BYTES = VA_ALIGNMENT
# An uncapped pool reserves an address window scaling with device memory, so
# this is what "can another pool still be created" costs. Used as the probe
# ceiling; see HeadroomProbe for why the probe refuses to ask for more.
POOL_RESERVATION_MULTIPLE = 2
# Fallback when the device memory size cannot be read.
DEFAULT_CEILING_BYTES = 64 * 1024 * 1024 * 1024

COLUMNS = (
    "index",
    "nodeid",
    "headroom_bytes",
    "mem_free",
    "graph_reserved",
    "graph_used",
    "pool_reserved",
    "pool_used",
    "probe_kind",
    "reserve_calls",
    "probe_ms",
    "query_ms",
    "errors",
)


def trace_enabled():
    return os.environ.get("CUDA_CORE_VA_TRACE", "1") not in ("0", "", "false", "False")


def refine_steps():
    """Bisection steps when the headroom is searched. 0 gives power-of-two resolution."""
    return int(os.environ.get("CUDA_CORE_VA_TRACE_REFINE", 4))


def reprobe_interval():
    """Samples between full searches, which are what make a recovery visible."""
    return int(os.environ.get("CUDA_CORE_VA_TRACE_REPROBE", 100))


def sample_every():
    """Sample one test in N. Raise it when the per-sample cost is too high.

    A reservation the size of a pool costs hundreds of milliseconds on some
    configurations, which is minutes across a full suite. The ``probe_ms``
    column and the end-of-run summary report what it actually cost, so the
    first short run tells you whether a full one is affordable.
    """
    return max(int(os.environ.get("CUDA_CORE_VA_TRACE_EVERY", 1)), 1)


def align_up(size):
    """Round to a multiple of the alignment.

    Device memory sizes are not generally a multiple of it, and
    cuMemAddressReserve rejects sizes that are not with
    CUDA_ERROR_INVALID_VALUE -- which would otherwise read as "no address space
    left" at every size probed.
    """
    return ((size + VA_ALIGNMENT - 1) // VA_ALIGNMENT) * VA_ALIGNMENT


def _reserve_and_release(size):
    """True if the driver still grants a contiguous reservation of ``size``.

    A reservation costs address space but no memory, so this is a direct read
    of what the address space can satisfy. Raises if the release fails, since a
    leaked reservation would corrupt every later sample.
    """
    size = align_up(size)
    err, ptr = driver.cuMemAddressReserve(size, VA_ALIGNMENT, 0, 0)
    if err != driver.CUresult.CUDA_SUCCESS:
        return False
    (err,) = driver.cuMemAddressFree(ptr, size)
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuMemAddressFree({ptr!r}, {size:#x}) -> {err!r}; later samples are unreliable")
    return True


class HeadroomProbe:
    """Whether the driver can still satisfy a pool-sized reservation.

    Cost drove this design. Probing a large range is expensive, and the cost
    sits entirely in the driver's ``cuMemAddressFree``, not in the reserve and
    not in any Python or Cython layer: raw ctypes straight to nvcuda measures
    the same as cuda.bindings (309 ms vs 323 ms per probe), and splitting the
    pair attributes all of it to the release. It also scales with the size
    asked for -- 2 MiB is microseconds, a 48 GiB round trip ~300 ms, and a
    850 GiB one 5.2 s once graph memory has fragmented the space. Searching for
    the exact boundary every test is therefore unaffordable.

    So the probe never asks for more than ``ceiling``, the size a new pool
    would need. That is the operationally meaningful question -- pool creation
    starts failing exactly when this reservation cannot be satisfied -- and it
    keeps the steady state to one cheap, successful call. Detail only costs
    something once the ceiling can no longer be met, which is the regime worth
    paying for.

    ``kind`` records how the value was obtained: ``ceiling`` (the full ask was
    granted), ``hold`` (the previous, lower watermark was granted again), or
    ``drop`` (the ask was refused and a new watermark searched for).
    ``reserve`` is injectable so the search can be tested without a GPU.
    """

    def __init__(self, reserve=None, refine_steps=4, reprobe_interval=100):
        self._raw_reserve = _reserve_and_release if reserve is None else reserve
        self._refine_steps = refine_steps
        self._reprobe_interval = reprobe_interval
        self._watermark = None
        self._held = 0
        self.calls = 0
        self.kind = None

    def _reserve(self, size):
        self.calls += 1
        return self._raw_reserve(size)

    def _refine(self, low, high):
        for _ in range(self._refine_steps):
            middle = ((low + high) // 2 // VA_ALIGNMENT) * VA_ALIGNMENT
            if middle <= low or middle >= high:
                break
            if self._reserve(middle):
                low = middle
            else:
                high = middle
        return low

    def _search_under(self, ceiling):
        """Largest grant strictly below ``ceiling``, which just failed.

        Starting below the ceiling rather than at it avoids repeating the
        failure we already paid for.
        """
        size = ceiling // 2
        while size >= MIN_PROBE_BYTES:
            if self._reserve(size):
                return self._refine(size, ceiling)
            ceiling, size = size, size // 2
        return 0

    def measure(self, ceiling):
        self.calls = 0
        target = self._watermark or ceiling
        if self._held >= self._reprobe_interval:
            # Retry the full ask periodically, so a recovery becomes visible.
            target = ceiling
        if self._reserve(target):
            self.kind = "ceiling" if target == ceiling else "hold"
            self._watermark = target
            self._held += 1
            return target
        self.kind = "drop"
        self._watermark = self._search_under(target)
        self._held = 0
        return self._watermark


def _query(fn, *args):
    """Return the driver's single result value, or None when the query fails."""
    err, *values = fn(*args)
    if err != driver.CUresult.CUDA_SUCCESS:
        return None
    return values[0]


def _query_bytes(fn, *args):
    """Like :func:`_query`, but unwraps the driver's cuuint64_t into an int."""
    value = _query(fn, *args)
    return None if value is None else int(value)


def _graph_mem(device):
    attrs = driver.CUgraphMem_attribute
    return (
        _query_bytes(driver.cuDeviceGetGraphMemAttribute, device, attrs.CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT),
        _query_bytes(driver.cuDeviceGetGraphMemAttribute, device, attrs.CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT),
    )


def _default_pool_mem(device):
    """Backing memory held by the device's default pool.

    Returns ``(None, None)`` once the address space is exhausted, because
    cuDeviceGetMemPool is itself one of the calls that starts failing then --
    which is the signal that turns a slow decline into the cascade.
    """
    pool = _query(driver.cuDeviceGetMemPool, device)
    if pool is None:
        return None, None
    attrs = driver.CUmemPool_attribute
    return (
        _query_bytes(driver.cuMemPoolGetAttribute, pool, attrs.CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT),
        _query_bytes(driver.cuMemPoolGetAttribute, pool, attrs.CU_MEMPOOL_ATTR_USED_MEM_CURRENT),
    )


class VaTracer:
    """Samples address-space headroom once per test into a CSV.

    Rows are flushed as they are written so a session that dies mid-run still
    leaves the trace up to that point.
    """

    def __init__(self, filename=DEFAULT_FILENAME):
        self._filename = filename
        self._probe = HeadroomProbe(refine_steps=refine_steps(), reprobe_interval=reprobe_interval())
        self._every = sample_every()
        self._path = None
        self._stream = None
        self._index = 0
        self._skipped = 0
        self._errors = 0
        self._ceiling = None
        self._probe_seconds = 0.0

    @property
    def path(self):
        return self._path

    @property
    def samples(self):
        return self._index

    @property
    def errors(self):
        return self._errors

    def open(self, directory):
        self._path = pathlib.Path(directory) / self._filename
        self._stream = self._path.open("w", encoding="utf-8", newline="")
        self._write_row(COLUMNS)

    def close(self):
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def _write_row(self, values):
        self._stream.write(",".join("" if v is None else str(v) for v in values) + "\n")
        self._stream.flush()

    def _probe_ceiling(self):
        """What a new pool would have to reserve. Read once; it cannot change."""
        if self._ceiling is None:
            err, _free, total = driver.cuMemGetInfo()
            if err != driver.CUresult.CUDA_SUCCESS:
                return DEFAULT_CEILING_BYTES
            self._ceiling = align_up(POOL_RESERVATION_MULTIPLE * int(total))
        return self._ceiling

    @property
    def probe_seconds(self):
        return self._probe_seconds

    def sample(self, nodeid, device_id):
        """Record one sample. Never raises: a broken probe must not fail a test."""
        if self._stream is None:
            return None
        if self._skipped + 1 < self._every:
            self._skipped += 1
            return None
        self._skipped = 0
        note = ""
        headroom = mem_free = graph_reserved = graph_used = pool_reserved = pool_used = None
        probe_ms = query_ms = None
        try:
            started = time.perf_counter()
            headroom = self._probe.measure(self._probe_ceiling())
            elapsed = time.perf_counter() - started
            self._probe_seconds += elapsed
            probe_ms = round(elapsed * 1000, 1)

            started = time.perf_counter()
            device = _query(driver.cuDeviceGet, device_id)
            mem_free = _query(driver.cuMemGetInfo)
            if device is not None:
                graph_reserved, graph_used = _graph_mem(device)
                pool_reserved, pool_used = _default_pool_mem(device)
            query_ms = round((time.perf_counter() - started) * 1000, 1)
        except Exception as exc:
            self._errors += 1
            note = repr(exc).replace(",", ";")
        self._index += 1
        self._write_row(
            (
                self._index,
                nodeid,
                headroom,
                mem_free,
                graph_reserved,
                graph_used,
                pool_reserved,
                pool_used,
                self._probe.kind,
                self._probe.calls,
                probe_ms,
                query_ms,
                note,
            )
        )
        return headroom


_tracer = VaTracer()
_current_nodeid = "<unknown>"


def configure(directory, tracer=None):
    """Open the trace file. No-op when tracing is disabled."""
    if not trace_enabled():
        return None
    tracer = _tracer if tracer is None else tracer
    tracer.open(directory)
    return tracer


def set_current_nodeid(nodeid):
    global _current_nodeid
    _current_nodeid = nodeid


def sample(device_id, tracer=None):
    """Sample at a test boundary, labelled with the test that just ran."""
    tracer = _tracer if tracer is None else tracer
    return tracer.sample(_current_nodeid, device_id)


def report_terminal_summary(terminalreporter, tracer=None):
    """Point at the trace from pytest's end-of-run summary.

    Returns the emitted line, or None when nothing was traced.
    """
    tracer = _tracer if tracer is None else tracer
    tracer.close()
    if tracer.path is None or not tracer.samples:
        return None
    line = f"{tracer.samples} address-space samples ({tracer.probe_seconds:.1f}s probing) written to {tracer.path}"
    if tracer.errors:
        line += f"; {tracer.errors} probe errors, see the errors column"
    terminalreporter.write_sep("=", "cuda_core address-space trace")
    terminalreporter.write_line(line)
    return line
