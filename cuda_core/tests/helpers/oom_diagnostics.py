# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Machine-state capture for the first CUDA OOM of a test session (issue #2381).

A failing ``cuda_core`` run reports ~190 ``CUDA_ERROR_OUT_OF_MEMORY`` failures
that all descend from a single earlier event, so only the first is worth
capturing; the probes below deliberately sleep and reserve address space, which
would add minutes and bury the log if repeated. Hence the latch in
:class:`OomDiagnosticsRecorder`.
"""

import os
import pathlib
import sys
import threading
import time

from cuda.bindings import driver

OOM_MARKER = "CUDA_ERROR_OUT_OF_MEMORY"
DEFAULT_FILENAME = "cuda_core_oom_diagnostics.txt"

# cuMemAddressReserve wants a power-of-two alignment; the driver log in nvbug
# 5815123 shows the failing pool reservation using 2 MiB.
VA_ALIGNMENT = 2 * 1024 * 1024
SMALL_VA_RESERVATION = 2 * 1024 * 1024
SMALL_POOL_MAX_SIZE = 2 * 1024 * 1024
# Escalating retries, cumulative 5s. A single long pause only answers whether
# the pool comes back; the steps also measure how long that takes, which is the
# latency any retry-based fix would have to tolerate.
RECOVERY_RETRY_DELAYS_SECONDS = (0.0, 0.1, 0.4, 1.0, 1.5, 2.0)

_BANNER = "=" * 78


def format_probe(label, fn, *args):
    """Call a driver API and render its raw result."""
    try:
        return f"{label} -> {fn(*args)!r}"
    except Exception as exc:  # diagnostics must never mask the original test failure
        return f"{label} -> <raised {exc!r}>"


def probe_va_reservation(label, size):
    """Reserve a virtual address range and release it again.

    Reserving costs address space but no memory, so this is a direct read of
    what the address space can still satisfy.
    """
    # Device memory sizes are not generally a multiple of the alignment, and
    # cuMemAddressReserve rejects sizes that are not with CUDA_ERROR_INVALID_VALUE.
    size = ((size + VA_ALIGNMENT - 1) // VA_ALIGNMENT) * VA_ALIGNMENT
    try:
        err, ptr = driver.cuMemAddressReserve(size, VA_ALIGNMENT, 0, 0)
    except Exception as exc:  # see format_probe
        return f"{label} ({size:#x}) -> <raised {exc!r}>"
    if err != driver.CUresult.CUDA_SUCCESS:
        return f"{label} ({size:#x}) -> {err!r}"
    released = format_probe("release", driver.cuMemAddressFree, ptr, size)
    return f"{label} ({size:#x}) -> {err!r}; {released}"


def probe_small_pool_create(ordinal):
    """Create and destroy a pool with an explicit small ``maxSize``.

    An explicit ``maxSize`` caps the pool's VA reservation, so this succeeding
    while the default pool fails means the address space is fragmented rather
    than exhausted.
    """
    label = f"cuMemPoolCreate(dev {ordinal}, maxSize={SMALL_POOL_MAX_SIZE:#x})"
    try:
        properties = driver.CUmemPoolProps()
        properties.allocType = driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        properties.handleTypes = driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE
        properties.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        properties.location.id = ordinal
        properties.maxSize = SMALL_POOL_MAX_SIZE
        err, pool = driver.cuMemPoolCreate(properties)
    except Exception as exc:  # see format_probe
        return f"{label} -> <raised {exc!r}>"
    if err != driver.CUresult.CUDA_SUCCESS:
        return f"{label} -> {err!r}"
    destroyed = format_probe("destroy", driver.cuMemPoolDestroy, pool)
    return f"{label} -> {err!r}; {destroyed}"


def probe_address_space(total_memory):
    """Separate an exhausted address space from a merely fragmented one.

    Each pool reserves roughly 2x device memory, and Windows MCDM caps the
    whole address space at 40 bits (1 TiB), so a couple of pools can consume it
    (nvbug 5387350). A small reservation succeeding while a pool-sized one
    fails points at contiguity rather than capacity.
    """
    lines = ["--- address space probes ---"]
    lines.append(probe_va_reservation("small reservation", SMALL_VA_RESERVATION))
    if total_memory is None:
        lines.append("pool-sized reservation -> <skipped: device memory size unknown>")
    else:
        lines.append(probe_va_reservation("pool-sized reservation", 2 * total_memory))
    return lines


def _default_pool_attempt(dev):
    """Look up the default pool. Returns ``(succeeded, rendered result)``."""
    try:
        err, _pool = driver.cuDeviceGetDefaultMemPool(dev)
    except Exception as exc:  # see format_probe
        return False, f"<raised {exc!r}>"
    return err == driver.CUresult.CUDA_SUCCESS, repr(err)


def probe_recovery(ordinal, dev):
    """Establish whether the default pool comes back, and what waiting costs.

    Synchronizing first separates a release gated on outstanding work, which a
    synchronous fix could handle without sleeping, from one owned by a driver
    background thread, where only elapsed time helps. The escalating retries
    then turn "it recovers eventually" into a latency.

    The probes above this one reserve and release address space themselves, so
    a latency measured here is an upper bound on what an otherwise idle process
    would see.
    """
    lines = ["--- default pool recovery ---"]
    label = f"cuDeviceGetDefaultMemPool(dev {ordinal})"

    lines.append(format_probe("cuCtxSynchronize()", driver.cuCtxSynchronize))
    recovered, rendered = _default_pool_attempt(dev)
    lines.append(f"{label} after cuCtxSynchronize -> {rendered}")
    if recovered:
        lines.append("=> recovered on synchronize: release is gated on outstanding work")
        return lines

    waited = 0.0
    for delay in RECOVERY_RETRY_DELAYS_SECONDS:
        time.sleep(delay)
        waited += delay
        recovered, rendered = _default_pool_attempt(dev)
        lines.append(f"{label} after {waited:.1f}s idle -> {rendered}")
        if recovered:
            lines.append(f"=> recovered after {waited:.1f}s idle: release is deferred, not gated on work")
            return lines

    lines.append(f"=> still unavailable after {waited:.1f}s idle: not a transient shortfall")
    return lines


def probe_driver_state():
    """Query the driver directly, bypassing cuda.core's error reporting.

    cuda.core surfaces handle-creation failures through a thread-local "last
    error" slot (see cuda/core/_cpp/DESIGN.md). Reading the driver directly
    shows whether the device is genuinely out of memory and whether the default
    mempool is actually unavailable, which is what distinguishes real
    exhaustion from a mempool setup failure.
    """
    lines = ["--- direct driver probe (bypasses cuda.core error reporting) ---"]
    lines.append(format_probe("cuCtxGetCurrent()", driver.cuCtxGetCurrent))

    total_memory = None
    try:
        err, free, total = driver.cuMemGetInfo()
        lines.append(f"cuMemGetInfo() -> ({err!r}, free={free}, total={total})")
        if err == driver.CUresult.CUDA_SUCCESS:
            total_memory = total
    except Exception as exc:  # see format_probe
        lines.append(f"cuMemGetInfo() -> <raised {exc!r}>")

    try:
        err, count = driver.cuDeviceGetCount()
    except Exception as exc:  # see format_probe
        lines.append(f"cuDeviceGetCount() -> <raised {exc!r}>")
        return "\n".join(lines)

    lines.append(f"cuDeviceGetCount() -> ({err!r}, {count})")
    if err != driver.CUresult.CUDA_SUCCESS:
        return "\n".join(lines)

    first_device = None
    first_pool_available = False
    for ordinal in range(count):
        try:
            err, dev = driver.cuDeviceGet(ordinal)
        except Exception as exc:  # see format_probe
            lines.append(f"cuDeviceGet({ordinal}) -> <raised {exc!r}>")
            continue
        if err != driver.CUresult.CUDA_SUCCESS:
            lines.append(f"cuDeviceGet({ordinal}) -> {err!r}")
            continue
        lines.append(format_probe(f"cuDeviceGetMemPool(dev {ordinal})", driver.cuDeviceGetMemPool, dev))
        available, rendered = _default_pool_attempt(dev)
        lines.append(f"cuDeviceGetDefaultMemPool(dev {ordinal}) -> {rendered}")
        if first_device is None:
            first_device = (ordinal, dev)
            first_pool_available = available
        lines.append(probe_small_pool_create(ordinal))

    lines.extend(probe_address_space(total_memory))

    # Last, so the probes above report state as it was at the moment of failure.
    if first_device is not None:
        if first_pool_available:
            # Recovery only means something if the pool was actually lost;
            # otherwise "recovered" would be read as a diagnosis of a failure
            # that happened somewhere else entirely.
            lines.append("--- default pool recovery ---")
            lines.append(f"not applicable: dev {first_device[0]} default pool was already available")
        else:
            lines.extend(probe_recovery(*first_device))

    return "\n".join(lines)


class OomDiagnosticsRecorder:
    """Captures machine state the first time a CUDA OOM is seen, and only then."""

    def __init__(self, filename=DEFAULT_FILENAME):
        self._filename = filename
        self._lock = threading.Lock()
        self._captured = False
        self._nodeid = None
        self._artifact_path = None
        self._artifact_written = False

    @property
    def captured(self):
        return self._captured

    @property
    def nodeid(self):
        """Node id of the test that triggered capture, or None."""
        return self._nodeid

    @property
    def artifact_path(self):
        """Where the report was written, or None if nothing was captured."""
        return self._artifact_path

    @property
    def artifact_written(self):
        return self._artifact_written

    @staticmethod
    def matches(exc_text):
        return OOM_MARKER in exc_text

    def build_report(self, nodeid, phase, exc_text):
        return "\n".join(
            [
                _BANNER,
                "cuda_core diagnostics: first CUDA_ERROR_OUT_OF_MEMORY of this session",
                _BANNER,
                f"test:      {nodeid}",
                f"phase:     {phase}",
                f"pid:       {os.getpid()}",
                f"platform:  {sys.platform}",
                f"exception: {exc_text}",
                "",
                probe_driver_state(),
                _BANNER,
            ]
        )

    def capture(self, nodeid, phase, exc_text, directory):
        """Build and persist the report. Returns None if already captured."""
        with self._lock:
            if self._captured:
                return None
            self._captured = True

        report = self.build_report(nodeid, phase, exc_text)
        destination = pathlib.Path(directory) / self._filename
        self._nodeid = nodeid
        self._artifact_path = destination
        try:
            destination.write_text(report, encoding="utf-8")
            self._artifact_written = True
            return f"{report}\n(diagnostics also written to {destination})"
        except OSError as exc:
            return f"{report}\n(could not write {destination}: {exc!r})"


_default_recorder = OomDiagnosticsRecorder()


def record_if_oom(item, call, report, recorder=None):
    """Capture diagnostics when ``report`` is the session's first CUDA OOM.

    ``recorder`` defaults to a module-level singleton so the conftest hook does
    not have to hold session state; tests pass their own to stay isolated.

    Returns the emitted text, or None when nothing was captured.
    """
    if recorder is None:
        recorder = _default_recorder

    if recorder.captured or not report.failed or call.excinfo is None:
        return None

    exc_text = str(call.excinfo.value)
    if not recorder.matches(exc_text):
        return None

    text = recorder.capture(item.nodeid, call.when, exc_text, item.config.rootpath)
    if text is None:
        return None

    # terminalreporter writes outside pytest's stdout capture, so this survives
    # into a redirected log; a bare print() would not.
    terminal_reporter = item.config.pluginmanager.get_plugin("terminalreporter")
    if terminal_reporter is not None:
        terminal_reporter.write_line("")
        terminal_reporter.write_line(text)
    return text


def report_terminal_summary(terminalreporter, recorder=None):
    """Point at the diagnostics artifact from pytest's end-of-run summary.

    The report itself is emitted beside the failing test, which in a real
    failing run is thousands of lines above the summary people actually read.

    Returns the emitted line, or None when nothing was captured.
    """
    if recorder is None:
        recorder = _default_recorder

    if not recorder.captured:
        return None

    verb = "written to" if recorder.artifact_written else "could NOT be written to"
    line = f"first CUDA OOM at {recorder.nodeid}; diagnostics {verb} {recorder.artifact_path}"
    terminalreporter.write_sep("=", "cuda_core OOM diagnostics", red=True)
    terminalreporter.write_line(line)
    return line
