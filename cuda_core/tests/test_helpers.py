# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import time
import types

import pytest
from helpers.buffers import PatternGen, compare_equal_buffers, make_scratch_buffer
from helpers.latch import LatchKernel
from helpers.logging import TimestampedLogger
from helpers.oom_diagnostics import (
    DEFAULT_FILENAME,
    OOM_MARKER,
    OomDiagnosticsRecorder,
    format_probe,
    probe_driver_state,
    probe_recovery,
    record_if_oom,
    report_terminal_summary,
)

from cuda.bindings import driver
from cuda.core import Device
from cuda_python_test_helpers import under_compute_sanitizer

ENABLE_LOGGING = False  # Set True for test debugging and development
NBYTES = 64


@pytest.mark.skipif(Device().compute_capability.major < 7, reason="__nanosleep is only available starting Volta (sm70)")
def test_latchkernel():
    """Test LatchKernel."""
    log = TimestampedLogger(enabled=ENABLE_LOGGING)
    log("begin")
    device = Device()
    device.set_current()
    stream = device.create_stream()
    target = make_scratch_buffer(device, 0, NBYTES)
    zeros = make_scratch_buffer(device, 0, NBYTES)
    ones = make_scratch_buffer(device, 1, NBYTES)
    latch = LatchKernel(device)
    log("launching latch kernel")
    latch.launch(stream)
    log("launching copy (0->1) kernel")
    target.copy_from(ones, stream=stream)
    log("going to sleep")
    time.sleep(1)
    if device.properties.concurrent_managed_access:
        # Host access to managed memory while a kernel is active is unsafe on
        # devices without concurrent managed access.
        log("checking target == 0")
        assert compare_equal_buffers(target, zeros)
    log("releasing latch and syncing")
    latch.release()
    stream.sync()
    log("checking target == 1")
    assert compare_equal_buffers(target, ones)
    log("done")


@pytest.mark.skipif(
    under_compute_sanitizer(),
    reason="Too slow under compute-sanitizer (UVM-heavy test).",
)
def test_patterngen_seeds():
    """Test PatternGen with seed argument."""
    device = Device()
    device.set_current()
    buffer = make_scratch_buffer(device, 0, NBYTES)

    # All seeds are pairwise different.
    # We test a sampling of values because exhaustive testing is too slow,
    # especially on Windows. See https://github.com/NVIDIA/cuda-python/issues/1455
    pgen = PatternGen(device, NBYTES)
    for i in (ii for ii in range(256) if ii < 5 or ii % 17 == 0):
        pgen.fill_buffer(buffer, seed=i)
        pgen.verify_buffer(buffer, seed=i)
        for j in (jj for jj in range(i + 1, 256) if jj < 5 or jj % 19 == 0):
            with pytest.raises(AssertionError):
                pgen.verify_buffer(buffer, seed=j)


def test_patterngen_values():
    """Test PatternGen with value argument, also compare_equal_buffers."""
    device = Device()
    device.set_current()
    ones = make_scratch_buffer(device, 1, NBYTES)
    twos = make_scratch_buffer(device, 2, NBYTES)
    assert compare_equal_buffers(ones, ones)
    assert not compare_equal_buffers(ones, twos)
    pgen = PatternGen(device, NBYTES)
    pgen.verify_buffer(ones, value=1)
    pgen.verify_buffer(twos, value=2)


# helpers.oom_diagnostics (issue #2381). These only ever run on a failing
# session, so no other test would notice if they silently broke.


def _fake_report(failed):
    return types.SimpleNamespace(failed=failed)


def _fake_call(exc_text, when="call"):
    excinfo = None if exc_text is None else types.SimpleNamespace(value=RuntimeError(exc_text))
    return types.SimpleNamespace(excinfo=excinfo, when=when)


def _fake_item(rootpath, nodeid="tests/test_x.py::test_a"):
    # get_plugin returns None so record_if_oom skips the terminal write.
    config = types.SimpleNamespace(
        rootpath=rootpath,
        pluginmanager=types.SimpleNamespace(get_plugin=lambda _: None),
    )
    return types.SimpleNamespace(nodeid=nodeid, config=config)


@pytest.fixture
def fast_oom_probes(monkeypatch):
    """Drop the recovery pauses, which only earn their cost on a real failure."""
    monkeypatch.setattr("helpers.oom_diagnostics.RECOVERY_RETRY_DELAYS_SECONDS", (0.0,))


@pytest.mark.agent_authored(model="claude-opus-5")
@pytest.mark.parametrize(
    ("failed", "exc_text", "should_capture"),
    [
        (True, f"boom {OOM_MARKER}", True),
        (True, "CUDA_ERROR_INVALID_CONTEXT", False),
        (False, f"boom {OOM_MARKER}", False),
        (True, None, False),
    ],
)
def test_oom_diagnostics_fire_only_on_a_failing_oom(tmp_path, fast_oom_probes, failed, exc_text, should_capture):
    recorder = OomDiagnosticsRecorder()
    result = record_if_oom(_fake_item(tmp_path), _fake_call(exc_text), _fake_report(failed), recorder=recorder)

    assert (result is not None) == should_capture
    assert recorder.captured == should_capture


@pytest.mark.agent_authored(model="claude-opus-5")
def test_oom_diagnostics_write_an_artifact_naming_the_failing_test(tmp_path, monkeypatch, fast_oom_probes):
    # Omitting `recorder` also pins the call signature conftest.py relies on.
    # Patching the singleton keeps this from latching diagnostics for the run.
    recorder = OomDiagnosticsRecorder()
    monkeypatch.setattr("helpers.oom_diagnostics._default_recorder", recorder)

    text = record_if_oom(
        _fake_item(tmp_path, nodeid="tests/test_x.py::test_first"),
        _fake_call(f"boom {OOM_MARKER}"),
        _fake_report(True),
    )

    written = (tmp_path / DEFAULT_FILENAME).read_text(encoding="utf-8")
    assert "tests/test_x.py::test_first" in written
    assert OOM_MARKER in written
    assert "tests/test_x.py::test_first" in text


@pytest.mark.agent_authored(model="claude-opus-5")
def test_oom_diagnostics_summary_points_at_the_artifact(tmp_path, fast_oom_probes):
    # The report is emitted beside the failing test, thousands of lines above
    # the summary; without this line the artifact is effectively invisible.
    recorder = OomDiagnosticsRecorder()
    lines = []
    reporter = types.SimpleNamespace(write_sep=lambda *_, **__: None, write_line=lines.append)

    assert report_terminal_summary(reporter, recorder=recorder) is None
    assert lines == []

    recorder.capture("tests/test_x.py::test_first", "call", OOM_MARKER, tmp_path)
    emitted = report_terminal_summary(reporter, recorder=recorder)

    assert "tests/test_x.py::test_first" in emitted
    assert str(tmp_path) in emitted
    assert emitted in lines


@pytest.mark.agent_authored(model="claude-opus-5")
def test_oom_diagnostics_latch_to_the_first_oom(tmp_path, fast_oom_probes):
    # A failing run produces ~190 OOMs; capturing each would bury the log.
    recorder = OomDiagnosticsRecorder()
    assert recorder.capture("first", "call", OOM_MARKER, tmp_path) is not None
    assert recorder.captured
    assert recorder.capture("second", "call", OOM_MARKER, tmp_path) is None

    assert "first" in (tmp_path / DEFAULT_FILENAME).read_text(encoding="utf-8")


@pytest.mark.agent_authored(model="claude-opus-5")
def test_oom_diagnostics_record_probe_failures_instead_of_raising():
    # Raising inside the pytest hook would mask the failure being diagnosed.
    def _explode():
        raise RuntimeError("driver is wedged")

    assert "driver is wedged" in format_probe("cuSomething()", _explode)


@pytest.mark.agent_authored(model="claude-opus-5")
def test_oom_diagnostics_probe_reports_live_driver_state(init_cuda, fast_oom_probes):
    text = probe_driver_state()
    assert "cuMemGetInfo()" in text
    assert "cuDeviceGetMemPool" in text
    # A healthy device must report a usable default mempool.
    assert "CUDA_SUCCESS" in text


@pytest.mark.agent_authored(model="claude-opus-5")
def test_oom_diagnostics_probe_exercises_pools_and_address_space(init_cuda, fast_oom_probes):
    # These probes are what separate an exhausted address space from a merely
    # fragmented one, so losing them would leave issue #2381 undiagnosable.
    text = probe_driver_state()
    assert "cuMemPoolCreate(dev 0" in text
    assert "small reservation" in text
    assert "pool-sized reservation" in text
    assert "--- default pool recovery ---" in text
    # A healthy device never lost the pool, so claiming it "recovered" would
    # read as a diagnosis of a failure that happened somewhere else.
    assert "not applicable" in text


@pytest.mark.agent_authored(model="claude-opus-5")
def test_oom_diagnostics_probes_release_what_they_reserve(init_cuda, fast_oom_probes):
    # A probe that leaked its reservation would corrupt the state it reports on.
    text = probe_driver_state()
    for line in text.splitlines():
        if "reservation" in line and "CUDA_SUCCESS" in line:
            assert "release" in line
        if line.startswith("cuMemPoolCreate(dev ") and "CUDA_SUCCESS" in line:
            assert "destroy" in line


def _device_zero():
    err, dev = driver.cuDeviceGet(0)
    assert err == driver.CUresult.CUDA_SUCCESS
    return dev


# A healthy device always recovers on the first attempt, so the remaining
# branches are only reachable by forcing the lookup to fail.


@pytest.mark.agent_authored(model="claude-opus-5")
def test_oom_diagnostics_recovery_distinguishes_a_synchronize_fix(init_cuda, monkeypatch):
    # Recovering on synchronize means the release is gated on outstanding work,
    # so a deterministic fix is possible. That is a materially different
    # conclusion from needing to wait, and the report must not conflate them.
    monkeypatch.setattr("helpers.oom_diagnostics._default_pool_attempt", lambda _dev: (True, "CUDA_SUCCESS"))

    text = "\n".join(probe_recovery(0, _device_zero()))

    assert "recovered on synchronize" in text
    assert "s idle ->" not in text


@pytest.mark.agent_authored(model="claude-opus-5")
def test_oom_diagnostics_recovery_reports_the_wait_that_worked(init_cuda, monkeypatch):
    # The latency is the point of the probe; reporting bare recovery would
    # leave a retry-based fix un-sizable.
    results = iter([(False, "CUDA_ERROR_OUT_OF_MEMORY"), (False, "CUDA_ERROR_OUT_OF_MEMORY"), (True, "CUDA_SUCCESS")])
    monkeypatch.setattr("helpers.oom_diagnostics.RECOVERY_RETRY_DELAYS_SECONDS", (0.0, 0.1))
    monkeypatch.setattr("helpers.oom_diagnostics._default_pool_attempt", lambda _dev: next(results))

    text = "\n".join(probe_recovery(0, _device_zero()))

    assert "recovered after 0.1s idle" in text
    assert "not a transient shortfall" not in text


@pytest.mark.agent_authored(model="claude-opus-5")
def test_oom_diagnostics_recovery_escalates_then_gives_up(init_cuda, monkeypatch):
    # Exhausting the retries has to be distinguishable from recovering, or a
    # structural failure reads as a transient one.
    monkeypatch.setattr("helpers.oom_diagnostics.RECOVERY_RETRY_DELAYS_SECONDS", (0.0, 0.0))
    monkeypatch.setattr(
        "helpers.oom_diagnostics._default_pool_attempt",
        lambda _dev: (False, "CUDA_ERROR_OUT_OF_MEMORY"),
    )

    text = "\n".join(probe_recovery(0, _device_zero()))

    assert text.count("s idle ->") == 2
    assert "not a transient shortfall" in text
