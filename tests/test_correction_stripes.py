import numpy as np
import pytest

from pllm.runtime.benchmark_correction_stripes import CorrectionStripeReceiver, CorrectionStripeSender


def pair(tmp_path, **options):
    contract = dict(key=b"k" * 32, context="b" * 64, rows=3, columns=11, stripe_columns=4, wire_bits=24)
    contract.update(options)
    return CorrectionStripeSender(**contract), CorrectionStripeReceiver(tmp_path / "receiver", **contract)


def test_stripes_require_all_acknowledgements_before_ready(tmp_path):
    sender, receiver = pair(tmp_path)
    expected = np.arange(33, dtype=np.uint32).reshape(3, 11)
    for column in range(0, 11, 4):
        with pytest.raises(ValueError, match="READY"):
            next(receiver.read())
        frame = sender.frame(expected[:, column:column + 4])
        sender.acknowledge(receiver.accept(frame))
    assert not receiver.ready
    sender.acknowledge(receiver.accept(sender.frame()))
    actual = np.concatenate([value for _, value in receiver.read()], axis=1)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("failure", ["replay", "wrong_context", "corruption", "premature_seal"])
def test_bad_stripes_burn_partial_owner(tmp_path, failure):
    from pllm.runtime.benchmark_correction_stripes import _authenticated
    sender, receiver = pair(tmp_path)
    frame = sender.frame(np.ones((3, 4), np.uint32))
    sender.acknowledge(receiver.accept(frame))
    if failure == "wrong_context":
        sender.context = "a" * 64
        frame = sender.frame(np.ones((3, 4), np.uint32))
    elif failure == "corruption":
        frame = frame[:-1] + bytes([frame[-1] ^ 1])
    elif failure == "premature_seal":
        frame = _authenticated(b"k" * 32, b"push", {"context": "b" * 64, "sequence": 1,
            "operation": "seal", "column": 4, "width": 0})
    with pytest.raises(ValueError):
        receiver.accept(frame)
    assert receiver.closed and not receiver.ready and not receiver.root.exists()


def test_substituted_acknowledgement_burns_sender(tmp_path):
    sender, receiver = pair(tmp_path)
    ack = receiver.accept(sender.frame(np.ones((3, 4), np.uint32)))
    sender.acknowledge(ack)
    sender.frame(np.ones((3, 4), np.uint32))
    with pytest.raises(ValueError):
        sender.acknowledge(ack)
    with pytest.raises(ValueError):
        sender.frame()
    receiver.cancel()
    assert not receiver.root.exists()


def test_unacknowledged_or_malformed_sender_burns_remaining_material(tmp_path):
    sender, receiver = pair(tmp_path)
    sender.frame(np.ones((3, 4), np.uint32))
    with pytest.raises(ValueError, match="not ready"):
        sender.frame(np.ones((3, 4), np.uint32))
    assert sender.closed and not sender.key and sender.pending is None
    receiver.cancel()
    sender, receiver = pair(tmp_path)
    with pytest.raises(ValueError, match="invalid"):
        sender.frame(np.ones((3, 4), np.float32))
    assert sender.closed and not sender.key
    receiver.cancel()


def test_pipe_ack_failure_retires_child_and_incomplete_files(tmp_path, monkeypatch):
    import subprocess
    import pllm
    from pllm._benchmark_worker import Measurement
    from pllm.metrics import BenchmarkTarget, Sampling
    from pllm.profiles import MaskedLinearCpu
    from pllm.runtime.benchmark_targets import execute_target
    experiment = pllm.Experiment("stripes", MaskedLinearCpu(pllm.Model.tiny()),
        pllm.Deployment.local(root=str(tmp_path)), pllm.ExecutionBudget(1, 16, 2))
    children = []
    spawn = subprocess.Popen
    def start(*args, **kwargs):
        process = spawn(*args, **kwargs)
        children.append(process)
        return process
    def reject(*args):
        raise ValueError("injected acknowledgement failure")
    monkeypatch.setattr(subprocess, "Popen", start)
    monkeypatch.setattr(CorrectionStripeSender, "acknowledge", reject)
    m = Measurement(dict(sampling=Sampling(processes=1).to_spec(), salt="aa" * 32,
                         inputs={}, directory=str(tmp_path)))
    with pytest.raises(ValueError, match="injected acknowledgement"):
        execute_target(BenchmarkTarget.preparation(experiment, output_columns=3), m)
    assert children and all(child.poll() is not None for child in children)
    assert not list(tmp_path.glob("receiver-*"))


@pytest.mark.parametrize("window", [2, 4, 8])
def test_window_keeps_ordered_credit_and_requires_draining_before_seal(tmp_path, window):
    contract = dict(key=b"k" * 32, context="b" * 64, rows=3, columns=window * 4,
                    stripe_columns=4, wire_bits=32)
    sender = CorrectionStripeSender(**contract, credit_window=window)
    receiver = CorrectionStripeReceiver(tmp_path / "receiver", **contract)
    expected = np.arange(3 * window * 4, dtype=np.uint32).reshape(3, window * 4)
    replies = [receiver.accept(sender.frame(expected[:, col:col + 4]))
               for col in range(0, window * 4, 4)]
    assert sender.outstanding == window and sender.column == 0 and not receiver.ready
    for reply in replies:
        sender.acknowledge(reply)
    assert sender.outstanding == 0 and sender.column == window * 4
    sender.acknowledge(receiver.accept(sender.frame()))
    np.testing.assert_array_equal(np.concatenate([v for _, v in receiver.read()], axis=1), expected)
    assert sender.closed and receiver.ready


@pytest.mark.parametrize("failure", ["out_of_order", "early_seal", "over_credit", "cancel"])
def test_window_failure_burns_all_pending_stripes(failure):
    sender = CorrectionStripeSender(key=b"k" * 32, context="b" * 64, rows=1, columns=12,
                                    stripe_columns=4, wire_bits=16, credit_window=2)
    from pllm.runtime.benchmark_correction_stripes import _authenticated
    first = sender.frame(np.ones((1, 4), np.uint32))
    second = sender.frame(np.ones((1, 4), np.uint32))
    import hashlib
    ack = _authenticated(b"k" * 32, b"ack", {"context": "b" * 64, "sequence": 1,
        "next_column": 8, "ready": False, "accepted_digest": hashlib.sha256(second).hexdigest()})
    if failure == "cancel":
        sender.cancel()
    else:
        with pytest.raises(ValueError):
            if failure == "out_of_order":
                sender.acknowledge(ack)
            elif failure == "early_seal":
                sender.frame()
            else:
                sender.frame(np.ones((1, 4), np.uint32))
    assert first != second and sender.closed and not sender.key and sender.outstanding == 0


def test_window_is_bounded_before_frames_and_requires_stripes():
    from pllm import Deployment, ExecutionBudget, Experiment, Model
    from pllm.metrics import BenchmarkTarget
    from pllm.profiles import MaskedLinearCpu
    for window in (0, 9, True):
        with pytest.raises(ValueError):
            CorrectionStripeSender(key=b"k" * 32, context="b" * 64, rows=1, columns=12,
                                   stripe_columns=4, wire_bits=16, credit_window=window)
    experiment = Experiment("test", MaskedLinearCpu(Model.tiny()), Deployment.local(root="local://credits"), ExecutionBudget(1, 16, 2))
    with pytest.raises(ValueError):
        BenchmarkTarget.preparation(experiment, credit_window=4)
    with pytest.raises(ValueError):
        BenchmarkTarget.preparation(experiment, output_columns=256, credit_window=9)
