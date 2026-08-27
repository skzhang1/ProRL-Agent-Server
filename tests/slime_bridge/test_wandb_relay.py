from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np

from slime_bridge import wandb_relay


def test_single_owner_flag_and_actor_name(monkeypatch) -> None:
    monkeypatch.setenv("SLIME_WANDB_SINGLE_OWNER", "true")
    monkeypatch.delenv("SLIME_WANDB_RELAY_NAME", raising=False)
    assert wandb_relay.single_owner_enabled()
    assert (
        wandb_relay.relay_actor_name(SimpleNamespace(wandb_run_id="hapo7/run 3ep"))
        == "slime-wandb-relay-hapo7-run-3ep"
    )


def test_log_via_relay_waits_for_writer(monkeypatch) -> None:
    calls = []

    class RemoteMethod:
        def remote(self, *args):
            calls.append(args)
            return "log-ref"

    handle = SimpleNamespace(log=RemoteMethod())

    class FakeRay:
        @staticmethod
        def get_actor(name):
            assert name == "slime-wandb-relay-test-run"
            return handle

        @staticmethod
        def get(ref):
            assert ref == "log-ref"
            return None

    monkeypatch.setitem(sys.modules, "ray", FakeRay)
    monkeypatch.delenv("SLIME_WANDB_RELAY_NAME", raising=False)
    monkeypatch.setattr(wandb_relay, "_cached_handle", None)
    monkeypatch.setattr(wandb_relay, "_cached_name", None)

    args = SimpleNamespace(wandb_run_id="test-run")
    wandb_relay.log_via_relay(args, {"polar/hapo/update_steps": 1.0}, "rollout/step")

    assert calls == [({"polar/hapo/update_steps": 1.0}, "rollout/step")]


def test_transport_values_detach_tensors_and_arrays(monkeypatch) -> None:
    class FakeTensor:
        def __init__(self, value):
            self.value = value

        def detach(self):
            return self

        def numel(self):
            return 1 if not isinstance(self.value, list) else len(self.value)

        def item(self):
            return self.value

        def cpu(self):
            return self

        def tolist(self):
            return self.value

    class FakeTorch:
        @staticmethod
        def is_tensor(value):
            return isinstance(value, FakeTensor)

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    value = {
        "scalar": FakeTensor(2.5),
        "vector": FakeTensor([1.0, 3.0]),
        "numpy_scalar": np.float32(4.0),
        "nested": [np.array([5, 6])],
    }

    assert wandb_relay._to_transport_value(value) == {
        "scalar": 2.5,
        "vector": [1.0, 3.0],
        "numpy_scalar": 4.0,
        "nested": [[5, 6]],
    }
