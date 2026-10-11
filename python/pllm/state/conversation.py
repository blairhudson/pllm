"""Trusted heads for explicit client-owned conversation checkpoints."""
from __future__ import annotations

from dataclasses import dataclass

from .checkpoint import CheckpointHead


@dataclass(frozen=True, slots=True)
class ConversationCheckpoint:
    """A local encrypted generation's commitment; retain it alongside your key.

    The caller supplies the newest trusted head on restoration. This record is
    neither a provider credential nor an anti-rollback/key-custody service.
    """

    state: CheckpointHead
    name: str
    digest: str
    bytes: int

    def __post_init__(self):
        if (type(self.state) is not CheckpointHead or self.state.storage != "context_graph"
                or self.state.revision != 0):
            raise ValueError("invalid conversation checkpoint state head")
        CheckpointHead("context_graph", self.state.page_rows, 0, self.name, self.digest, self.bytes)

    def to_spec(self) -> dict:
        return {"schema": "pllm.conversation_checkpoint.v1", "state": self.state.to_spec(),
                "name": self.name, "digest": self.digest, "bytes": self.bytes}

    @classmethod
    def from_spec(cls, spec: dict) -> ConversationCheckpoint:
        if (type(spec) is not dict or set(spec) != {"schema", "state", "name", "digest", "bytes"}
                or spec["schema"] != "pllm.conversation_checkpoint.v1"):
            raise ValueError("invalid conversation checkpoint document")
        return cls(CheckpointHead.from_spec(spec["state"]), spec["name"], spec["digest"], spec["bytes"])
