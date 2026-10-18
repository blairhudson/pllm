"""Shared bounded Responses lifecycle with final-only replicated token disclosure."""
from .delegated_client import DelegatedClient, _counter_fields, _DelegatedAudit
from .client import ProtocolError
from .replicated_protocol import ROLES, PEER_COUNTERS


class ReplicatedClient(DelegatedClient):
    runtime_id = "compiled_replicated_v1"
    role_names = ROLES
    counter_fields = _counter_fields(role_names) | PEER_COUNTERS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.audit = _DelegatedAudit(self._lock, self.counter_fields)

    def _ensure_source(self):
        from .replicated_source import admit_source
        with self._lock:
            self._check_open()
            if not self._source_checked:
                source = self._source or admit_source(self._declaration)
                source.runtime_contract(self._declaration, self._experiment_budget.max_input_tokens,
                                        self._experiment_budget.max_new_tokens)
                self._source, self._source_checked = source, True
            return self._source

    def _execute(self, source, ids, count, attempt):
        from .replicated_transport import execute_request
        return execute_request(self._declaration, source, ids, count,
                               self.role_connections, self.timeout, attempt)

    def _validate_result(self, result, source, ids, count):
        from .replicated_transport import ReplicatedResult
        if (type(result) is not ReplicatedResult or len(result.selected) != count
                or any(type(token) is not int or not 0 <= token < source.contract["vocabulary"]
                       for token in result.selected)):
            raise ProtocolError("replicated final output differs from admitted request")

    def list_models(self):
        source = self._ensure_source()
        return {"object": "list", "data": [{"id": self.default_model, "object": "model", "owned_by": "client",
            "context_length": source.source.manifest.context_length, "runtime": {
                "client_runtime": self.runtime_id, "privacy_mode": "replicated_public",
                "numeric_profile": source.contract["numeric_profile"], "generation_policy": "fixed_length_greedy",
                 "streaming": "buffered_final_ids", "state": "replicated_private_full_prefix", "kv_reuse": False,
                 "checkpoint_digest": source.checkpoint_digest,
                 "source_lock_digest": source.source_lock_digest, "tokenizer_digest": source.tokenizer_digest,
            }}]}
