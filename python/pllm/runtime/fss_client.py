"""Bounded Responses lifecycle with final-only two-holder FSS disclosure."""
from .client import ProtocolError
from .delegated_client import DelegatedClient, _counter_fields, _DelegatedAudit
from .fss_protocol import HOLDERS, PEER_COUNTERS, ROLES


_STORAGE = frozenset(f"fss_storage.{role}.{direction}_bytes"
                     for role in HOLDERS for direction in ("read", "write"))


class _FssAudit(_DelegatedAudit):
    def to_dict(self):
        with self._lock:
            result = super().to_dict()
            result.update({key: value for key, value in self._totals.items()
                           if key in _STORAGE and value is not None})
            return result


class FssClient(DelegatedClient):
    runtime_id = "compiled_fss_v1"
    role_names = ROLES
    counter_fields = _counter_fields(role_names) | PEER_COUNTERS | _STORAGE

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.timeout > 600:
            raise ValueError("FuseFSS request lifetime exceeds 600 seconds")
        self.audit = _FssAudit(self._lock, self.counter_fields)
        self._completed_program = None

    def _ensure_source(self):
        from .fss_source import FrozenFssSource, resolve_client_source
        with self._lock:
            self._check_open()
            if not self._source_checked:
                source = self._source or resolve_client_source(self._declaration)
                if not isinstance(source, FrozenFssSource):
                    raise TypeError("FuseFSS requires its admitted source metadata")
                source.runtime_contract(self._declaration, self._experiment_budget.max_input_tokens,
                                        self._experiment_budget.max_new_tokens)
                self._source, self._source_checked = source, True
            return self._source

    def _execute(self, source, ids, count, attempt):
        from .fss_transport import execute_request
        self._completed_program = None
        def completed(manifest):
            self._completed_program = manifest
        return execute_request(self._declaration, source, ids, count,
                               self.role_connections, self.timeout, attempt, completed_program=completed)

    def _validate_result(self, result, source, ids, count):
        from .fss_transport import FssResult
        if (type(result) is not FssResult or len(result.selected) != count
                or any(type(token) is not int or not 0 <= token < source.contract["vocabulary"]
                       for token in result.selected)):
            raise ProtocolError("FuseFSS final output differs from the admitted request")

    def list_models(self):
        source = self._ensure_source()
        return {"object": "list", "data": [{"id": self.default_model, "object": "model",
            "owned_by": "client", "context_length": source.contract["maximum_sequence"], "runtime": {
                "client_runtime": self.runtime_id, "privacy_mode": "fss_public",
                "numeric_profile": source.contract["numeric_profile"],
                "generation_policy": "fixed_length_greedy", "streaming": "buffered_final_ids",
                "state": "additive_private_full_prefix", "kv_reuse": False, "verification": None,
                "checkpoint_digest": source.checkpoint_digest, "source_lock_digest": source.source_lock_digest,
                "tokenizer_digest": source.tokenizer_digest,
            }}]}
