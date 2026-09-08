from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class SecurityClaim:
    name: str
    input_confidentiality: bool
    output_confidentiality: bool
    model_confidentiality: Literal[
        "none",
        "honest client",
        "query limited",
        "malicious client",
    ]
    malicious_client_security: bool
    malicious_server_integrity: bool
    preprocessing_assumption: str
    reveals_intermediate_activations_to_client: bool
    output_disclosure: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


PUBLIC_WEIGHTS = SecurityClaim(
    name="public_weights",
    input_confidentiality=True,
    output_confidentiality=True,
    model_confidentiality="none",
    malicious_client_security=False,
    malicious_server_integrity=False,
    preprocessing_assumption="fresh correlations generated with HE",
    reveals_intermediate_activations_to_client=True,
    output_disclosure="client sees intermediate values and final logits",
)

PROPRIETARY_BLINDED = SecurityClaim(
    name="proprietary_blinded",
    input_confidentiality=True,
    output_confidentiality=True,
    model_confidentiality="honest client",
    malicious_client_security=False,
    malicious_server_integrity=False,
    preprocessing_assumption="fresh blinded correlations generated with HE",
    reveals_intermediate_activations_to_client=True,
    output_disclosure="client sees intermediate values and final logits",
)

PROPRIETARY_GUARDED = SecurityClaim(
    name="proprietary_guarded",
    input_confidentiality=True,
    output_confidentiality=True,
    model_confidentiality="query limited",
    malicious_client_security=False,
    malicious_server_integrity=False,
    preprocessing_assumption="fresh blinded correlations plus query limits",
    reveals_intermediate_activations_to_client=True,
    output_disclosure="client sees intermediate values and final logits",
)

SECURE_PREVIEW = SecurityClaim(
    name="secure_preview",
    input_confidentiality=True,
    output_confidentiality=True,
    model_confidentiality="malicious client",
    malicious_client_security=True,
    malicious_server_integrity=False,
    preprocessing_assumption=(
        "authenticated shares and multiplication material generated with HE; "
        "the model server follows the preprocessing protocol"
    ),
    reveals_intermediate_activations_to_client=False,
    output_disclosure="sampled token only by default, or final logits by explicit policy",
)

DIRECT_REFERENCE = SecurityClaim(
    name="direct_reference",
    input_confidentiality=True,
    output_confidentiality=True,
    model_confidentiality="honest client",
    malicious_client_security=False,
    malicious_server_integrity=False,
    preprocessing_assumption="client BFV key and public evaluation context",
    reveals_intermediate_activations_to_client=True,
    output_disclosure="client sees intermediate values and final logits",
)

PROFILES = {
    item.name: item
    for item in (
        PUBLIC_WEIGHTS,
        PROPRIETARY_BLINDED,
        PROPRIETARY_GUARDED,
        SECURE_PREVIEW,
        DIRECT_REFERENCE,
    )
}
