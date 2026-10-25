"""Public envelopes shared by the replicated client and authenticated workers."""

ROLES = ("worker_a", "worker_b", "worker_c")
FRAME_LIMIT = 128 << 10
ADMISSION_LIMIT = 1 << 20
REQUEST_MEMORY = 192 << 20
STREAM_MEMORY = 8 << 20
PEER_COUNTERS = frozenset(
    f"peer_link.{source}.{destination}.{phase}_bytes"
    for source in ROLES for destination in ROLES if source != destination
    for phase in ("setup", "online")
)


def execution_policy(experiment):
    """Return the full native policy only from canonical Experiment parameters."""
    from pllm.protocols import PumaThreePartyDecoder

    component = experiment.pipeline.components["linear"]
    selected = PumaThreePartyDecoder.from_params(component.params)
    if component != selected:
        raise ValueError("noncanonical replicated execution parameters")
    params = selected.get_params()
    params.pop("weights")
    # Storage is bound by the canonical Experiment/composition, never the
    # ArithmeticExecutionPolicy or peer envelope.
    params.pop("storage")
    return params


def storage_policy(experiment):
    """Read storage only from the same validated canonical server/client component."""
    from pllm.protocols import PumaThreePartyDecoder

    execution_policy(experiment)
    return PumaThreePartyDecoder.from_params(experiment.pipeline.components["linear"].params).storage


def execution_binding(experiment):
    policy = execution_policy(experiment)
    return policy if policy != {"range_policy": "static", "max_abs_activation": 256,
                                "lookup": "per_token", "selection": "expanded", "peer_transport": "http"} else None


def worker_request_memory(experiment):
    return REQUEST_MEMORY + (STREAM_MEMORY if execution_policy(experiment)["peer_transport"] == "streams" else 0)


def validate_execution_binding(experiment, envelope):
    """Require exact canonical policy, including JSON integer/string types."""
    expected = execution_binding(experiment)
    actual = envelope.get("execution")
    if expected is None:
        valid = "execution" not in envelope
    else:
        valid = (type(actual) is dict and set(actual) == set(expected)
                 and all(type(actual[key]) is type(value) and actual[key] == value
                         for key, value in expected.items()))
    if not valid:
        raise ValueError("replicated execution binding differs from Experiment")


def validate_transport_selector(experiment, selector):
    """A supervisor hint may confirm the Experiment, never override it."""
    selected = execution_policy(experiment)["peer_transport"]
    if selector is not None and selector != selected:
        raise ValueError("replicated supervisor transport differs from Experiment")
    return selected


def input_frame(value):
    """Encode the native ordered-input header after bounded outer validation."""
    if (set(value) != {"key", "offset", "frame"} or type(value["key"]) is not str
            or type(value["offset"]) is not int or not 0 <= value["offset"] <= 1 << 20
            or type(value["frame"]) is not bytes or not 0 < len(value["frame"]) <= 64 << 10
            or len(value["frame"]) % 16):
        raise ValueError("invalid replicated source frame")
    key = value["key"].encode("utf-8")
    if not 0 < len(key) <= 512:
        raise ValueError("invalid replicated source key")
    return (len(key).to_bytes(2, "little") + key + value["offset"].to_bytes(4, "little")
            + (len(value["frame"]) // 16).to_bytes(4, "little") + value["frame"])
