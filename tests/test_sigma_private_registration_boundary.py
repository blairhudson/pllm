"""Private input/role prerequisites do not qualify public Python execution.

Current Rust source checks complement namespace inspection of an installed native
module; inspecting an older wheel alone cannot establish current registrations.
No native constructor, parser, capability probe or execution is called here.
"""

from __future__ import annotations

from pathlib import Path
import re

import pytest

from pllm import _native
from pllm.components import planned_component
from pllm.configuration import ConfigurationError
from pllm.providers import discover_providers

from test_providers import provider_fixture


ROOT = Path(__file__).resolve().parents[1]
BINDINGS = ROOT / "crates/pllm-python/src"
PRIVATE_MODULES = (
    "owned_raw_input",
    "sigma_linear_activation",
    "sigma_linear_activation_admission",
)
PRIVATE_ROLE_NAMES = {
    "_SigmaLocalCoordinator",
    "_SigmaLocalRole",
    "_SigmaLocalFrame",
    "_SigmaLocalOutput",
}


def test_current_binding_source_keeps_raw_input_and_sigma_registration_closed() -> None:
    source = (BINDINGS / "lib.rs").read_text(encoding="utf-8")
    for name in PRIVATE_MODULES:
        assert re.search(rf"^mod {name};$", source, flags=re.MULTILINE)
        # A private module declaration is not itself a constructor registration.
        # Reject newly added imports/qualified forwarding in the entrypoint file.
        assert len(re.findall(rf"\b{name}\b", source)) == 1

    _, separator, registration = source.partition("#[pymodule]")
    assert separator and "fn _native(" in registration
    # Positive controls: inspect the real existing module registration, not an
    # empty/mislocated source fragment that would trivially lack private types.
    assert "module.add_class::<CompiledPlan>()?;" in registration
    assert "wrap_pyfunction!(compile_plan, module)" in registration
    assert not any(name in registration for name in PRIVATE_ROLE_NAMES)

    raw = (BINDINGS / "owned_raw_input.rs").read_text(encoding="utf-8")
    assert "pub(crate) fn from_exact_bytes(" in raw
    assert "#[pyclass" not in raw
    assert "#[pyfunction" not in raw

    roles = (BINDINGS / "sigma_linear_activation.rs").read_text(encoding="utf-8")
    names = set(re.findall(r'#\[pyclass\([^\]]*name = "([^"]+)"', roles))
    assert names == PRIVATE_ROLE_NAMES
    admission = (BINDINGS / "sigma_linear_activation_admission.rs").read_text(
        encoding="utf-8"
    )
    assert re.search(
        r"fn require_python_allocation_contract\(\)\s*->\s*Result<\(\), BridgeError>"
        r"\s*\{\s*Err\(BridgeError::UnadmittedPython\)\s*\}",
        admission,
    )


def test_installed_native_namespace_has_no_private_role_or_raw_value_constructor() -> None:
    # Inspect only. Calling an existing general compiler would execute native
    # code, and fabricating a raw-Value Python method would not test a real route.
    namespace = vars(_native)
    assert "CompiledPlan" in namespace and "compile_plan" in namespace
    assert PRIVATE_ROLE_NAMES.isdisjoint(namespace)
    assert {"FrozenInput", "ImportedValue", "ValueQuote"}.isdisjoint(namespace)


@pytest.mark.parametrize("method", ["", "dealer-fss-keys", "shared-nonlinear"])
def test_provider_cannot_publish_sigma_before_importing_any_factory(
    tmp_path: Path, method: str
) -> None:
    plan = planned_component("sigma", method)
    entry_point, _manifest, _path, _package, _distribution = provider_fixture(
        tmp_path, component=plan.identity
    )
    with pytest.raises(ConfigurationError, match="sigma"):
        discover_providers(entry_points=[entry_point])
    assert entry_point.loaded is False
