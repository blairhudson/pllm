from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

import pllm
from pllm.components import create_component, get, list_component_classes
from pllm.kernels import KernelBackend
from pllm.profiles import MaskedLinearCpu
from pllm.providers import (
    ProviderDescriptor,
    ProviderDiscoveryError,
    ProviderResource,
    discover_providers,
    load_component_factory,
    provider_component_classes,
)

ROOT = Path(__file__).resolve().parents[1]


class FakeDistribution:
    def __init__(
        self,
        root: Path,
        files: list[str],
        *,
        name: str = "Acme_Provider",
        version: str = "2.0.0",
        editable: bool = False,
    ) -> None:
        self.root = root
        self.files = tuple(PurePosixPath(item) for item in files)
        self.metadata = {"Name": name}
        self.version = version
        self.editable = editable

    def locate_file(self, item) -> Path:
        return self.root / str(item)

    def read_text(self, name: str) -> str | None:
        if name != "direct_url.json":
            return None
        return json.dumps({"url": self.root.as_uri(), "dir_info": {"editable": self.editable}})


class FakeEntryPoint:
    def __init__(self, name: str, value: str, distribution: FakeDistribution) -> None:
        self.name = name
        self.value = value
        self.dist = distribution
        self.loaded = False

    def load(self):
        self.loaded = True
        raise AssertionError("provider entry-point code executed during discovery")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provider_fixture(
    tmp_path: Path,
    *,
    package: str = "acme_provider",
    provider: str = "org.example",
    component: str = "org.example/kernel",
    editable: bool = False,
):
    distribution_root = tmp_path / f"dist-{package}"
    package_root = distribution_root / package
    package_root.mkdir(parents=True)
    schema_path = package_root / "schemas" / "kernel.schema.json"
    schema_path.parent.mkdir()
    schema_path.write_text('{"type":"object"}\n', encoding="utf-8")
    docs_path = package_root / "docs" / "kernel.md"
    docs_path.parent.mkdir()
    docs_path.write_text("# Kernel\n", encoding="utf-8")
    manifest = {
        "schema": "pllm.provider_manifest.v1",
        "provider": provider,
        "distribution": "acme-provider",
        "version": "2.0.0",
        "host_versions": ["1"],
        "category_versions": {"pllm/kernel-backend": ["1"]},
        "components": [
            {
                "component": component,
                "provider": provider,
                "distribution": "acme-provider",
                "version": "7",
                "category": "pllm/kernel-backend",
                "category_version": "1",
                "lifecycle_phase": "compilation",
                "parameter_schema": {
                    "type": "object",
                    "properties": {"limit": {"type": "integer", "minimum": 1}},
                    "required": ["limit"],
                    "additionalProperties": False,
                },
                "capabilities": ["external-cpu"],
                "required_host_features": [],
                "role_eligibility": ["inference"],
                "artifacts": [],
                "evidence": [],
            }
        ],
        "schemas": [{"path": "schemas/kernel.schema.json", "sha256": _sha(schema_path)}],
        "docs": [{"path": "docs/kernel.md", "sha256": _sha(docs_path)}],
        "native_artifacts": [],
        "python_factories": {component: f"{package}.factory:create"},
        "tested_builds": [
            {
                "python": "3.13",
                "platform": "macos",
                "architecture": "arm64",
                "host_version": "1",
            }
        ],
    }
    manifest_path = package_root / "pllm-plugin.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    files = [
        f"{package}/pllm-plugin.json",
        f"{package}/schemas/kernel.schema.json",
        f"{package}/docs/kernel.md",
    ]
    distribution = FakeDistribution(distribution_root, files, editable=editable)
    entry_point = FakeEntryPoint(provider, f"{package}.provider:metadata", distribution)
    return entry_point, manifest, manifest_path, package_root, distribution


def _write_manifest(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def test_provider_schema_is_packaged_from_one_canonical_source() -> None:
    source = ROOT / "schemas/provider-manifest.schema.json"
    packaged = ROOT / "python/pllm/providers/provider-manifest.schema.json"
    assert source.read_bytes() == packaged.read_bytes()
    assert pllm.ProviderDescriptor is ProviderDescriptor
    assert pllm.ProviderDiscoveryError is ProviderDiscoveryError
    assert pllm.ProviderResource is ProviderResource
    assert pllm.discover_providers is discover_providers
    assert pllm.load_component_factory is load_component_factory
    Draft202012Validator.check_schema(json.loads(source.read_text(encoding="utf-8")))


def test_default_discovery_selects_only_versioned_provider_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected: list[str] = []

    class EntryPoints:
        @staticmethod
        def select(*, group: str):
            selected.append(group)
            return ()

    monkeypatch.setattr("pllm.providers.importlib.metadata.entry_points", EntryPoints)
    assert discover_providers() == ()
    assert selected == ["pllm.providers.v1"]


def test_valid_discovery_is_inert_deterministic_and_registry_aware(tmp_path: Path) -> None:
    entry_point, _manifest, _path, _package, _distribution = provider_fixture(tmp_path)
    providers = discover_providers(entry_points=[entry_point])
    assert entry_point.loaded is False
    assert len(providers) == 1
    provider = providers[0]
    assert provider.provider == "org.example"
    assert provider.version == "2.0.0"
    assert provider.components[0].version == "7"
    assert provider.components[0].component == "org.example/kernel"
    assert len(provider.manifest_digest) == 64
    repeated = discover_providers(entry_points=[entry_point])[0]
    assert provider == repeated
    assert hash(provider) == hash(repeated)
    assert provider.to_dict() == repeated.to_dict()
    assert str(tmp_path) not in json.dumps(provider.to_dict(), sort_keys=True)
    with pytest.raises(TypeError):
        provider.python_factories["other"] = "other:create"
    with pytest.raises(TypeError):
        provider.category_versions["other"] = ("1",)

    component_class = provider.component_classes[0]
    assert issubclass(component_class, KernelBackend)
    assert provider_component_classes(providers) == (component_class,)
    assert get("org.example/kernel", providers=providers) is component_class
    assert component_class.describe() is provider.components[0]
    component = create_component("org.example/kernel", {"limit": 2}, providers=providers)
    assert type(component) is component_class
    assert component.get_params() == {"limit": 2}
    assert component.with_params(limit=3).get_params() == {"limit": 3}
    pipeline = MaskedLinearCpu(pllm.Model("org/model"), kernels=component)
    assert pipeline.kernels is component
    experiment = pllm.Experiment(
        name="external-kernel",
        pipeline=pipeline,
        deployment=pllm.Deployment.local(root="local://external-kernel"),
        budget=pllm.ExecutionBudget(
            requests=1,
            max_input_tokens=1,
            max_new_tokens=1,
        ),
    )
    restored = pllm.Experiment.from_spec(experiment.to_spec(), providers=providers)
    assert type(restored.pipeline.kernels) is component_class
    assert restored == experiment
    assert pllm.canonical_bytes(experiment.to_spec(), providers=providers) == experiment.canonical_bytes()
    with pytest.raises(pllm.ConfigurationError):
        pllm.Experiment.from_spec(experiment.to_spec())
    with pytest.raises(pllm.ConfigurationError):
        experiment.resolve()
    with pytest.raises(pllm.ConfigurationError, match="invalid|integer"):
        component_class(limit=0)
    assert len(list_component_classes()) == 20
    assert len(list_component_classes(providers=providers)) == 21


def test_discovery_rejects_duplicate_keys_and_unknown_fields(tmp_path: Path) -> None:
    entry_point, manifest, path, _package, _distribution = provider_fixture(tmp_path)
    text = json.dumps(manifest)
    path.write_text(text[:-1] + ',"provider":"other"}', encoding="utf-8")
    with pytest.raises(ProviderDiscoveryError, match="duplicate key"):
        discover_providers(entry_points=[entry_point])

    _write_manifest(path, {**manifest, "unknown": True})
    with pytest.raises(ProviderDiscoveryError, match="invalid"):
        discover_providers(entry_points=[entry_point])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda ep, value: value.update(provider="org.other"), "entry point"),
        (lambda ep, value: value.update(distribution="other"), "distribution"),
        (lambda ep, value: value.update(version="3.0.0"), "version"),
        (lambda ep, value: value.update(host_versions=["2"]), "host version"),
        (
            lambda ep, value: value["components"][0].update(category_version="2"),
            "category version",
        ),
        (
            lambda ep, value: value["python_factories"].update(
                {"org.example/kernel": "other.factory:create"}
            ),
            "outside",
        ),
        (
            lambda ep, value: value["components"][0].update(
                parameter_schema={"type": "unknown"}
            ),
            "parameter schema",
        ),
        (
            lambda ep, value: value["components"][0].update(
                artifacts=[{"path": "missing.bin", "sha256": "0" * 64}]
            ),
            "unverified artifact",
        ),
    ],
)
def test_discovery_rejects_identity_and_version_mismatches(
    tmp_path: Path, mutation, message: str
) -> None:
    entry_point, manifest, path, _package, _distribution = provider_fixture(tmp_path)
    mutation(entry_point, manifest)
    _write_manifest(path, manifest)
    with pytest.raises(ProviderDiscoveryError, match=message):
        discover_providers(entry_points=[entry_point])


def test_editable_provider_requires_explicit_opt_in(tmp_path: Path) -> None:
    entry_point, _manifest, _path, _package, _distribution = provider_fixture(
        tmp_path, editable=True
    )
    with pytest.raises(ProviderDiscoveryError, match="explicit opt-in"):
        discover_providers(entry_points=[entry_point])
    provider = discover_providers(entry_points=[entry_point], allow_editable=True)[0]
    assert provider.editable is True


@pytest.mark.parametrize("mode", ["traversal", "unlisted", "hash", "symlink", "oversize"])
def test_resource_validation_fails_closed(tmp_path: Path, mode: str) -> None:
    entry_point, manifest, path, package, distribution = provider_fixture(tmp_path)
    resource = manifest["schemas"][0]
    if mode == "traversal":
        resource["path"] = "../escape.json"
    elif mode == "unlisted":
        extra = package / "schemas" / "other.json"
        extra.write_text("{}", encoding="utf-8")
        resource.update(path="schemas/other.json", sha256=_sha(extra))
    elif mode == "hash":
        resource["sha256"] = "0" * 64
    elif mode == "symlink":
        outside = tmp_path / "outside.json"
        outside.write_text("{}", encoding="utf-8")
        link = package / "schemas" / "link.json"
        link.symlink_to(outside)
        resource.update(path="schemas/link.json", sha256=_sha(outside))
        distribution.files += (PurePosixPath("acme_provider/schemas/link.json"),)
    else:
        large = package / "schemas" / "large.json"
        with large.open("wb") as stream:
            stream.seek((16 << 20) + 1)
            stream.write(b"0")
        resource.update(path="schemas/large.json", sha256=_sha(large))
        distribution.files += (PurePosixPath("acme_provider/schemas/large.json"),)
    _write_manifest(path, manifest)
    with pytest.raises(ProviderDiscoveryError):
        discover_providers(entry_points=[entry_point])


def test_native_artifact_requires_supported_abi(tmp_path: Path) -> None:
    entry_point, manifest, path, package, distribution = provider_fixture(tmp_path)
    artifact = package / "native.so"
    artifact.write_bytes(b"fixture")
    manifest["native_artifacts"] = [
        {
            "path": "native.so",
            "sha256": _sha(artifact),
            "abi_version": "1",
            "targets": ["test-target"],
        }
    ]
    distribution.files += (PurePosixPath("acme_provider/native.so"),)
    _write_manifest(path, manifest)
    discovered = discover_providers(entry_points=[entry_point])[0]
    assert discovered.native_artifacts[0].abi_version == "1"
    manifest["native_artifacts"][0]["abi_version"] = "2"
    _write_manifest(path, manifest)
    with pytest.raises(ProviderDiscoveryError, match="unsupported native plugin ABI"):
        discover_providers(entry_points=[entry_point])


def test_duplicate_component_identities_fail_across_providers_and_builtins(tmp_path: Path) -> None:
    first, _manifest, _path, _package, _distribution = provider_fixture(tmp_path / "a")
    second, second_manifest, second_path, _package, second_distribution = provider_fixture(
        tmp_path / "b", package="other_provider", provider="org.other"
    )
    second_manifest["distribution"] = "other-provider"
    second_manifest["components"][0]["distribution"] = "other-provider"
    second_distribution.metadata["Name"] = "Other_Provider"
    _write_manifest(second_path, second_manifest)
    with pytest.raises(ProviderDiscoveryError, match="duplicate"):
        discover_providers(entry_points=[first, second])

    builtin, manifest, path, _package, _distribution = provider_fixture(tmp_path / "c")
    manifest["components"][0]["component"] = "pllm/cpu"
    manifest["python_factories"] = {"pllm/cpu": "acme_provider.factory:create"}
    _write_manifest(path, manifest)
    provider = discover_providers(entry_points=[builtin])[0]
    with pytest.raises(ProviderDiscoveryError, match="collision"):
        provider_component_classes([provider])


def test_factory_loading_requires_explicit_provider_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry_point, _manifest, _path, _package, _distribution = provider_fixture(tmp_path)
    provider = discover_providers(entry_points=[entry_point])[0]
    imported: list[str] = []
    factory = object()

    def importer(name: str):
        imported.append(name)
        return SimpleNamespace(create=factory)

    monkeypatch.setattr("pllm.providers.importlib.import_module", importer)
    with pytest.raises(ProviderDiscoveryError, match="not approved"):
        load_component_factory(provider, "org.example/kernel", approved_providers=())
    assert imported == []
    assert (
        load_component_factory(
            provider,
            "org.example/kernel",
            approved_providers={"org.example"},
        )
        is factory
    )
    assert imported == ["acme_provider.factory"]
