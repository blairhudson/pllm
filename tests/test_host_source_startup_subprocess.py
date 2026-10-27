"""Real native startup imports in isolated children; matching Root wheel required.

Run future qualification from an intended venv OUTSIDE the checkout with that
wheel installed. Missing/wrong provenance fails, never skips or adds source paths.
Fixtures assert operator CONFIG only: no index/tensor payload, candidate validation,
inference, factory-count proof or genuine Rust-reader + DiskLease qualification.
Definitions authored source-only; subprocesses must await explicit native grant.
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


_ENV = "PLLM_HOST_SOURCE_CONFIG"
_CHECKOUT = Path(__file__).resolve().parents[1]
_CONFIG = b'{"model_type":"qwen2","hidden_size":2,"intermediate_size":2,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"vocab_size":3,"max_position_embeddings":16,"rms_norm_eps":0.000001,"hidden_act":"silu","tie_word_embeddings":false,"attention_bias":false,"rope_theta":10000.0}'

_CHILD = """
import importlib
from importlib.machinery import EXTENSION_SUFFIXES, ExtensionFileLoader
from importlib.util import find_spec
import os
from pathlib import Path
import sys

assert sys.flags.isolated == 1 and sys.flags.optimize == 0
prefix = Path(sys.argv[1]).resolve(strict=True)
checkout = Path(sys.argv[2]).resolve(strict=True)
assert sys.prefix != sys.base_prefix, "matching-wheel venv required"
assert Path(sys.prefix).resolve(strict=True) == prefix, "wrong child venv"
assert (prefix / "pyvenv.cfg").is_file(), "intended venv required"
assert Path(sys.executable).absolute() == Path(sys.argv[3]).absolute()
assert not prefix.is_relative_to(checkout), "use matching-wheel venv outside checkout"
assert not Path.cwd().resolve().is_relative_to(checkout), "isolated cwd required"

def installed_file(origin):
    assert origin, "installed module origin missing"
    path = Path(origin).resolve(strict=True)
    assert path.is_file()
    assert path.is_relative_to(prefix), "module outside intended venv"
    assert not path.is_relative_to(checkout), "checkout import forbidden"
    return path

import pllm
package_file = installed_file(pllm.__file__)
spec = find_spec("pllm._native")
assert spec is not None, "matching native extension missing"
assert isinstance(spec.loader, ExtensionFileLoader), "native extension loader required"
native_file = installed_file(spec.origin)
assert native_file.parent == package_file.parent, "package/extension install mismatch"
assert native_file.name.startswith("_native.")
assert any(native_file.name.endswith(suffix) for suffix in EXTENSION_SUFFIXES)

def import_native():
    native = importlib.import_module("pllm._native")
    assert installed_file(native.__file__) == native_file
    assert isinstance(native.__spec__.loader, ExtensionFileLoader)
    return native

def expect_startup_error(reason):
    try:
        import_native()
    except ValueError as error:
        assert str(error) == reason, (type(error).__name__, str(error))
    else:
        raise AssertionError("configured-invalid native startup unexpectedly succeeded")
    assert "pllm._native" not in sys.modules, "failed import unexpectedly cached"

"""


def _operator_manifest(tmp_path, *, version=1, name="operator.json"):
    """Mirror Rust fixture(), not a candidate-derived inventory or full model."""
    config_digest = hashlib.sha256(_CONFIG).hexdigest()
    manifest = {
        "version": version,
        "approved_lock_digest": "02" * 32,
        "config_sha256": config_digest,
        "semantic_digest": "05" * 32,
        "recipe": "D31",
        "files": [
            {"name": "config.json", "size": len(_CONFIG), "sha256": config_digest},
            {"name": "model.safetensors.index.json", "size": 16, "sha256": "03" * 32},
            {"name": "synthetic.safetensors", "size": 1024, "sha256": "04" * 32},
        ],
        "tensors": [
            {"name": "x.weight", "file": 2, "dtype": "BF16", "shape": [2, 2],
             "begin": 0, "end": 8, "scale": None},
        ],
    }
    encoded = json.dumps(manifest, separators=(",", ":")).encode("ascii")
    path = (tmp_path / name).absolute()
    assert len(encoded) < 4096
    assert path.is_absolute() and len(os.fsencode(path)) < 4096
    path.write_bytes(encoded)  # ONLY the operator manifest; no declared payloads
    return path


def _run_child(tmp_path, body, *, configuration=None, arguments=()):
    # Deliberate allowlist, not os.environ.copy(): no credentials, PYTHONPATH,
    # loader injection or inherited operator setting. Runtime temp files stay local.
    environment = {
        "PATH": os.pathsep.join((str(Path(sys.executable).parent), os.defpath)),
        "VIRTUAL_ENV": sys.prefix,
        "TMPDIR": str(tmp_path),
        "TMP": str(tmp_path),
        "TEMP": str(tmp_path),
        "LC_ALL": "C",
    }
    for key in ("SYSTEMROOT", "WINDIR"):
        if key in os.environ:
            environment[key] = os.environ[key]
    if configuration is not None:
        environment[_ENV] = os.fspath(configuration)
    result = subprocess.run(
        [sys.executable, "-I", "-c", _CHILD + body,
         sys.prefix, str(_CHECKOUT), sys.executable, *map(str, arguments)],
        cwd=tmp_path,
        env=environment,
        close_fds=True,
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 0, (
        f"native startup child exited {result.returncode}\n"
        f"stdout: {result.stdout.decode(errors='replace')}\n"
        f"stderr: {result.stderr.decode(errors='replace')}"
    )


def test_native_startup_absent_environment_imports(tmp_path):
    _run_child(tmp_path, """
assert "PLLM_HOST_SOURCE_CONFIG" not in os.environ
import_native()
""")


@pytest.mark.parametrize(("case", "reason"), [
    ("relative", "host configuration path must be bounded and absolute"),
    ("missing", "host configuration metadata failed"),
    ("malformed-json", "host configuration JSON rejected"),
    ("version-2", "unsupported host configuration version"),
])
def test_native_startup_configured_invalid_import_is_value_error(tmp_path, case, reason):
    path = _operator_manifest(tmp_path, version=2 if case == "version-2" else 1)
    if case == "relative":
        configuration = path.name  # file exists, but operator path must be absolute
    elif case == "missing":
        configuration = path.with_name("missing.json")
        assert not configuration.exists()
    else:
        configuration = path
        if case == "malformed-json":
            path.write_bytes(b"{")
    _run_child(tmp_path, "expect_startup_error(sys.argv[4])\n",
               configuration=configuration, arguments=(reason,))


def test_native_startup_valid_operator_manifest_needs_no_candidate_files(tmp_path):
    path = _operator_manifest(tmp_path)
    assert sorted(entry.name for entry in tmp_path.iterdir()) == [path.name]
    _run_child(tmp_path, """
root = Path(os.environ["PLLM_HOST_SOURCE_CONFIG"]).parent
for name in ("config.json", "model.safetensors.index.json", "synthetic.safetensors"):
    assert not (root / name).exists()
import_native()
for name in ("config.json", "model.safetensors.index.json", "synthetic.safetensors"):
    assert not (root / name).exists()
""", configuration=path)


def test_native_startup_failed_version_sticks_after_repair_and_environment_change(tmp_path):
    bad = _operator_manifest(tmp_path, version=2)
    replacement = _operator_manifest(tmp_path, name="replacement.json")
    _run_child(tmp_path, """
reason = "unsupported host configuration version"
configured = Path(os.environ["PLLM_HOST_SOURCE_CONFIG"])
expect_startup_error(reason)
replacement = Path(sys.argv[4])
configured.write_bytes(replacement.read_bytes())  # repair original valid JSON/schema
os.environ["PLLM_HOST_SOURCE_CONFIG"] = str(replacement)
for _ in range(2):
    expect_startup_error(reason)  # retry import, never reset Rust STARTUP
""", configuration=bad, arguments=(replacement,))


@pytest.mark.parametrize("mutation", ["corrupt", "delete"])
def test_native_startup_success_reimport_reload_stable_after_manifest_mutation(tmp_path, mutation):
    path = _operator_manifest(tmp_path)
    _run_child(tmp_path, """
native = import_native()
configured = Path(os.environ["PLLM_HOST_SOURCE_CONFIG"])
if sys.argv[4] == "corrupt":
    configured.write_bytes(b"{")
else:
    assert sys.argv[4] == "delete"
    configured.unlink()
missing = configured.with_name("missing-after-startup.json")
assert not missing.exists()
os.environ["PLLM_HOST_SOURCE_CONFIG"] = str(missing)
assert import_native() is native
assert importlib.reload(native) is native
assert installed_file(native.__file__) == native_file
# Import stability ONLY: reload need not rerun PyO3 init, and does not prove
# retained expectation identity. The existing Rust initialized_inventory test
# provides the stronger immutable-entry/no-recapture check.
""", configuration=path, arguments=(mutation,))
