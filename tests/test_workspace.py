"""Packaging invariants for the single mixed Python and Rust distribution."""
from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def test_single_python_namespace_and_maturin_binding():
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    maturin = project['tool']['maturin']
    assert project['build-system']['build-backend'] == 'maturin'
    assert maturin['python-source'] == 'python'
    assert maturin['python-packages'] == ['pllm']
    assert maturin['manifest-path'] == 'crates/pllm-python/Cargo.toml'
    assert maturin['module-name'] == 'pllm._native'
    assert [p.name for p in (ROOT / 'python').iterdir() if p.is_dir()] == ['pllm']
    assert not (ROOT / 'he_openai').exists()
    assert not (ROOT / 'pllm').exists()


def test_core_is_independent_of_python():
    workspace = tomllib.loads((ROOT / 'Cargo.toml').read_text())
    assert workspace['workspace']['default-members'] == ['crates/pllm-core']
    core = tomllib.loads((ROOT / 'crates/pllm-core/Cargo.toml').read_text())
    binding = tomllib.loads((ROOT / 'crates/pllm-python/Cargo.toml').read_text())
    assert 'pyo3' not in core['dependencies']
    assert set(binding['dependencies']) == {'pllm-core', 'pyo3'}
    assert core['package']['version']['workspace'] is True
    assert binding['package']['version']['workspace'] is True
    assert binding['lib']['name'] == '_native'
    assert 'cdylib' in binding['lib']['crate-type']


def test_no_retired_namespace_in_active_imports():
    for path in (ROOT / 'python').rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                assert not any(n.name.startswith('he_openai') for n in node.names), path
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or '').startswith('he_openai'), path


def test_lightweight_import_does_not_load_provider_or_he():
    environment = dict(os.environ, PYTHONPATH=str(ROOT / 'python'))
    code = '''
import sys
before = set(sys.modules)
import pllm
loaded = set(sys.modules) - before
assert pllm.__version__
assert 'pllm.runtime.server' not in sys.modules
assert 'tenseal' not in sys.modules
assert 'numpy' not in loaded
assert 'pllm._native' not in sys.modules
assert 'OpenAI' in dir(pllm)
'''
    subprocess.run([sys.executable, '-c', code], check=True, env=environment, cwd=ROOT)


def test_source_python_module_entrypoint():
    environment = dict(os.environ, PYTHONPATH=str(ROOT / 'python'))
    result = subprocess.run([sys.executable, '-m', 'pllm', '--help'], check=True,
                            env=environment, capture_output=True, text=True)
    assert 'serve' in result.stdout and 'chat' in result.stdout


def test_native_source_does_not_expose_mutable_matrix_fields():
    text = (ROOT / 'crates/pllm-core/src/kernels.rs').read_text()
    for field in ('weights', 'rows', 'cols'):
        assert f'pub {field}:' not in text
    assert 'pub fn shape(&self)' in text
    assert 'pub fn weight_bytes(&self)' in text
