def test_replicated_public_linear_algebra():
    from pathlib import Path
    import importlib.util

    path = Path(__file__).parents[1] / "scripts/probe_three_party_cost.py"
    spec = importlib.util.spec_from_file_location("three_party_probe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    import sys

    original = list(sys.path)
    try:
        sys.path.insert(0, str(path.parent))
        spec.loader.exec_module(module)
        module.replicated_linear_oracle()
    finally:
        sys.path[:] = original
