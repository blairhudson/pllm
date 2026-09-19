from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_native_plugin_header_package_copy_is_exact() -> None:
    canonical = ROOT / "crates/pllm-plugin-api/include/pllm_plugin.h"
    packaged = ROOT / "python/pllm/include/pllm_plugin.h"
    assert canonical.read_bytes() == packaged.read_bytes()
    text = canonical.read_text(encoding="utf-8")
    assert '#define PLLM_PLUGIN_ABI_VERSION 1u' in text
    assert '#define PLLM_PLUGIN_ENTRY_SYMBOL "pllm_plugin_v1"' in text
    assert "pllm_submit_region_fn" in text
    assert "pllm_poll_fn" in text
    assert "pllm_cancel_fn" in text
