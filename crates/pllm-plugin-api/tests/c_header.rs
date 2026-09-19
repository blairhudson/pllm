#[cfg(not(target_os = "windows"))]
#[test]
fn public_header_compiles_as_c11() {
    use std::fs;
    use std::path::Path;
    use std::process::Command;

    let directory = tempfile::tempdir().expect("temporary directory");
    let source = directory.path().join("abi.c");
    fs::write(
        &source,
        r#"
#include <stddef.h>
#include "pllm_plugin.h"
_Static_assert(sizeof(pllm_status) == sizeof(int32_t), "status layout");
_Static_assert(sizeof(pllm_abi_header) == 8, "header layout");
_Static_assert(sizeof(pllm_handle) == 16, "handle layout");
_Static_assert(offsetof(pllm_owned_buffer, owner) > offsetof(pllm_owned_buffer, capacity), "buffer order");
static pllm_status fixture_create(const pllm_host_vtable *host, pllm_byte_slice config, pllm_handle *instance) {
    (void)host;
    (void)config;
    *instance = (pllm_handle){1, 1};
    return (pllm_status){PLLM_STATUS_OK_VALUE};
}
int main(void) {
    pllm_create_fn function = fixture_create;
    return function == NULL || PLLM_PLUGIN_ABI_VERSION != 1u;
}
"#,
    )
    .expect("write C fixture");
    let compiler = std::env::var_os("CC").unwrap_or_else(|| "cc".into());
    let status = Command::new(compiler)
        .args(["-std=c11", "-Werror", "-c"])
        .arg(&source)
        .arg("-I")
        .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("include"))
        .arg("-o")
        .arg(directory.path().join("abi.o"))
        .status()
        .expect("C compiler must start");
    assert!(status.success());
}
