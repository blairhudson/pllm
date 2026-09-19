#ifndef PLLM_PLUGIN_H
#define PLLM_PLUGIN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PLLM_PLUGIN_ABI_VERSION 1u
#define PLLM_PLUGIN_ENTRY_SYMBOL "pllm_plugin_v1"

#if defined(_WIN32)
#define PLLM_PLUGIN_EXPORT __declspec(dllexport)
#else
#define PLLM_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

typedef int32_t pllm_status;

#define PLLM_STATUS_OK_VALUE 0
#define PLLM_STATUS_INVALID_ARGUMENT_VALUE 1
#define PLLM_STATUS_UNSUPPORTED_VALUE 2
#define PLLM_STATUS_NOT_FOUND_VALUE 3
#define PLLM_STATUS_RESOURCE_EXHAUSTED_VALUE 4
#define PLLM_STATUS_CANCELLED_VALUE 5
#define PLLM_STATUS_INTERNAL_VALUE 6
#define PLLM_STATUS_PANIC_VALUE 7

typedef struct pllm_abi_header {
    uint32_t size;
    uint32_t abi_version;
} pllm_abi_header;

typedef struct pllm_handle {
    uint64_t slot;
    uint64_t generation;
} pllm_handle;

typedef struct pllm_byte_slice {
    const uint8_t *data;
    uint64_t len;
} pllm_byte_slice;

typedef struct pllm_owned_buffer {
    uint8_t *data;
    uint64_t len;
    uint64_t capacity;
    pllm_handle owner;
} pllm_owned_buffer;

typedef pllm_status (*pllm_emit_event_fn)(pllm_handle host, pllm_byte_slice event);
typedef uint8_t (*pllm_is_cancelled_fn)(pllm_handle host);

typedef struct pllm_host_vtable {
    pllm_abi_header header;
    pllm_handle host;
    pllm_emit_event_fn emit_event;
    pllm_is_cancelled_fn is_cancelled;
    void *reserved[4];
} pllm_host_vtable;

typedef pllm_status (*pllm_create_fn)(
    const pllm_host_vtable *host,
    pllm_byte_slice configuration,
    pllm_handle *instance
);
typedef pllm_status (*pllm_destroy_fn)(pllm_handle instance);
typedef pllm_status (*pllm_describe_fn)(
    pllm_handle instance,
    pllm_owned_buffer *output
);
typedef pllm_status (*pllm_execute_region_fn)(
    pllm_handle instance,
    pllm_byte_slice request,
    pllm_owned_buffer *output
);
typedef pllm_status (*pllm_submit_region_fn)(
    pllm_handle instance,
    pllm_byte_slice request,
    pllm_handle *operation
);
typedef pllm_status (*pllm_poll_fn)(
    pllm_handle instance,
    pllm_handle operation,
    uint8_t *ready,
    pllm_owned_buffer *output
);
typedef pllm_status (*pllm_cancel_fn)(
    pllm_handle instance,
    pllm_handle operation
);
typedef pllm_status (*pllm_release_buffer_fn)(
    pllm_handle instance,
    pllm_owned_buffer buffer
);
typedef pllm_status (*pllm_last_error_fn)(
    pllm_handle instance,
    pllm_owned_buffer *output
);

typedef struct pllm_plugin_vtable {
    pllm_abi_header header;
    pllm_create_fn create;
    pllm_destroy_fn destroy;
    pllm_describe_fn describe;
    pllm_execute_region_fn execute_region;
    pllm_submit_region_fn submit_region;
    pllm_poll_fn poll;
    pllm_cancel_fn cancel;
    pllm_release_buffer_fn release_buffer;
    pllm_last_error_fn last_error;
    void *reserved[8];
} pllm_plugin_vtable;

typedef pllm_status (*pllm_plugin_entry_fn)(
    const pllm_host_vtable *host,
    const pllm_plugin_vtable **plugin
);

PLLM_PLUGIN_EXPORT pllm_status pllm_plugin_v1(
    const pllm_host_vtable *host,
    const pllm_plugin_vtable **plugin
);

#ifdef __cplusplus
}
#endif

#endif
