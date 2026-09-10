# Troubleshooting

Resolve configuration and lifecycle failures without weakening the protocol.


## The provider returns 426 for a Responses request

You sent plaintext to the private provider endpoint. Start `pllm sidecar` in the trusted client environment and use its `/v1` URL in the ordinary SDK. Do not enable a trusted plaintext backend merely to make the request succeed.

## The provider returns 401

Inference, preparation, and the local gateway use separate credentials. The
application uses `PLLM_LOCAL_API_KEY`; the PLLM client uses `PLLM_API_KEY` and
`PLLM_PREPARATION_API_KEY`. Verify them without printing them into logs.

## The client cannot select a model

Set `--model` in `pllm configure`, or pass `model` explicitly. Automatic selection only works when the provider exposes one suitable private model.

## A package named pllm behaves differently

Use the supplied local source and verify `pllm.__file__`. A registry name is not a source identity. Reinstall into the correct UV project or isolated tool environment.

## TenSEAL is unavailable

Check interpreter and operating system wheel support. Use the `he` extra and verify `import tenseal`. Do not substitute `local-test` correlations to process private data.

## Generation stalls

Check whether the prepared inventory reached `READY` and has enough unreserved
rows. Restart and inference idle expiry discard it; refill runs only between
executions. If preparation stalls, compare preparation's
`correction_push_attempts`, `correction_channel_upload_bytes`, and
`correction_push_ns` with inference's correction-channel frames, durable
acknowledgements, bytes, failures, and processing time. A connection count near
frame count indicates that a proxy or idle timeout is defeating persistence.
Confirm `/v1/he/corrections/ws` supports binary WebSocket upgrades and preserves
the provider-push `Authorization` header. Once inventory is `READY`, online stalls
are on the client-to-inference path; preparation should be idle.

## A stream fails halfway through

Stop that execution and discard its reservation. Cancellation, early end of
stream, and failure burn all unused reserved rows. Do not replay a ticket. A later
response may use remaining unreserved rows; if too few remain, the idle client
creates and seals a new inventory before starting it.

## A checkpoint loads but does not match its original outputs

Compare against the same clear quantized graph first. Quantization changes the model; passing integer parity tests does not establish equality with the original floating point logits or language quality.
