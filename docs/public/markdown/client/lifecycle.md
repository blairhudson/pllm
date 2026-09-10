# Follow a request

From preparation to the text your application receives.


## 1. Validate the model plan

The client obtains stage names, dimensions, arithmetic parameters, tokenizer information, scales, and quantized token-boundary matrices from the inference provider. It separately verifies that the preparation service exposes the same public model body and per-stage weight commitments.

## 2. Prepare inventory before chat

Before chat, the client asks inference to create an inventory and sends trusted
preparation one compact authorization for it. It then sends one root seed and a
batch size for each remote stage. The default is 64 rows per stage; set
`PLLM_PREPARED_INVENTORY_ROWS` to change it. Domain separation expands each root
into one-time tickets, input masks `r`, and output masks `s`, bound to the
inventory and exact stage commitments.

Preparation computes batched `W·r-s` rows and pushes them directly to inference's
fixed endpoint. It waits for inference's durable acknowledgement for every stage.
After all batches arrive, inference seals the inventory and reports `READY`.
Preparation does not participate in online chat.

## 3. Process the prompt

At chat start, the client reserves enough inventory rows for the execution. Token
lookup is local for public weights. Prompt rows then pass through masked
transformer-body projections and local nonlinear operations. For each stage, the
client sends inference only the row's ticket and `x-r`. Inference atomically
consumes the matching preloaded correction and returns `W·x-s`. There is no
client-to-preparation online request or preparation acknowledgement. The client
applies the local output head only to the final prefill row.

## 4. Generate and decode

The client adds `s` and center-decodes the smallest sufficient `u16`, `u24`, or
`u32` ring. Bias is added once after reconstruction.
Attention, recurrent updates,
activation scales, and sampling remain local. The gateway emits ordinary text
events only after the client reconstructs them.

## 5. Burn or stop

Tickets and rows are never reused. An in-memory inventory may supply multiple
chats, but restart or idle expiry discards it. A reservation burns its full range:
cancellation, early end of stream, and failure burn any rows that execution did
not reach. Refill creates a new batch only while the client is idle between
executions, never concurrently with online chat.

## What to measure

Record inventory creation and refill separately from total request duration, time
to first token, gaps between tokens, inference server time, failed attempts,
client-to-inference traffic, preparation authorization and seed bytes, and
preparation-to-inference correction bytes. Report reserved, consumed, and burned
rows. A high rate from an already prepared inventory is not cold-start throughput.

The [benchmark guide](/docs/research/benchmarks) separates complete request measurements from matrix kernels and model scale projections.
