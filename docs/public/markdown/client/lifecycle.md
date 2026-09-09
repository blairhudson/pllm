# Follow a request

From preparation to the text your application receives.


## 1. Validate the model plan

The client obtains stage names, dimensions, arithmetic parameters, tokenizer information, scales, and quantized token-boundary matrices from the inference provider. It separately verifies that the preparation service exposes the same public model body and per-stage weight commitments. After inference registers a random prepared session, the client sends preparation one compact authorization binding that session, the model commitments, quantization, and bounded attempt budget. Preparation validates and relays it through its provider-only push channel. Failure closes the inference session; authorization cannot be replayed.

## 2. Protect each activation

For every linear stage, the client creates a fresh 128-bit attempt ID and
256-bit seed and domain-separates input mask `r` from output mask `s`. Expansion
binds session, model, body, stage, weight, dimensions, quantization, and exact
ring profile. It sends only the seed and bound stage metadata
to the trusted preparation service. At the same time, it sends `x-r` to the
untrusted inference provider. Preparation pushes `W·r-s` directly to the fixed
inference endpoint and returns a small acknowledgement. There is no callback URL
or durable preparation inventory. Inference rejects stage activation before the
one-time session authorization. Once authorized, it starts stage computation on
activation while concurrently awaiting the correction.

## 3. Process the prompt

Token lookup is local for public weights. Prompt rows then pass through masked transformer-body projections and local nonlinear operations. Every stage call consumes its seed before either request is sent; timeout, cancellation, or an ambiguous result burns that attempt without retry. The client applies the local output head only to the final prefill row.

## 4. Generate and decode

Inference atomically consumes both halves and returns `W·x-s`. The client adds
`s` and center-decodes the smallest sufficient `u16`, `u24`, or `u32` ring. Bias
is added once after reconstruction.
Attention, recurrent updates,
activation scales, and sampling remain local. The gateway emits ordinary text
events only after the client reconstructs them.

## 5. Burn or stop

Seeds are never reused. A failed attempt starts over with a newly sampled seed;
the client does not independently retry either half. Completion and cancellation
leave no reusable ticket batch or durable correlation state.

## What to measure

Record total request duration, time to first token, gaps between tokens, preparation and inference server time, failed attempts, client-to-service traffic, one-time authorization bytes, and preparation-to-inference correction bytes separately. A high rate during a short warm interval is not sustained throughput.

The [benchmark guide](/docs/research/benchmarks) separates complete request measurements from matrix kernels and model scale projections.
