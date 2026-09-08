# Follow a request

From preparation to the text your application receives.


## 1. Validate the model plan

The client obtains the public stage names, dimensions, arithmetic parameters, tokenizer information, scales, and quantized token-boundary matrices for public weights. It keeps its secret key local. Pin the model revision and verify the operator's artifact manifest before accepting a plan.

## 2. Protect each activation

The default BFV strategy creates a random input mask and asks one provider to
transform it under encryption. The client decrypts the transformed mask and
stores a single-use pair before the activation is known.

The optional public-weight two-provider strategy instead splits each quantized
activation into two fresh uniform ring shares. It sends the shares concurrently
over separate confidential authenticated channels. Neither non-colluding
provider receives the activation, the other share, or private activation
scales. This strategy has no preparation inventory.

## 3. Process the prompt

Token lookup is local for public weights. Prompt rows then pass through private transformer-body projections and local nonlinear operations. BFV consumes one distinct correlation for every private body row; direct sharing consumes fresh shares instead. The client applies the local output head only to the final prefill row. Any preparation must be included in first token latency.

## 4. Generate and decode

With BFV, the client subtracts the prepared output mask. With two providers, it
adds both wrapping output shares and interprets the bounded result as signed.
Bias is added once after reconstruction. Attention, recurrent updates,
activation scales, and sampling remain local. The gateway emits ordinary text
events only after the client reconstructs them.

## 5. Refill or stop

An empty inventory causes preparation work, not mask reuse. Cancelled requests and rejected proposals still spend every correlation that reached the wire. Closing a study session discards remaining preparation; the reference runtime's durable inventory must not be treated as resistant to restoring old storage snapshots.

## What to measure

Record total request duration, time to first token, gaps between tokens, preparation time, correlations consumed and discarded, and both directions of traffic. A high rate during a short warm interval is not sustained throughput.

The [benchmark guide](/docs/research/benchmarks) separates complete request measurements from matrix kernels and model scale projections.
