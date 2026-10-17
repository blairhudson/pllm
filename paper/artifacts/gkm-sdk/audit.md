# SDK integration audit and closure

Two separate source/test reviews inspected the native transport and ordinary SDK
integration. Bounded installed-runtime probes accompanied the reviews. These are
implementation reviews, not independent cryptographic certification.

## Native boundary

The transport review found no blocking core/compiler defect in role-local owners,
source/publication/context/schedule pins, original arithmetic, exact claim counts,
native final verification, retained-frame leases or sticky native failure handling.

It found that PyO3 rejected malformed signatures before a method could consume
its owner. State-changing bindings now take the owner before bounded positional
and keyword parsing. The 48 binding checks cover missing, extra and duplicate
arguments, conversion errors, keyword-subclass aliases and complete accepted
call forms. Errors raised by Python before native dispatch cannot consume a
native owner; this limit is explicit.

## SDK and lifecycle findings

| Finding | Repair and regression boundary |
| --- | --- |
| Tokenizer padding could exceed the admitted input allocation | Bound the original tokenizer metadata and padding before constructing or encoding input. |
| Gateway parsed an unbounded request body before adapter admission | Bound raw ingress before JSON parsing, including whitespace and streamed bodies. |
| Closing an unstarted async generator retained its exclusive request seat | The owning wrapper explicitly closes the source and retires the attempt. |
| Per-read HTTP timeouts allowed trickle responses beyond the request deadline | Absolute deadlines and cancellation checks cover complete exchanges and retirement. |
| Busy or stale worker requests could retire another request | Admission and routing reject unrelated IDs before current-owner failure retirement. |
| Gateway disconnect waited for execution before signalling cancellation | Signal first, then drain outstanding thread work. |
| Failed online work supplied fabricated zero timing/call totals | Unavailable observations remain unknown; received bodies remain charged. |

The follow-up found three further defects, all closed in the final installed build:

1. Concurrent `aclose()` could race a running async generator. Advancement and
   close now serialize, cancellation signals before waiting, and a shared close
   task survives cancellation of its caller.
2. DNS executor shutdown could exceed the retirement deadline. Admitted origins
   use IP literals; HTTP `localhost` normalizes to loopback. The request-owned
   I/O loop prohibits executor work and drains its explicit non-daemon thread.
3. A synchronous call inside an active asyncio loop could not run the former
   nested Runner. The I/O owner now uses a separate selector thread.

The sidecar also preserves one copied context across iterator advancement and
closure, so request-owned ContextVar tokens retire in the correct context.

The final read-only recheck passed 56 targeted installed-wheel regressions and
matched nine relevant Python files plus the native extension against the wheel.
The parent suite separately passed all 547 installed checks and 117 native
checks. No audit reproducers are treated as new checkpoint or performance data.
