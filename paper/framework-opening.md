# 1 Private inference and its opportunity

Private language-model inference divides computation between a trusted client and one or more providers.
Cryptographic protocols protect the client's input, intermediate values, and output under each protocol's security assumptions [@knott2021crypten; @huang2022cheetah; @gupta2026outsourcing].
Providers learn no more than public information and declared leakage.
Leakage means information that the protocol permits a party to receive.
Each method states whether lengths, message sizes, timing, and termination belong to its leakage.
This study treats model weights and dimensions as public information.

This creates two opportunities.
First, sensitive documents, conversations, and application state can use remote computation without disclosure of their content.
Second, individuals and operators of data centers around the world can supply computation without access to that content.
Clients can use capacity beyond their own memory and computation limits without trusting providers with plaintext.

Each method states assumptions about provider behavior and collusion.
Collusion means cooperation between parties that combine information which the protocol keeps separate.
The shared comparison profile assumes a trusted client and setup, confidential authenticated channels, and at most one provider under adversarial control.
The adversary selects that provider before execution.
The provider is semi-honest: it obeys the protocol but examines all information it receives.

Privacy and correctness are separate properties.
Privacy limits disclosure, whereas correctness concerns the specified computation.
Verification adds checks against incorrect results under its own assumptions [@tramer2019slalom].

A token is an item in the model's vocabulary, such as a word or part of a word.
Prefill processes the input prompt, and decode processes subsequent tokens.
Attention uses the key-value (KV) cache, which keeps keys and values from processed tokens.
Logits are scores for vocabulary tokens before the client selects the next token.
These repeated operations make costs for complete requests important.
We ask which methods give the best performance for each client memory budget under the same network conditions.

# 2 PLLM

PLLM combines a Rust-based native runtime with Python interfaces for experiments [@pllm2026].
The native runtime is compiled software for model and protocol operations.
Rust supplies routines for numerical operations, bounded message formats, scheduling, and one-use material.
Python controls experiment configuration, model sources, service lifecycles, and result presentation.
The software development kit (SDK), local gateway for applications, and benchmark driver use the same permitted execution path.

Model adapters translate model operations and saved state into a model-neutral intermediate representation (IR).
The IR describes operations independently of model-specific names.
Components declare capabilities, which specify supported operations, numerical rules, protected representations, computation locations, and material lifetimes.
The compiler combines compatible components and rejects incomplete operation coverage.

PLLM supplies independent reimplementations of selected published mechanisms, each with a stated execution scope.
Examples include Slalom-derived verification and nonlinear operations from SIGMA and FuseFSS [@tramer2019slalom; @gupta2024sigma; @ma2026fusefss].
Each implementation records its source, assumptions, adaptations, and comparisons with reference results.
The source software can supply a reference, but PLLM uses its own execution components.

An experiment fixes the model, component choices, role locations, and request budget.
Immutable plans bind these choices to source identities, numerical rules, and the execution schedule.
Native admission checks compare source identities, capabilities, and resource bounds with the plan before execution.
Native one-use owners control consumable protocol material.
Use, cancellation, failure, or replay retires the affected owner.
This retirement prevents reuse through that owner.

The research harness automatically generates bounded search candidates and does resource checks before measurement.
It schedules ablations, which remove selected changes to measure their effects, and confirmation trials with fresh processes and matched controls.
Coverage reports connect each declared capability to benchmark cases, build identities, and measured or missing results.
Reports preserve configurations, workloads, environments, and measurement windows for replay.
Search comparisons keep workload, numerical, and privacy requirements fixed.
Cryptographic review remains a separate expert task.

```text
Python experiments / research harness
                  |
 Model IR + component capabilities
                  |
     Immutable execution plan
                  |
     SDK / gateway / benchmark
                  |
       Rust-backed execution
       Client <--> Providers
                  |
      Measurements + coverage
```

**Figure 1.** Shared execution and measurement path.
Components determine the provider roles and which operations stay at the trusted client.
