# Strict semantic contract validation

The independent Python suite passes 50 tests. Native replay now matches all five
generated models and all 918 complete semantic observations. This is the semantic
exchange contract; the original 5,125 internal-register exchange remains unclaimed.

* Root `freeze.py --refresh-source-lock` checks original inputs and all 5,125
  expected tensors against the independent generator/oracle before updating the
  source lock. Fixture changes remain visible in Git; no receipt archive is needed.
* Original and new freezes pass `--check`: 5 original graphs / 5,125 trace tensors;
  5 strict semantic graphs / 918 complete physical-node observations.
* Python discovery `python3 -B -m unittest discover -s
  paper/reproduction-contracts/fusefss/decoder -p 'test_*.py'`: 50 tests pass,
  including 25 independent semantic checker tests. Clear-reference positive
  roundtrip is checker self-validation, **not** native evidence.
* Negative controls reject missing/extra/reordered cases, operators, states,
  source encodings; changed active/padding/QKV values; private selection/feedback;
  wrong profile/graph/input hashes, shape/scales/types; incomplete vocabulary;
  changed IEEE/Q12 source values; extra primitive traces; duplicate JSON keys;
  and original-schema or empty batches presented as semantic replay.
* Active final-normalization values, complete last-row vocabulary logits and all
  final token IDs match original oracle exactly for every selection of all cases.

Run the native producer and independent checker with ordinary test commands:

```sh
PLLM_FSS_DECODER_EXPORT_DIR=target/fss-semantic cargo test --release --locked --offline -p pllm-compiler --lib fss_decoder::conformance::conformance_tests::attempt_strict_semantic_execution_replay -- --ignored --exact --test-threads=1
python3 -B paper/reproduction-contracts/fusefss/decoder/check_semantic.py target/fss-semantic/semantic-v1-native-output.json
```

The original full-EXCHANGE checker does not accept the semantic schema. Separate
275-case primitive checks cannot replace complete semantic replay. Generated
SDK/gateway checks and ordinary benchmark outputs are described in
[`docs/research/fusefss-sdk-reference.md`](../../../../docs/research/fusefss-sdk-reference.md).
No trained-checkpoint quality or complete-method performance follows.
