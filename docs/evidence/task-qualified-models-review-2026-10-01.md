# Task-qualified models: review (2026-10-01)

Fixed checkpoints, no training; source-native prompts; compiled native W8A8 semantic clear-kernel execution. One CPU/native/BLAS compute thread; separate model loads.

## Objective results

| Policy | Success | Attempt EOS / truncation | Input / output tokens | Projected all-link arithmetic MB | Projected MB / success |
|---|---:|---:|---:|---:|---:|
| qwen | 4/8 | 8 / 0 | 421 / 22 | 1331.076 | 332.769 |
| smol | 0/8 | 8 / 0 | 457 / 93 | 747.101 | undefined (0 successes) |
| smol_then_qwen | 4/8 | 16 / 0 | 878 / 115 | 2078.177 | 519.544 |

Fallback: 8 additional Qwen attempts; 16 total attempts. All failed Smol attempts charged. No retries. Objective grader is offline oracle, not a deployable general-purpose router.

## Fixed decision gate

Each cohort independently requires >=75% task success, at least Qwen's successes, and <=0.8 times Qwen's projected all-link arithmetic bytes per successful task. Gate and canonical examples fixed before either cohort executes.

- smol: quality-qualified=False; promising=False; cost ratio=None.
- smol_then_qwen: quality-qualified=False; promising=False; cost ratio=1.5612757995167243.

## Success by task family

| Family | Qwen | Smol |
|---|---:|---:|
| arithmetic | 2/2 | 0/2 |
| classification | 0/2 | 0/2 |
| extraction | 1/2 | 0/2 |
| formatting | 1/2 | 0/2 |

## Accounting and limitations

- 16 fixed new public tasks total, 8 screen + 8 disjoint review; same families, no tuning between cohorts. One greedy sample per task/model; too small for broad quality claims.
- Success means exact objective answer AND EOS within fixed 32-token output cap. Inputs bounded at 512; actual counts recorded, never cropped. Longer outputs and open-ended generation untested.
- Clear-kernel task results are measured; cohort traffic is numeric arithmetic projection only. Canonical examples validate covered serialized role bodies only where status=validated; not full-wire or full-cohort network measurement.
- Projection excludes seed requests, envelopes, setup/control, client bundle, HTTP/TLS, checkpoint distribution, local token lookup/head, and upstream source download. Full-wire bytes unknown.
- Source snapshots and compiled cache already available. Cached-source load CPU is not cold provisioning; client bundle cache not isolated. Cold download/network/CPU and accelerator energy/memory unmeasured.
- Fallback uses known public objective grader as offline oracle. Router implementation/latency and failures of a real-world quality detector unmeasured; model+fallback promise applies only where such a grader exists.
- Compute threads capped at one; process may retain housekeeping threads. Co-located CPU timings are descriptive, not WAN latency, independent operators, or cryptographic privacy evidence.
- Output fingerprints are restricted to this explicit public-task diagnostic; ordinary benchmark capture_output_digest defaults to False and keeps response status only. Benchmark history never stores output digests, text, or token IDs.

Prepared arithmetic projection uses actual checkpoint seeded rings: online = consumed rows * (input width + output width) * ring bits / 8; preparation correction = prepared rows * output width * ring bits / 8. Fixed request-sized preparation reserves input tokens + output cap - 1 rows before EOS is known; unused prepared rows charged on every attempt, including fallback. Sum across compiled remote body stages. Token lookup and LM head remain client-local. No historical 39+32 body figure substituted.

EOS token not returned as text; counted separately among selected tokens. Rows per stage = input tokens + selected tokens - 1, including final EOS prediction. Output-cap exhaustion fails even if answer prefix is correct. Only outer whitespace stripped; case, internal spacing, extra prose graded strictly.

## Canonical prespecified examples

Controlled ordinary benchmark API uses temperature=0.0. Validation checks output-text digest parity, EOS/selected-token counts, consumed/prepared/burned inventory rows, body fingerprint, full stage set, protocol-body reconciliation, and arithmetic payload bounds per stage/edge. Setup/client-bundle bodies shown separately; client cache is not isolated.

```json
[
  {
    "arithmetic_projection_by_edge": {
      "client->inference": 33024000,
      "inference->client": 46694400,
      "preparation->inference": 73777152
    },
    "body_fingerprint": "5d631be30158b3ea2a72cae355ce09a3b304bbfddde48b5758c339c346a34974",
    "checks": {
      "completed": true,
      "explicit_greedy_sampling": true,
      "no_plaintext_prompt": true,
      "no_plaintext_token_ids": true,
      "reconciled_stage_bodies": true,
      "same_body": true,
      "same_burned_inventory_rows": true,
      "same_consumed_inventory_rows": true,
      "same_eos_termination": true,
      "same_input_tokens": true,
      "same_output_text_digest": true,
      "same_output_tokens": true,
      "same_prepared_inventory_rows": true,
      "same_selected_tokens": true
    },
    "clear_output_content_parity_measured": true,
    "consumed_rows_per_stage": 50,
    "covered_all_link_serialized_role_body_bytes": 299728812,
    "covered_online_serialized_role_body_bytes": 79921280,
    "eos": true,
    "full_wire_bytes": null,
    "input_tokens": 48,
    "model": "qwen",
    "observed_burned_rows_per_stage": 29,
    "observed_consumed_rows_per_stage": 50,
    "observed_prepared_rows_per_stage": 79,
    "other_setup_control_bundle_body_bytes": 145979820,
    "output_text_digest": "4bf17f4cd842cc1812f81a696010ee962c1c6f4db784a26b47cfde3144db8ae4",
    "output_tokens": 2,
    "prepared_rows_per_stage": 79,
    "projected_all_link_arithmetic_bytes": 153495552,
    "projection_payloads_bounded_by_matching_stage_bodies": true,
    "sampling": {
      "effective_temperature": 0.0,
      "mode": "greedy",
      "requested_temperature": 0.0,
      "top_p": null
    },
    "selected_tokens_including_eos": 3,
    "serialized_stage_body_bytes": 153748992,
    "serialized_stage_body_bytes_by_role_and_edge": {
      "attention_output": {
        "client->inference": 3267700,
        "client->preparation": 2472,
        "inference->client": 3234148,
        "preparation->client": 2150,
        "preparation->inference": 5104454
      },
      "mlp_down": {
        "client->inference": 23389348,
        "client->preparation": 2472,
        "inference->client": 4309396,
        "preparation->client": 2078,
        "preparation->inference": 6803294
      },
      "mlp_gate_up": {
        "client->inference": 3267700,
        "client->preparation": 2472,
        "inference->client": 35029348,
        "preparation->client": 2150,
        "preparation->inference": 55340870
      },
      "qkv_projection": {
        "client->inference": 3267796,
        "client->preparation": 2472,
        "inference->client": 4155844,
        "preparation->client": 2198,
        "preparation->inference": 6560630
      }
    },
    "stage_body_bytes_beyond_arithmetic_projection": 253440,
    "stage_body_bytes_reconciled": true,
    "status": "validated",
    "task_id": "review-arithmetic-1"
  },
  {
    "arithmetic_projection_by_edge": {
      "client->inference": 17919360,
      "inference->client": 28460160,
      "preparation->inference": 38257920
    },
    "body_fingerprint": "9adb95b8cda428d8c6ff6367020fc5f365651bf50dcb7e712219886ecadc5469",
    "checks": {
      "completed": true,
      "explicit_greedy_sampling": true,
      "no_plaintext_prompt": true,
      "no_plaintext_token_ids": true,
      "reconciled_stage_bodies": true,
      "same_body": true,
      "same_burned_inventory_rows": true,
      "same_consumed_inventory_rows": true,
      "same_eos_termination": true,
      "same_input_tokens": true,
      "same_output_text_digest": true,
      "same_output_tokens": true,
      "same_prepared_inventory_rows": true,
      "same_selected_tokens": true
    },
    "clear_output_content_parity_measured": true,
    "consumed_rows_per_stage": 61,
    "covered_all_link_serialized_role_body_bytes": 117253369,
    "covered_online_serialized_role_body_bytes": 47252200,
    "eos": true,
    "full_wire_bytes": null,
    "input_tokens": 51,
    "model": "smol",
    "observed_burned_rows_per_stage": 21,
    "observed_consumed_rows_per_stage": 61,
    "observed_prepared_rows_per_stage": 82,
    "other_setup_control_bundle_body_bytes": 31678949,
    "output_text_digest": "5422f8175d721ee1b0c65e68be16915376975ea09b39f481fa0743056dd17dbe",
    "output_tokens": 10,
    "prepared_rows_per_stage": 82,
    "projected_all_link_arithmetic_bytes": 84637440,
    "projection_payloads_bounded_by_matching_stage_bodies": true,
    "sampling": {
      "effective_temperature": 0.0,
      "mode": "greedy",
      "requested_temperature": 0.0,
      "top_p": null
    },
    "selected_tokens_including_eos": 11,
    "serialized_stage_body_bytes": 85574420,
    "serialized_stage_body_bytes_by_role_and_edge": {
      "attention_output": {
        "client->inference": 3332510,
        "client->preparation": 3090,
        "inference->client": 3210290,
        "preparation->client": 2690,
        "preparation->inference": 4261190
      },
      "mlp_down": {
        "client->inference": 8602010,
        "client->preparation": 3090,
        "inference->client": 3209390,
        "preparation->client": 2600,
        "preparation->inference": 4261100
      },
      "mlp_gate_up": {
        "client->inference": 3332510,
        "client->preparation": 3090,
        "inference->client": 16913330,
        "preparation->client": 2690,
        "preparation->inference": 22681670
      },
      "qkv_projection": {
        "client->inference": 3333110,
        "client->preparation": 3090,
        "inference->client": 5319050,
        "preparation->client": 2750,
        "preparation->inference": 7095170
      }
    },
    "stage_body_bytes_beyond_arithmetic_projection": 936980,
    "stage_body_bytes_reconciled": true,
    "status": "validated",
    "task_id": "review-arithmetic-1"
  }
]
```

## Sampling correction and historical provenance

Prior canonical dashboard requests omitted temperature; SDK effective default was 0.8, whereas clear execution used greedy argmax. Optional temperature now flows through ordinary benchmark API/config and dashboard request into SDK only when explicit. None preserves historical default. Effective sampling included in report and comparison cohort keys; unreported sampling cannot rank. Internal Python API and dashboard JSON request support this control; no new CLI flag.

```json
{
  "client_default_source": "python/pllm/runtime/client.py::_sampling_temperature",
  "controlled_canonical_temperature": 0.0,
  "historical_canonical_examples": [
    {
      "arithmetic_projection_by_edge": {
        "client->inference": 33024000,
        "inference->client": 46694400,
        "preparation->inference": 73777152
      },
      "body_fingerprint": "5d631be30158b3ea2a72cae355ce09a3b304bbfddde48b5758c339c346a34974",
      "checks": {
        "completed": true,
        "no_plaintext_prompt": true,
        "no_plaintext_token_ids": true,
        "reconciled_stage_bodies": true,
        "same_body": true,
        "same_input_tokens": true,
        "same_output_tokens": true
      },
      "clear_output_content_parity_measured": false,
      "consumed_rows_per_stage": 50,
      "covered_all_link_serialized_role_body_bytes": 299728812,
      "covered_online_serialized_role_body_bytes": 79921280,
      "full_wire_bytes": null,
      "input_tokens": 48,
      "model": "qwen",
      "other_setup_control_bundle_body_bytes": 145979820,
      "output_tokens": 2,
      "prepared_rows_per_stage": 79,
      "projected_all_link_arithmetic_bytes": 153495552,
      "projection_payloads_bounded_by_matching_stage_bodies": true,
      "serialized_stage_body_bytes": 153748992,
      "serialized_stage_body_bytes_by_role_and_edge": {
        "attention_output": {
          "client->inference": 3267700,
          "client->preparation": 2472,
          "inference->client": 3234148,
          "preparation->client": 2150,
          "preparation->inference": 5104454
        },
        "mlp_down": {
          "client->inference": 23389348,
          "client->preparation": 2472,
          "inference->client": 4309396,
          "preparation->client": 2078,
          "preparation->inference": 6803294
        },
        "mlp_gate_up": {
          "client->inference": 3267700,
          "client->preparation": 2472,
          "inference->client": 35029348,
          "preparation->client": 2150,
          "preparation->inference": 55340870
        },
        "qkv_projection": {
          "client->inference": 3267796,
          "client->preparation": 2472,
          "inference->client": 4155844,
          "preparation->client": 2198,
          "preparation->inference": 6560630
        }
      },
      "stage_body_bytes_beyond_arithmetic_projection": 253440,
      "stage_body_bytes_reconciled": true,
      "status": "validated",
      "task_id": "review-arithmetic-1"
    },
    {
      "checks": {
        "completed": true,
        "no_plaintext_prompt": true,
        "no_plaintext_token_ids": true,
        "reconciled_stage_bodies": true,
        "same_body": true,
        "same_input_tokens": true,
        "same_output_tokens": false
      },
      "clear_input_tokens": 51,
      "clear_output_tokens": 10,
      "full_wire_bytes": null,
      "model": "smol",
      "observed_input_tokens": 51,
      "observed_output_tokens": 32,
      "status": "cohort_mismatch",
      "task_id": "review-arithmetic-1"
    }
  ],
  "historical_effective_temperature": 0.8,
  "historical_examples_are_greedy_parity_evidence": false,
  "historical_sampling_source": "inferred from omitted request and unchanged SDK default; historical records lacked sampling field",
  "root_cause": "historical dashboard omitted temperature; client None default was 0.8, clear execution was greedy",
  "task_contract_unchanged": true
}
```

## Public output-digest capture policy

Public task canonical calls explicitly enable capture_output_digest=True. Ordinary benchmark API/config defaults to False: generation.response_status remains available, output fingerprints are omitted. Opt-in flag requires exact bool; sampling defaults and cohort matching remain independent. Benchmark archive remains text/token-ID/digest-free. No new CLI flag.

```json
{
  "canonical_capture_provenance": "retained prior public diagnostic observations from unconditional capture; opt-in-only metadata change, not rerun",
  "canonical_examples_rerun_for_opt_in_policy": false,
  "capture_output_digest": true,
  "ordinary_benchmark_default": false,
  "scope": "public fixed-task diagnostic only; no raw text or token IDs persisted"
}
```

## Reproduce

```sh
.venv/bin/python scripts/probe_task_qualified_models.py --canonical
.venv/bin/python -m pytest tests/test_benchmark_sampling.py tests/test_benchmark_cli.py tests/test_dashboard.py tests/test_dashboard_history.py tests/test_task_qualified_models.py -m 'not integration'
.venv/bin/python -m ruff check python/pllm/runtime/dashboard.py python/pllm/runtime/benchmark_cli.py scripts/probe_task_qualified_models.py tests/test_benchmark_sampling.py tests/test_task_qualified_models.py
.venv/bin/python -m ruff format --check python/pllm/runtime/dashboard.py python/pllm/runtime/benchmark_cli.py scripts/probe_task_qualified_models.py tests/test_benchmark_sampling.py tests/test_task_qualified_models.py
```

Task contract: `examples/benchmarks/task_qualified_tasks.json`. JSON evidence stores scalar counts, source locks, and digests; no generated text or token IDs.
