import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { adaptedCohorts, benchmarkSuiteCohorts, epochPrefixCohorts, median, applyTrafficAttribution, applicationBodyAttribution, checkedCommunicationWindow } from '../scripts/research-scorecard-adapters.mjs';
import { buildScorecard, metrics, executionPlacement, providerPeak } from '../scripts/research-scorecard.mjs';
import { sortMetricRows, sotaTable } from '../lib/research-rankings.mjs';

const root = fileURLToPath(new URL('../../', import.meta.url));
const read = (file) => fs.readFileSync(root + file);
const manifest = JSON.parse(read('docs/data/research/benchmark-cohorts.json'));
const specs = manifest.adapted_cohorts;
const helpers = { metrics, executionPlacement, providerPeak };
const load = (index) => ({ spec: specs[index], document: JSON.parse(read(specs[index].report)),
  plan: specs[index].plan ? JSON.parse(read(specs[index].plan)) : null });

test('original SDK suites yield independent workload and cold/repeat medians', () => {
  assert.equal(median([4, 1, 2]), 2);
  assert.equal(median([4, 1]), 2.5);
  assert.throws(() => median([1, NaN]));
  for (const index of [0, 1, 2]) {
    const { spec, document, plan } = load(index);
    const views = benchmarkSuiteCohorts(spec, document, plan, helpers);
    assert.equal(views.length, index === 1 ? 4 : 2);
    for (const view of views) for (const row of view.rows) {
      const samples = document.results.filter((result) => result.execution.case.name === row.id);
      const request = view.identity.requestIndex;
      const reports = samples.map((result) => result.execution.details.report);
      assert.equal(row.sampleCount, 3);
      assert.deepEqual(row.pipeline, samples[0].execution.case.target.parameters.model.experiment.pipeline);
      assert.equal(row.metrics.full, median(reports.map((report) => report.runs[request].durations.full_seconds)));
      assert.equal(row.metrics.ttft, median(reports.map((report) => report.runs[request].durations.ttft_seconds)));
      assert.equal(row.metrics.decodeTps, median(reports.map((report) => 7 / report.runs[request].durations.generation_seconds)));
      assert.equal(row.metrics.clientPeak, median(reports.map((report) => report.client_process_memory.after.lifetime_peak_rss_bytes / 1e6)));
      assert.equal(row.clientMemory.maxPeakBytes, Math.max(...reports.map((report) => report.client_process_memory.after.lifetime_peak_rss_bytes)));
      assert.equal(row.clientMemory.sampleCount, reports.length);
      assert.equal(row.metrics.totalNetwork, median(reports.map((report) => {
        const sample = report.native_network_accounting.samples.runs[request];
        return (sample.after.wan_emulation.total_stream_bytes - (request === 0 ? 0 : sample.before.wan_emulation.total_stream_bytes)) / 1e6;
      })));
      assert.equal(row.metrics.preparationCpu, request === 0
        ? median(reports.map((report) => report.process_cpu_accounting.cold_first_response_cpu_seconds_by_role.preparation)) : null);
      assert.equal(row.metrics.baseNetwork, null);
      assert.equal(row.metrics.tokenNetwork, null);
      assert.equal(row.metrics.artifactBytes, null, 'Local artifact read counters are not unique publication size');
      assert.match(view.measurement.clientRssLabel, /scenario-lifetime \(cold \+ repeat\)/);
      assert.deepEqual(row.recordedCase, samples[0].execution.case);
      assert.deepEqual(row.samplingContract, document.suite.sampling);
      if (index === 1) {
        assert.equal(row.researchSelector.pipeline_component, false);
        assert.equal(row.pipeline.components.client_mlp, undefined);
        assert.deepEqual(row.researchSelector.selection, samples[0].execution.case.target.parameters.client_mlp);
      }
    }
  }
});

test('SDK adapter rejects mutated matched gates, incomplete cells and changed metric windows', () => {
  const { spec, document, plan } = load(0);
  const mutations = [
    (d) => { d.complete = false; },
    (d) => { d.results.pop(); },
    (d) => { d.results[1].execution.replicate = d.results[0].execution.replicate; d.results[1].execution.case = d.results[0].execution.case; },
    (d) => { d.results[0].execution.cohort = '0'.repeat(64); },
    (d) => { d.results[0].execution.build.digest = '0'.repeat(64); },
    (d) => { d.results[0].execution.isolation.fresh_process = false; },
    (d) => { d.results[0].execution.fixture.metadata.owned_client_cache = 'shared'; },
    (d) => { d.results[0].execution.comparison.privacy_digest = null; },
    (d) => { d.results[0].execution.comparison.measurement_digest = '0'.repeat(64); },
    (d) => { d.results[0].execution.case.target.parameters.model.experiment.pipeline.components.quantization.params.activation_bits = 4; },
    (d) => { d.results[0].execution.details.report.configuration.source_lock_digest = '0'.repeat(64); },
    (d) => { d.results[0].execution.details.report.configuration.prompt_digest = '0'.repeat(64); },
    (d) => { d.results[0].execution.details.report.runs[0].generation.output_text_digest = '0'.repeat(64); },
    (d) => { d.results[0].execution.details.report.runs[0].tokens.authoritative = false; },
    (d) => { d.results[0].execution.details.report.runs[0].privacy.plaintext_prompt_bytes_sent = 1; },
    (d) => { d.results[0].execution.details.report.runs[0].privacy.prepared_stage_rows_burned++; },
    (d) => { d.results[0].execution.details.report.runs[1].warm = false; },
    (d) => { d.results[0].execution.details.report.warmup_runs.push({}); },
    (d) => { d.results[0].execution.details.report.topology_accounting.runs[0].all_link_serialized_body_bytes++; },
    (d) => {
      const report = d.results[0].execution.details.report, comm = report.communication_per_token.runs[0];
      comm.online_body_bytes++;
      comm.online_mb_per_output_token = comm.online_body_bytes / 1e6 / comm.generated_output_tokens;
    },
    (d) => {
      const report = d.results[0].execution.details.report, comm = report.communication_per_token.runs[0];
      comm.decode_online_body_bytes++;
      comm.decode_online_mb_per_output_token = comm.decode_online_body_bytes / 1e6 / comm.decode_output_tokens;
    },
    (d) => { d.results[0].execution.details.report.native_network_accounting.samples.runs[0].after.wan_emulation.total_stream_bytes++; },
    (d) => { d.results[0].metrics[0].window = 'different CPU window'; },
  ];
  for (const mutate of mutations) {
    const copy = structuredClone(document); mutate(copy);
    assert.throws(() => benchmarkSuiteCohorts(spec, copy, plan, helpers));
  }
  const changedPlan = structuredClone(plan);
  changedPlan.study.variants.find((variant) => variant.name === 'digit8').parameters.pipeline__preparation__min_rows++;
  assert.throws(() => benchmarkSuiteCohorts(spec, document, changedPlan, helpers), /suite\/plan/);
  for (const target of [spec.report, spec.plan, spec.replay_file]) {
    assert.throws(() => adaptedCohorts(spec, (file) => file === target ? Buffer.from('{}') : read(file), helpers), /changed/);
  }
});

function communicationWindowFixture() {
  return {
    runs: [{ status: 'completed', tokens: { authoritative: true, output_tokens: 3 },
      network: { decode_scope: 'after-first-output-through-completion', decode_output_tokens: 2, decode_online_body_bytes: 40 } }],
    topology_accounting: { runs: [{ schema: 'pllm.topology_body_accounting.v1', tracked_body_counter_set_present: true,
      all_link_serialized_body_bytes: 100, online_all_link_serialized_body_bytes: 60,
      body_bytes_by_edge: [{ phase: 'offline', serialized_body_bytes: 40 }, { phase: 'online', serialized_body_bytes: 60 }] }] },
    communication_per_token: { schema: 'pllm.communication_per_generated_token.v1',
      byte_unit: 'decimal MB (1000000 bytes)',
      denominator: 'authoritative generated output tokens, not input tokens or output cap',
      scope: 'covered application bodies; no full wire or checkpoint download',
      decode_scope: 'online bodies after first output through completion, divided by N-1',
      runs: [{ generated_output_tokens: 3, covered_body_bytes: 100, online_body_bytes: 60,
        decode_output_tokens: 2, decode_online_body_bytes: 40, covered_mb_per_output_token: 100 / 1e6 / 3,
        online_mb_per_output_token: 60 / 1e6 / 3, decode_online_mb_per_output_token: 40 / 1e6 / 2 }] },
  };
}

test('cached body numerators and rates bind to authoritative request samples and windows', () => {
  const original = communicationWindowFixture();
  assert.equal(checkedCommunicationWindow(original, 0), original.communication_per_token.runs[0]);
  const mutations = [
    (r) => { const c = r.communication_per_token.runs[0]; c.online_body_bytes = 50; c.online_mb_per_output_token = 50 / 1e6 / 3; },
    (r) => { const c = r.communication_per_token.runs[0]; c.decode_online_body_bytes = 20; c.decode_online_mb_per_output_token = 20 / 1e6 / 2; },
    (r) => { const c = r.communication_per_token.runs[0]; c.generated_output_tokens = 4; c.covered_mb_per_output_token = 100 / 1e6 / 4; c.online_mb_per_output_token = 60 / 1e6 / 4; },
    (r) => { const c = r.communication_per_token.runs[0]; c.decode_output_tokens = 3; c.decode_online_mb_per_output_token = 40 / 1e6 / 3; },
    (r) => { r.communication_per_token.runs.push(structuredClone(r.communication_per_token.runs[0])); },
    (r) => { r.topology_accounting.runs.push(structuredClone(r.topology_accounting.runs[0])); },
    (r) => { r.communication_per_token.denominator = 'requested output allowance'; },
    (r) => { r.communication_per_token.decode_scope = 'whole response'; },
    (r) => { r.communication_per_token.byte_unit = 'MiB'; },
    (r) => { r.communication_per_token.scope = 'tcp_stream_payload'; },
    (r) => { r.runs[0].network.decode_scope = 'whole response'; },
    (r) => { r.runs[0].network.decode_output_tokens = 3; },
    (r) => { r.runs[0].network.decode_online_body_bytes = 61; },
    (r) => { r.communication_per_token.runs[0].body_counter_coverage = 'other coverage'; },
    (r) => { r.topology_accounting.runs[0].online_all_link_serialized_body_bytes = 50; const c = r.communication_per_token.runs[0]; c.online_body_bytes = 50; c.online_mb_per_output_token = 50 / 1e6 / 3; },
  ];
  for (const mutate of mutations) {
    const copy = structuredClone(original); mutate(copy);
    assert.throws(() => checkedCommunicationWindow(copy, 0));
  }
  for (const index of [-1, 1, 0.5]) assert.throws(() => checkedCommunicationWindow(original, index));
});

test('unknown decode boundaries and one-output zero denominators remain unknown, not zero rates', () => {
  for (const change of [
    (r) => { delete r.runs[0].network; },
    (r) => { r.runs[0].network.decode_scope = 'unknown boundary'; },
    (r) => { r.topology_accounting.runs[0].online_all_link_serialized_body_bytes = null;
      const c = r.communication_per_token.runs[0]; c.online_body_bytes = c.online_mb_per_output_token = null; },
  ]) {
    const report = communicationWindowFixture(); change(report);
    const cached = report.communication_per_token.runs[0];
    cached.decode_output_tokens = cached.decode_online_body_bytes = cached.decode_online_mb_per_output_token = null;
    assert.equal(checkedCommunicationWindow(report, 0).decode_online_mb_per_output_token, null);
    cached.decode_online_mb_per_output_token = 0;
    assert.throws(() => checkedCommunicationWindow(report, 0));
  }
  const report = communicationWindowFixture(), c = report.communication_per_token.runs[0];
  report.runs[0].tokens.output_tokens = c.generated_output_tokens = 1;
  report.runs[0].network.decode_output_tokens = c.decode_output_tokens = 0;
  report.runs[0].network.decode_online_body_bytes = c.decode_online_body_bytes = 0;
  c.covered_mb_per_output_token = 100 / 1e6; c.online_mb_per_output_token = 60 / 1e6;
  c.decode_online_mb_per_output_token = null;
  assert.equal(checkedCommunicationWindow(report, 0).decode_online_body_bytes, 0);
  c.decode_online_mb_per_output_token = 0;
  assert.throws(() => checkedCommunicationWindow(report, 0));
});

test('new ACK body coverage is explicit, reconciled and never supplies partial traffic heads', () => {
  const { spec, document, plan } = load(0);
  const next = structuredClone(document);
  for (const result of next.results) for (const [index, ledger] of result.execution.details.report.topology_accounting.runs.entries()) {
    ledger.schema = 'pllm.topology_body_accounting.v3';
    ledger.body_counter_coverage = 'prepared_protocol_lifecycle_controls_and_correction_acks_v3';
    ledger.control_body_counter_set_present = true;
    ledger.correction_ack_counter_set_present = true;
    result.execution.details.report.communication_per_token.runs[index].body_counter_coverage = ledger.body_counter_coverage;
    ledger.body_bytes_by_edge.push({ source: 'inference', destination: 'preparation', phase: 'offline', serialized_body_bytes: 0 });
    const audit = result.execution.details.report.runs[index].privacy;
    audit.correction_ack_bytes = 0;
    audit.correction_ack_accounted_push_bytes = audit.correction_push_bytes;
  }
  const views = benchmarkSuiteCohorts(spec, next, plan, helpers);
  assert.ok(views.every((view) => view.identity.bodyCoverage.endsWith('_v3')));
  assert.ok(views.flatMap((view) => view.rows).every((row) => row.metrics.baseNetwork === null && row.metrics.tokenNetwork === null));
  for (const mutate of [
    (d) => { d.results[0].execution.details.report.topology_accounting.runs[0].correction_ack_counter_set_present = false; },
    (d) => { d.results[0].execution.details.report.runs[0].privacy.correction_ack_accounted_push_bytes++; },
    (d) => { d.results[0].execution.details.report.runs[0].privacy.correction_ack_bytes++; },
  ]) {
    const copy = structuredClone(next); mutate(copy);
    assert.throws(() => benchmarkSuiteCohorts(spec, copy, plan, helpers));
  }
});

test('epoch views retain 36 original cells, exact final states and grouped timing semantics', () => {
  const { spec, document } = load(3), views = epochPrefixCohorts(spec, document, helpers);
  assert.equal(views.length, 4);
  assert.equal(views.reduce((sum, view) => sum + view.rows.length * 3, 0), 36);
  for (const view of views) for (const row of view.rows) {
    const workload = view.id.slice(spec.id.length + 1);
    const samples = document.results.filter((cell) => cell.workload === workload && cell.variant === row.id);
    assert.deepEqual(row.experiment, document.experiments[`${workload}/${row.id}`]);
    assert.equal(row.metrics.preparationCpu, median(samples.map((cell) => cell.provider_birth_cpu_seconds.preparation)));
    assert.equal(row.metrics.aggregateCpu, median(samples.map((cell) => cell.aggregate_cleanup_cpu_seconds)));
    assert.equal(row.clientMemory.maxPeakBytes, Math.max(...samples.map((cell) => cell.client.client_peak_rss_bytes)));
    assert.equal(row.metrics.ttft, null, 'Queue/setup-inclusive first text is not online TTFT');
    assert.equal(row.metrics.decodeTps, null, 'Text deltas are not authoritative ordinary token intervals');
    assert.equal(row.scopedTimings.firstTextSeconds, median(samples.map((cell) => median(cell.client.responses.map((response) => response.first_text_seconds)))));
    assert.equal(row.scopedTimings.textDeltaDecodeTps, median(samples.map((cell) => median(cell.client.responses.map((response) => response.decode_tps)))));
    assert.equal(row.metrics.totalPeak, null, 'Independent peaks cannot be summed');
    assert.equal(row.metrics.baseNetwork, null);
    if (['shared', 'unrelated'].includes(workload)) {
      assert.equal(row.metrics.full, null);
      assert.equal(row.metrics.requestTps, null);
      assert.equal(view.identity.outputs, 16);
      assert.match(view.measurement.requestScope, /group/);
    }
  }
  for (const mutate of [
    (d) => { d.complete = false; },
    (d) => { d.results.pop(); },
    (d) => { d.memory_guard.failed = true; },
    (d) => { d.results[1].client.responses[0].state.bits = '0'.repeat(64); },
    (d) => { d.results[1].client.cleanup.known_private_containers_empty = false; },
    (d) => { d.results[1].client.ledger.control_body_counter_set_present = false; },
    (d) => { d.results[1].wan.conditions_digest = '0'.repeat(64); },
    (d) => { d.results[1].client.ledger.body_counter_coverage = 'different coverage'; },
    (d) => { d.results[1].client.ledger.measurement_window = 'different window'; },
    (d) => { d.results[6].client.audit.inflight_joins = 0; },
    (d) => { d.results[3].client.audit.decode_epoch_pauses = 0; },
    (d) => { d.experiments['full/epochs'].pipeline.components.quantization.params.activation_bits = 4; },
  ]) {
    const copy = structuredClone(document); mutate(copy);
    assert.throws(() => epochPrefixCohorts(spec, copy, helpers));
  }
});

test('attribution hook rejects TCP splits and never derives historical base/token values', () => {
  assert.deepEqual(applyTrafficAttribution({ totalNetwork: 4 }, {}), { totalNetwork: 4, baseNetwork: null, tokenNetwork: null });
  assert.throws(() => applyTrafficAttribution({}, {}, () => ({ scope: 'tcp_stream_payload', validated: true, complete: true, baseMbPerRequest: 1, outputDependentMbPerToken: 2 })));
  assert.throws(() => applyTrafficAttribution({}, {}, () => ({ scope: 'covered_application_bodies', validated: true, complete: false, baseMbPerRequest: 1, outputDependentMbPerToken: 2 })));
  assert.throws(() => applyTrafficAttribution({}, {}, () => ({ scope: 'covered_application_bodies', validated: true, complete: true, baseMbPerRequest: 1, outputDependentMbPerToken: 2 })),
    /canonical application bodies/);
});

function attributionFixture(complete = false) {
  const schema = 'pllm.benchmark_traffic_decomposition.v1', source = 'sdk_audit_phase_and_prepared_row_payload_counters_v1';
  const coverage = 'prepared_protocol_lifecycle_controls_and_correction_acks_v3';
  const partition = (total, base, output) => {
    const residual = total - base - output;
    return { schema, status: residual === 0 ? 'Attributed' : 'Partial', total_all_link_body_bytes: total,
      base_all_link_body_bytes: residual === 0 ? base : null, output_dependent_all_link_body_bytes: residual === 0 ? output : null,
      known_base_all_link_body_bytes: base, known_output_dependent_all_link_body_bytes: output,
      unattributed_all_link_body_bytes: residual, unused_output_material_body_bytes: null,
      conservation_checked: true, attribution_valid: true, headline_eligible: true };
  };
  const rates = (window, tokens, input = 5) => ({ ...window, generated_output_tokens: tokens, input_tokens: input, output_allowance: 8,
    base_all_link_mb_per_request: window.base_all_link_body_bytes === null || !tokens ? null : window.base_all_link_body_bytes / 1e6,
    incremental_all_link_mb_per_generated_output: window.output_dependent_all_link_body_bytes === null || !tokens ? null : window.output_dependent_all_link_body_bytes / tokens / 1e6 });
  const ledger = (edges, attribution) => ({ schema: 'pllm.topology_body_accounting.v3', body_counter_coverage: coverage,
    tracked_body_counter_set_present: true, control_body_counter_set_present: true, correction_ack_counter_set_present: true,
    body_bytes_by_edge: edges, all_link_serialized_body_bytes: edges.reduce((sum, edge) => sum + edge.serialized_body_bytes, 0),
    online_all_link_serialized_body_bytes: edges.filter((edge) => edge.phase === 'online').reduce((sum, edge) => sum + edge.serialized_body_bytes, 0),
    traffic_attribution: attribution });
  const startupAttribution = { ...partition(10, 10, 0), source, components_body_bytes: { setup_and_delivery: 10 } };
  const startupLedger = ledger([{ source: 'inference', destination: 'client', phase: 'cold', serialized_body_bytes: 10 }], startupAttribution);
  const startup = { ...rates(partition(10, 10, 0), null, null), attribution: startupAttribution };
  const runs = [], ledgers = [], windows = [];
  for (let index = 0; index < 2; index++) {
    const outputs = index === 0 ? 3 : 2;
    const audit = { bundle_network_bytes: index === 0 ? 100 : 0,
      session_authorization_upload_bytes: 4, session_authorization_download_bytes: 6,
      preparation_upload_bytes: 5, preparation_download_bytes: 5, correction_push_bytes: 40,
      correction_ack_bytes: complete ? 0 : 2, correction_ack_accounted_push_bytes: 40,
      inference_upload_bytes: 20, inference_download_bytes: 30,
      generation_attribution_supported_responses: 1, generation_completed_responses: 1,
      generation_output_tokens: outputs, generation_decode_output_tokens: outputs - 1,
      generation_executed_decode_steps: outputs - 1, generation_requested_output_tokens: 8,
      generation_prefill_online_body_bytes: 30, generation_decode_online_body_bytes: 20,
      preparation_prefill_material_body_bytes: complete ? 20 : 10,
      preparation_decode_material_body_bytes: complete ? 30 : 20, preparation_attributed_body_bytes: 50,
      preparation_prefill_rows_issued: 1, preparation_decode_rows_issued: 1, preparation_padding_rows_issued: 1,
      prepared_stage_rows_issued: 3, prepared_stage_rows_reserved: 3, prepared_stage_rows_claimed: 3, prepared_stage_rows_burned: 0 };
    const edge = (source, destination, phase, bytes) => ({ source, destination, phase, serialized_body_bytes: bytes });
    const edges = [edge('client', 'preparation', 'offline', 9), edge('preparation', 'client', 'offline', 11),
      edge('preparation', 'inference', 'offline', 40), edge('inference', 'preparation', 'offline', audit.correction_ack_bytes),
      edge('inference', 'client', 'cold', audit.bundle_network_bytes), edge('client', 'inference', 'online', 20), edge('inference', 'client', 'online', 30)];
    for (const role of ['inference', 'preparation']) {
      audit[`${role}_control_upload_bytes`] = complete ? 0 : 1;
      audit[`${role}_control_download_bytes`] = complete ? 0 : 1;
      edges.push(edge('client', role, 'control', audit[`${role}_control_upload_bytes`]), edge(role, 'client', 'control', audit[`${role}_control_download_bytes`]));
    }
    const components = { setup_and_delivery: audit.bundle_network_bytes + 10, prefill_online: 30, decode_online: 20,
      prefill_material: audit.preparation_prefill_material_body_bytes, decode_material: audit.preparation_decode_material_body_bytes };
    const total = edges.reduce((sum, edge) => sum + edge.serialized_body_bytes, 0), base = components.setup_and_delivery + components.prefill_online + components.prefill_material;
    const output = components.decode_online + components.decode_material;
    const attribution = { ...partition(total, base, output), source, components_body_bytes: components,
      generated_output_tokens: outputs, output_allowance: 8 };
    ledgers.push(ledger(edges, attribution));
    windows.push({ ...rates(partition(total, base, output), outputs), cold: index === 0, attribution,
      known_output_dependent_all_link_mb_per_generated_output: output / outputs / 1e6 });
    runs.push({ status: 'completed', cold: index === 0, tokens: { authoritative: true, input_tokens: 5, output_tokens: outputs }, max_output_tokens: 8, privacy: audit });
  }
  // Startup has no generated response, but its directed audit snapshot is evidence.
  const startupAudit = Object.fromEntries(Object.keys(runs[0].privacy).map((key) => [key, 0]));
  startupAudit.bundle_network_bytes = 10;
  startupAttribution.audit_counters = startupAudit;
  startupAttribution.generated_output_tokens = 0;
  startupAttribution.output_allowance = 0;
  Object.assign(startupAttribution.components_body_bytes,
    { prefill_online: 0, decode_online: 0, prefill_material: 0, decode_material: 0 });
  startupLedger.body_bytes_by_edge = ledgers[0].body_bytes_by_edge.map((edge) => ({ ...edge,
    serialized_body_bytes: edge.phase === 'cold' ? 10 : 0 }));
  const cold = rates(partition(10 + windows[0].total_all_link_body_bytes, 10 + windows[0].known_base_all_link_body_bytes,
    windows[0].known_output_dependent_all_link_body_bytes), 3);
  const lifecycle = partition(10 + windows.reduce((sum, window) => sum + window.total_all_link_body_bytes, 0),
    10 + windows.reduce((sum, window) => sum + window.known_base_all_link_body_bytes, 0), windows.reduce((sum, window) => sum + window.known_output_dependent_all_link_body_bytes, 0));
  const decomposition = { schema, source, scope: 'measured directed all-link serialized application bodies; not TCP stream or physical wire',
    byte_unit: 'decimal MB (1000000 bytes)', tcp_stream_bytes: null, physical_wire_bytes: null,
    body_counter_coverage: coverage, windows_aligned: true,
    lifecycle_scope: 'startup charged once to first warmup or first measured request; repeats exclude startup',
    startup, cold_first_request: cold, warmups: [], runs: windows, repeat_requests: windows.slice(1), covered_lifecycle: lifecycle,
    rate_scope: 'workload-specific output-dependent average, not constant next-token marginal cost' };
  return { schema_version: 'pllm.loopback_benchmark.v1', runs, warmup_runs: [], topology_accounting: { startup: startupLedger, runs: ledgers, warmups: [] },
    traffic_decomposition: decomposition, communication_per_token: { traffic_decomposition: decomposition } };
}

test('canonical partial application bodies disclose conserved windows without complete or TCP headlines', () => {
  const report = attributionFixture(), cold = applicationBodyAttribution({ report, requestIndex: 0 });
  const repeat = applicationBodyAttribution({ report, requestIndex: 1 });
  assert.equal(cold.complete, false);
  assert.equal(cold.disclosure.totalBytes, report.topology_accounting.startup.all_link_serialized_body_bytes + report.topology_accounting.runs[0].all_link_serialized_body_bytes);
  assert.equal(repeat.disclosure.totalBytes, report.topology_accounting.runs[1].all_link_serialized_body_bytes);
  for (const selected of [cold, repeat]) assert.equal(selected.disclosure.knownBaseBytes + selected.disclosure.knownOutputDependentBytes + selected.disclosure.residualBytes, selected.disclosure.totalBytes);
  assert.equal(repeat.disclosure.generatedOutputTokens, 2);
  assert.deepEqual(applyTrafficAttribution({ totalNetwork: 9 }, { report, requestIndex: 0 }), { totalNetwork: 9, baseNetwork: null, tokenNetwork: null });
  assert.equal(applicationBodyAttribution({ report: load(0).document.results[0].execution.details.report, requestIndex: 0 }), null);
});

test('complete canonical application-body heads use actual outputs and charge startup once', () => {
  const report = attributionFixture(true);
  for (const requestIndex of [0, 1]) {
    const split = applicationBodyAttribution({ report, requestIndex });
    assert.equal(split.complete, true);
    const values = applyTrafficAttribution({ totalNetwork: 9 }, { report, requestIndex });
    assert.equal(values.totalNetwork, 9);
    assert.equal(values.baseNetwork, split.disclosure.knownBaseBytes / 1e6);
    assert.equal(values.tokenNetwork, split.disclosure.knownOutputDependentBytes / report.runs[requestIndex].tokens.output_tokens / 1e6);
  }
  assert.throws(() => applyTrafficAttribution({}, { report, requestIndex: 0 }, () => ({ scope: 'tcp_stream_payload', complete: true, validated: true })), /canonical/);
});

function recombineStartupFixture(report) {
  const document = report.traffic_decomposition, windows = [document.startup, ...document.runs];
  const merge = (target, selected, tokens = null) => {
    const total = selected.reduce((sum, window) => sum + window.total_all_link_body_bytes, 0);
    const base = selected.reduce((sum, window) => sum + window.known_base_all_link_body_bytes, 0);
    const output = selected.reduce((sum, window) => sum + window.known_output_dependent_all_link_body_bytes, 0);
    const eligible = selected.every((window) => window.headline_eligible);
    const complete = eligible && base + output === total;
    Object.assign(target, { total_all_link_body_bytes: total, known_base_all_link_body_bytes: base,
      known_output_dependent_all_link_body_bytes: output, unattributed_all_link_body_bytes: total - base - output,
      headline_eligible: eligible, status: complete ? 'Attributed' : base + output > 0 ? 'Partial' : 'Unattributed',
      base_all_link_body_bytes: complete ? base : null, output_dependent_all_link_body_bytes: complete ? output : null });
    if (tokens !== null) Object.assign(target, { base_all_link_mb_per_request: complete ? base / 1e6 : null,
      incremental_all_link_mb_per_generated_output: complete ? output / tokens / 1e6 : null });
  };
  merge(document.cold_first_request, windows.slice(0, 2), report.runs[0].tokens.output_tokens);
  merge(document.covered_lifecycle, windows);
}

test('startup raw audit rejects consistently rewritten cached partitions and ledgers', () => {
  const mutations = [
    (r) => {
      const startup = r.traffic_decomposition.startup, attribution = startup.attribution;
      for (const partition of [startup, attribution]) {
        partition.known_base_all_link_body_bytes = partition.base_all_link_body_bytes = 9;
        partition.known_output_dependent_all_link_body_bytes = partition.output_dependent_all_link_body_bytes = 1;
      }
      attribution.components_body_bytes.setup_and_delivery = 9;
      attribution.components_body_bytes.decode_material = 1;
      recombineStartupFixture(r);
    },
    (r) => {
      const ledger = r.topology_accounting.startup, startup = r.traffic_decomposition.startup;
      ledger.body_bytes_by_edge.find((edge) => edge.phase === 'cold').serialized_body_bytes++;
      ledger.all_link_serialized_body_bytes++;
      for (const partition of [startup, startup.attribution]) {
        partition.total_all_link_body_bytes++;
        partition.known_base_all_link_body_bytes++;
        partition.base_all_link_body_bytes++;
      }
      startup.attribution.components_body_bytes.setup_and_delivery++;
      recombineStartupFixture(r);
    },
    (r) => { r.traffic_decomposition.startup.attribution.audit_counters.prepared_stage_rows_burned++; },
    (r) => { r.traffic_decomposition.startup.attribution.audit_counters.preparation_prefill_rows_issued++; },
    (r) => { r.traffic_decomposition.startup.attribution.audit_counters.generation_output_tokens++; },
    (r) => { r.traffic_decomposition.startup.attribution.audit_counters = []; },
  ];
  for (const mutate of mutations) {
    const report = attributionFixture(true); mutate(report);
    assert.throws(() => applicationBodyAttribution({ report, requestIndex: 0 }));
  }
});

test('zero-online startup validates with zero generated responses and neutral legacy zero totals', () => {
  const report = attributionFixture(true);
  const startup = report.traffic_decomposition.startup;
  assert.equal(startup.attribution.audit_counters.generation_completed_responses, 0);
  assert.equal(applicationBodyAttribution({ report, requestIndex: 0 }).complete, true);
  delete startup.attribution.audit_counters;
  for (const partition of [startup, startup.attribution]) {
    partition.total_all_link_body_bytes = partition.known_base_all_link_body_bytes = partition.base_all_link_body_bytes = 0;
  }
  startup.attribution.components_body_bytes.setup_and_delivery = 0;
  report.topology_accounting.startup.all_link_serialized_body_bytes = 0;
  report.topology_accounting.startup.body_bytes_by_edge.forEach((edge) => { edge.serialized_body_bytes = 0; });
  recombineStartupFixture(report);
  const split = applicationBodyAttribution({ report, requestIndex: 0 });
  assert.equal(split.complete, true);
  assert.equal(split.disclosure.startupRawAuditValidated, false);
});

test('positive cached-only historical startup cannot mint headlines but preserves partial disclosure', () => {
  const report = attributionFixture(true), startup = report.traffic_decomposition.startup;
  delete startup.attribution.audit_counters;
  const cachedComplete = applicationBodyAttribution({ report, requestIndex: 0 });
  assert.equal(cachedComplete.complete, false);
  assert.equal(cachedComplete.baseMbPerRequest, null);
  assert.equal(cachedComplete.outputDependentMbPerToken, null);
  assert.equal(cachedComplete.disclosure.status, 'Partial');
  assert.equal(applicationBodyAttribution({ report, requestIndex: 1 }).complete, true);
  assert.deepEqual(applyTrafficAttribution({ totalNetwork: 9 }, { report, requestIndex: 0 }),
    { totalNetwork: 9, baseNetwork: null, tokenNetwork: null });
  for (const partition of [startup, startup.attribution]) {
    partition.known_base_all_link_body_bytes = 7;
    partition.unattributed_all_link_body_bytes = 3;
    partition.base_all_link_body_bytes = partition.output_dependent_all_link_body_bytes = null;
    partition.status = 'Partial';
  }
  startup.attribution.components_body_bytes.setup_and_delivery = 7;
  recombineStartupFixture(report);
  const partial = applicationBodyAttribution({ report, requestIndex: 0 });
  assert.equal(partial.complete, false);
  assert.equal(partial.disclosure.residualBytes, 3);
  assert.equal(partial.disclosure.knownBaseBytes, 7 + report.traffic_decomposition.runs[0].known_base_all_link_body_bytes);
  assert.equal(partial.disclosure.totalBytes, partial.disclosure.knownBaseBytes + partial.disclosure.knownOutputDependentBytes + 3);
});

test('raw startup controls remain residual and missing row conservation cannot earn headlines', () => {
  const report = attributionFixture(true), startup = report.traffic_decomposition.startup;
  const ledger = report.topology_accounting.startup;
  startup.attribution.audit_counters.inference_control_upload_bytes = 3;
  ledger.body_bytes_by_edge.find((edge) => edge.source === 'client' && edge.destination === 'inference' && edge.phase === 'control')
    .serialized_body_bytes = 3;
  ledger.all_link_serialized_body_bytes += 3;
  for (const partition of [startup, startup.attribution]) {
    partition.total_all_link_body_bytes += 3;
    partition.unattributed_all_link_body_bytes = 3;
    partition.base_all_link_body_bytes = partition.output_dependent_all_link_body_bytes = null;
    partition.status = 'Partial';
  }
  recombineStartupFixture(report);
  const partial = applicationBodyAttribution({ report, requestIndex: 0 });
  assert.equal(partial.complete, false);
  assert.equal(partial.disclosure.startupRawAuditValidated, true);
  assert.equal(partial.disclosure.residualBytes, 3);
  assert.equal(partial.baseMbPerRequest, null);

  const missingRows = attributionFixture(true), missingStartup = missingRows.traffic_decomposition.startup;
  for (const name of ['reserved', 'claimed', 'burned']) delete missingStartup.attribution.audit_counters[`prepared_stage_rows_${name}`];
  for (const partition of [missingStartup, missingStartup.attribution]) {
    partition.headline_eligible = false;
    partition.status = 'Partial';
    partition.base_all_link_body_bytes = partition.output_dependent_all_link_body_bytes = null;
  }
  recombineStartupFixture(missingRows);
  assert.equal(applicationBodyAttribution({ report: missingRows, requestIndex: 0 }).complete, false);
});

test('canonical attribution rejects stale counters, wrong windows, output denominators and startup duplication', () => {
  for (const mutate of [
    (r) => { r.traffic_decomposition.schema = 'unknown'; },
    (r) => { r.traffic_decomposition.tcp_stream_bytes = 123; },
    (r) => { r.traffic_decomposition.windows_aligned = false; },
    (r) => { r.traffic_decomposition.body_counter_coverage = 'old'; },
    (r) => { r.runs[0].tokens.authoritative = false; },
    (r) => { r.runs[0].privacy.prepared_stage_rows_burned++; },
    (r) => { r.traffic_decomposition.runs[0].generated_output_tokens++; },
    (r) => { r.traffic_decomposition.runs[0].known_output_dependent_all_link_mb_per_generated_output *= 2; },
    (r) => { r.traffic_decomposition.cold_first_request.total_all_link_body_bytes += 10; r.traffic_decomposition.cold_first_request.unattributed_all_link_body_bytes += 10; },
    (r) => { r.traffic_decomposition.repeat_requests = [r.traffic_decomposition.cold_first_request]; },
    (r) => { r.traffic_decomposition.runs[0].known_base_all_link_body_bytes++; },
    (r) => { r.runs[0].privacy.preparation_prefill_material_body_bytes++; },
    (r) => { r.runs[0].privacy.generation_decode_online_body_bytes++; },
    (r) => { r.runs[0].privacy.inference_upload_bytes++; },
    (r) => { r.traffic_decomposition = structuredClone(r.traffic_decomposition); r.traffic_decomposition.scope = 'TCP streams'; },
  ]) {
    const report = attributionFixture(); mutate(report);
    assert.throws(() => applicationBodyAttribution({ report, requestIndex: 0 }));
  }
  const report = attributionFixture(true);
  report.traffic_decomposition.runs[0].incremental_all_link_mb_per_generated_output *= 2;
  assert.throws(() => applicationBodyAttribution({ report, requestIndex: 0 }), /output rate/);
});

test('all metrics sort unknown last in both directions; leaders remain local', () => {
  const views = specs.flatMap((spec) => adaptedCohorts(spec, read, helpers));
  assert.equal(views.length, 16);
  for (const view of views) for (const metric of metrics) {
    const table = sotaTable(view, 'Prepared', [metric]);
    assert.ok(table.leaders[metric.id].every((id) => view.rows.some((row) => row.id === id)));
    const rows = [...view.rows, { id: 'unknown', metrics: { [metric.id]: null } }];
    for (const direction of ['ascending', 'descending']) {
      const sorted = sortMetricRows(rows, metric.id, direction);
      assert.equal(sorted.at(-1).metrics[metric.id], null);
    }
  }
});

test('reconciled registry has no duplicate confirmation path and retains isolated history', () => {
  const reports = new Set(specs.map((spec) => spec.report));
  assert.ok(manifest.cohorts.every((cohort) => !reports.has(cohort.report)));
  const card = buildScorecard();
  assert.equal(card.cohorts.filter((view) => ['benchmark-suite-v1', 'epoch-prefix-v1'].includes(view.adapter)).length, 16);
  for (const cohort of card.cohorts.filter((view) => view.adapter === 'epoch-prefix-v1')) {
    assert.ok(cohort.rows.every((row) => row.metrics.ttft === null && row.metrics.decodeTps === null));
  }
  for (const cohort of card.cohorts.filter((view) => view.adapter === 'pllm.benchmark_run.v1')) {
    assert.ok(cohort.rows.every((row) => row.recordedCase.build.expected_digest === cohort.identity.build));
    assert.match(cohort.measurement.clientRssLabel, /scenario-lifetime/);
    assert.equal(cohort.measurement.phase, 'cold + repeat');
    assert.equal(cohort.identity.tokenCountScope, 'cold + repeat scenario totals');
    assert.equal(cohort.identity.requestCount, 2);
    for (const key of ['inputs', 'outputs', 'cap']) {
      assert.equal(cohort.identity[key], cohort.identity.workload.reduce((sum, request) => sum + request[key], 0));
    }
    assert.equal(cohort.identity.outputs, 16);
    assert.ok(cohort.rows.every((row) => row.metrics.ttft === null && row.metrics.decodeTps === null));
  }
  for (const key of ['baseNetwork', 'tokenNetwork']) assert.equal(metrics.find((metric) => metric.id === key).accountingPlane, 'covered_application_bodies');
  assert.equal(metrics.find((metric) => metric.id === 'totalNetwork').accountingPlane, 'tcp_stream_payload');
});

test('completed five-way screen retains separate request windows and partial real-body partitions', () => {
  const spec = specs.find((entry) => entry.id === 'qwen25-sota-combinations-screen');
  const views = adaptedCohorts({ ...spec, frozen_factory: manifest.historical_factory }, read, helpers);
  assert.equal(views.length, 4);
  assert.deepEqual(views.map((view) => [view.identity.inputs, view.identity.requestIndex]), [[150, 0], [150, 1], [288, 0], [288, 1]]);
  for (const view of views) {
    assert.equal(view.evidenceTier, 'screen');
    assert.equal(view.rows.length, 5);
    assert.equal(view.identity.outputs, 8);
    assert.match(view.measurement.clientRssLabel, /scenario-lifetime/);
    const table = sotaTable(view, 'Prepared', metrics);
    assert.deepEqual(table.leaders.baseNetwork, []);
    assert.deepEqual(table.leaders.tokenNetwork, []);
    for (const row of view.rows) {
      assert.equal(row.sampleCount, 3);
      assert.equal(row.researchSelector, null, 'Tiling is an installed Pipeline choice');
      assert.equal(row.metrics.baseNetwork, null);
      assert.equal(row.metrics.tokenNetwork, null);
      assert.equal(row.applicationBodyAttribution.length, 3);
      for (const sample of row.applicationBodyAttribution) {
        assert.equal(sample.status, 'Partial');
        assert.equal(sample.accountingPlane, 'covered_application_bodies');
        assert.equal(sample.generatedOutputTokens, 8);
        assert.ok(sample.residualBytes > 0);
        assert.equal(sample.knownBaseBytes + sample.knownOutputDependentBytes + sample.residualBytes, sample.totalBytes);
        assert.notEqual(sample.totalBytes / 1e6, row.metrics.totalNetwork);
      }
      const inventory = row.pipeline.components.inventory.params;
      assert.equal(inventory.stage_window ?? 1, view.identity.inputs === 150 ? 4 : 1);
      if (view.identity.requestIndex > 0) assert.equal(row.metrics.preparationCpu, null);
    }
  }
});

test('registry evidence tiers are explicit and never inferred from repetition', () => {
  const entries = [...manifest.cohorts, ...manifest.adapted_cohorts];
  const confirmations = ['qwen25-native-wan-frontier-150-8', 'qwen25-sdk-finalists-150-8',
    'qwen25-maintained-digit8-confirm', 'qwen25-mlp-tiling-confirm-c',
    'qwen25-all-link-placement-confirm', 'qwen25-epoch-prefix-confirm'];
  assert.ok(entries.every((cohort) => ['confirmation', 'screen', 'observation'].includes(cohort.evidenceTier)));
  assert.deepEqual(entries.filter((cohort) => cohort.evidenceTier === 'confirmation').map((cohort) => cohort.id).sort(), confirmations.sort());
  assert.ok(entries.filter((cohort) => cohort.id.includes('screen')).every((cohort) => cohort.evidenceTier === 'screen'));
  assert.ok(entries.filter((cohort) => /^qwen25-tiling-[ab]-/.test(cohort.id)).every((cohort) => cohort.evidenceTier === 'observation'));
  for (const cohort of buildScorecard().cohorts) {
    const registered = entries.find((entry) => cohort.id === entry.id || cohort.id.startsWith(`${entry.id}-`));
    assert.equal(cohort.evidenceTier, registered.evidenceTier);
  }
});

test('frozen loader preserves original BuildRef, target and epoch driver lifecycle without SDK execution', () => {
  const script = `
import importlib.util, json, sys, types
from pathlib import Path
spec = importlib.util.spec_from_file_location("frozen", "benchmarks/research/historical_evidence.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
source = "research-prepared-study-qwen25-confirm"
document = module.source_document(source)
contract = module.recorded_contract(source, "prepared/digit8")
expected = next(case for case in document["suite"]["cases"] if case["name"] == "prepared/digit8")
assert contract["case"] == expected
assert contract["lifecycle"] == expected["target"]["parameters"]["scenario"]
assert contract["sampling"] == document["suite"]["sampling"]
class FrozenCase:
    @staticmethod
    def from_spec(value):
        return types.SimpleNamespace(target=value["target"], build=value["build"])
fake_pllm = types.ModuleType("pllm")
fake_pllm.__path__ = []
fake_pllm.BenchmarkCase = FrozenCase
sys.modules["pllm"] = fake_pllm
sys.modules["pllm.metrics"] = types.SimpleNamespace(BenchmarkCase=FrozenCase)
assert module.case(source, "prepared/digit8").build == expected["build"]
assert module.target(source, "prepared/digit8") == expected["target"]
epoch = module.recorded_contract("epoch-prefix-sdk-qwen25-2026-10-06", "shared/inflight")
assert epoch["requires_original_measurement_driver"] is True
assert epoch["lifecycle"]["variant"] == "inflight"
assert epoch["lifecycle"]["response_statuses"] == ["incomplete", "incomplete"]
assert epoch["build"]["expected_digest"] == module.source_document("epoch-prefix-sdk-qwen25-2026-10-06")["build"]["expected_digest"]
screen = module.source_document("research-sota-combinations-qwen25")
for original in screen["suite"]["cases"]:
    assert module.recorded_contract("research-sota-combinations-qwen25", original["name"])["case"] == original
print("frozen contracts passed")
`;
  const result = spawnSync('python3', ['-c', script], { cwd: root, encoding: 'utf8', timeout: 10000 });
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /frozen contracts passed/);
  const { spec } = load(0);
  const frozen = { ...spec, frozen_factory: manifest.historical_factory };
  const [view] = adaptedCohorts(frozen, read, helpers);
  assert.ok(view.rows.every((row) => row.file === manifest.historical_factory.file));
  assert.throws(() => adaptedCohorts(frozen, (file) => file === manifest.historical_factory.file ? Buffer.from('changed') : read(file), helpers), /factory changed/);
});
