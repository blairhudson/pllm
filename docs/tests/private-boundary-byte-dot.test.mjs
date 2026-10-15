/** Synthetic negative controls plus read-only registered evidence replay. */
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { privateBoundaryByteDotCohorts } from '../scripts/research-private-boundary-tiers.mjs';
import { adaptedCohorts } from '../scripts/research-scorecard-adapters.mjs';
import { executionPlacement, metrics, providerPeak } from '../scripts/research-scorecard.mjs';

const root = fileURLToPath(new URL('../../', import.meta.url));
const read = (file) => fs.readFileSync(root + file);
const hash = (value) => createHash('sha256').update(value).digest('hex');
const names = ['parallel-residues-native', 'parallel-residues-byte-dot'];
const driver = 'scripts/benchmark_private_boundary_byte_dot.py';
const factory = 'benchmarks/research/private_boundary_byte_dot.py';
const sourceFiles = [driver, factory, 'benchmarks/research/private_boundary_decode_confirmation.py',
  'benchmarks/research/private_boundary_decode.py', 'benchmarks/research/private_boundary_tiers.py',
  'benchmarks/research/historical_evidence.py', 'benchmarks/research/sota_combinations.py',
  'scripts/benchmark_sota_combinations.py', 'examples/prepared_sdk.py', 'docs/evidence/private-boundary-qwen-control.json'];
const priorities = ['process.peak_rss', 'client.cold_cpu', 'preparation.cold_cpu', 'preparation.peak_rss',
  'response.ttft', 'response.decode_tps', 'network.all_link_stream_bytes', 'request.0.full',
  'client.peak_rss', 'network.covered_body_bytes', 'target.latency', 'topology.cold_cpu'];
const canonical = (v, spaced = false, key = '') => {
  const comma = spaced ? ', ' : ',', colon = spaced ? ': ' : ':';
  if (Array.isArray(v)) return `[${v.map((x) => canonical(x, spaced)).join(comma)}]`;
  if (v && typeof v === 'object') return `{${Object.keys(v).sort().map((k) => `${JSON.stringify(k)}${colon}${canonical(v[k], spaced, k)}`).join(comma)}}`;
  if (Number.isInteger(v) && ['download_mbps', 'upload_mbps'].includes(key)) return `${v}.0`;
  return JSON.stringify(v);
};
const digest = (v) => hash(canonical(v));
const median = (v) => v.includes(null) ? null : [...v].sort((a, b) => a - b)[1];
const pick = (v, keys) => Object.fromEntries(keys.map((k) => [k, v[k]]));
const metric = (row, id) => row.metrics.find((m) => m.id === id);
const reportOf = (row) => row.execution.details.report;

test('registered installed SDK screen uses production dispatch and retains non-winning decode', () => {
  const spec = JSON.parse(read('docs/data/research/benchmark-cohorts.json')).adapted_cohorts
    .find((entry) => entry.id === 'qwen25-private-boundary-byte-dot-sdk-150-8');
  assert.equal(spec.evidenceTier, 'screen');
  const [cohort] = adaptedCohorts(spec, read, { executionPlacement, metrics, providerPeak });
  assert.equal(cohort.rows.length, 2);
  assert.ok(cohort.rows.every((row) => row.observedTiers.low));
  const summary = JSON.parse(read(spec.summary));
  assert.equal(summary.comparisons[names[1]][names[0]]['response.decode_tps'].candidate_wins, false);
  assert.ok(cohort.rows[1].metrics.decodeTps < cohort.rows[0].metrics.decodeTps);
});

// Independent test-only helper oracles. Production dispatch supplies the existing
// checkRun/checkedWan/checkedResponseBodies helpers (also covered by legacy tests).
const helpers = { executionPlacement, metrics, providerPeak,
  checkRun(run) {
    assert.equal(run.status, 'completed'); assert.equal(run.tokens.authoritative, true);
    assert.equal(run.privacy.plaintext_prompt_bytes_sent, 0);
    assert.equal(run.privacy.plaintext_token_ids_sent, 0);
    assert.equal(run.privacy.preparation_requests_during_online, 0);
  },
  checkedWan(wan) {
    assert.equal(wan.schema, 'pllm.native_wan.v1'); assert.equal(wan.enforced, true);
    const edges = Object.entries(wan.directed_stream_bytes);
    assert.ok(edges.every(([, n]) => Number.isSafeInteger(n) && n >= 0));
    assert.equal(edges.reduce((s, [, n]) => s + n, 0), wan.total_stream_bytes);
    assert.equal(edges.filter(([edge]) => edge.split('->').includes('client')).reduce((s, [, n]) => s + n, 0), wan.client_stream_bytes);
    return wan;
  },
  checkedResponseBodies(report, values, boundary) {
    assert.equal(boundary, true);
    const ledgers = [report.topology_accounting.startup, report.topology_accounting.runs[0]];
    const tcp = report.native_network_accounting.samples;
    const bounds = [tcp.startup.wan_emulation.directed_stream_bytes,
      Object.fromEntries(Object.entries(tcp.runs[0].after.wan_emulation.directed_stream_bytes).map(([edge, n]) =>
        [edge, n - (tcp.runs[0].before.wan_emulation.directed_stream_bytes[edge] ?? 0)]))];
    for (const [i, ledger] of ledgers.entries()) {
      const p = i ? report.runs[0].privacy : ledger.traffic_attribution.audit_counters;
      const directed = {};
      for (const edge of ledger.body_bytes_by_edge) {
        const link = `${edge.source}->${edge.destination}`;
        directed[link] = (directed[link] ?? 0) + edge.serialized_body_bytes;
        if (edge.source.startsWith('boundary_') || edge.destination.startsWith('boundary_')) {
          const role = edge.source.startsWith('boundary_') ? edge.source : edge.destination;
          assert.equal(edge.serialized_body_bytes, p[`role_link.${role}.${edge.phase}_${edge.source === 'client' ? 'upload' : 'download'}_bytes`]);
        }
      }
      assert.ok(Object.entries(directed).every(([link, n]) => n <= (bounds[i][link] ?? 0)));
      for (const [key, filter] of [
        ['all_link_serialized_body_bytes', () => true],
        ['client_serialized_body_bytes', (e) => [e.source, e.destination].includes('client')],
        ['online_all_link_serialized_body_bytes', (e) => e.phase === 'online'],
        ['online_client_serialized_body_bytes', (e) => e.phase === 'online' && [e.source, e.destination].includes('client')],
        ['lifecycle_control_serialized_body_bytes', (e) => e.phase === 'control'],
      ]) assert.equal(ledger[key], ledger.body_bytes_by_edge.filter(filter).reduce((s, e) => s + e.serialized_body_bytes, 0));
    }
    assert.equal(ledgers.reduce((s, l) => s + l.all_link_serialized_body_bytes, 0), values['network.covered_body_bytes']);
  },
};

function fixture() {
  const archived = JSON.parse(read('docs/evidence/private-boundary-decode-confirmation-150x8.json'));
  const templates = archived.results.filter((r) => r.execution.case.name === 'coupled-parallel-uniform_residues')
    .sort((a, b) => a.execution.replicate - b.execution.replicate);
  const suite = structuredClone(archived.suite);
  suite.name = 'private-boundary-byte-dot-v1';
  suite.cases = names.map((name, index) => {
    const cell = structuredClone(templates[0].execution.case); cell.name = name;
    if (index) cell.target.parameters.model.experiment.pipeline.components.boundary.params.head_kernel = 'byte_dot';
    return cell;
  });
  const document = { schema: 'pllm.benchmark_run.v1', complete: true, cohort: 'e'.repeat(64), suite, results: [] };
  for (const cell of suite.cases) for (const template of templates) {
    const row = structuredClone(template), e = row.execution, report = reportOf(row);
    row.limitations = ['SYNTHETIC TEST ONLY: copied retained ordinary reports; no byte-dot model execution.'];
    e.case = structuredClone(cell); e.suite_digest = digest(suite); e.case_digest = digest(cell); e.cohort = document.cohort;
    e.sampling = structuredClone(suite.sampling);
    row.configuration_digest = hash('pllm.benchmark_target.v1\0' + canonical(cell.target));
    report.experiment.pipeline_digest = hash('pllm.pipeline.v2\0' + canonical(cell.target.parameters.model.experiment.pipeline));
    document.results.push(row);
  }
  const first = document.results[0].execution, report = first.details.report, binding = first.details.response_state_audit.bindings[0];
  const summary = { schema: 'pllm.private_boundary_byte_dot.v1', tiny: false, complete: true,
    cohort: document.cohort, clients: 6, replicates: 3, output_cap: 8,
    all_logits_active_kv_positions_selections_rng_and_outputs_match: true,
    identity: { source_lock_digest: report.configuration.source_lock_digest, prompt_digest: report.configuration.prompt_digest,
      wan_emulation_digest: report.configuration.wan_emulation_digest, sampling: report.configuration.sampling,
      body: binding.numeric_body_fingerprint, tokens: report.runs[0].tokens,
      output: report.runs[0].generation.output_text_digest, build: first.build },
    checked_values: document.results.reduce((s, r) => s + r.execution.oracle.checked_values, 0),
    state_commitment: hash(canonical(first.details.response_state_audit.bindings, true)),
    maximum_observed_swap_growth_bytes: 0, groups: {}, comparisons: { [names[1]]: {} },
    body_reconciliation: { [names[1]]: [] } };
  const workFields = ['preparation_rows', 'preparation_prefill_rows_issued', 'preparation_decode_rows_issued',
    'preparation_padding_rows_issued', 'inference_stage_calls', 'prepared_stage_rows_reserved',
    'prepared_stage_rows_claimed', 'prepared_stage_rows_burned', 'prepared_stage_rows_discarded'];
  for (const name of names) {
    const rows = document.results.filter((r) => r.execution.case.name === name);
    const samples = rows.map((r) => ({ ...Object.fromEntries(r.metrics.map((m) => [m.id, m.value])),
      'topology.cold_cpu': reportOf(r).process_cpu_accounting.aggregate_cold_first_response_cpu_seconds }));
    const peak = Math.max(...samples.map((v) => v['process.peak_rss']));
    summary.groups[name] = { samples, medians: Object.fromEntries(priorities.map((id) => [id, median(samples.map((v) => v[id]))])),
      maximum_client_peak_bytes: peak, observed_tiers: { low: peak <= 256 * 2 ** 20, medium: peak <= 2 * 2 ** 30, high: peak <= 16 * 2 ** 30 },
      executed_work: rows.map((r) => ({ output_tokens: 8, sampled_selections: 8, executed_decode_steps: 7,
        terminated_before_cap: false, ...pick(reportOf(r).runs[0].privacy, workFields) })),
      cold_first_cpu_seconds_by_role: rows.map((r) => structuredClone(reportOf(r).process_cpu_accounting.cold_first_response_cpu_seconds_by_role)) };
  }
  refreshComparisons(summary);
  for (let replicate = 0; replicate < 3; replicate++) {
    const p = reportOf(document.results[replicate]).runs[0].privacy;
    summary.body_reconciliation[names[1]].push({ replicate, covered_body_delta_bytes: 0, head_reply_delta_bytes: 0,
      lifecycle_control_delta_bytes: 0, prepared_stage_reply_delta_bytes: 0,
      prepared_stage_reply_metadata_compatibility_bound_bytes: 8 * (Math.ceil(p.prepared_stage_rows_issued / 16) + p.inference_stage_calls - 1),
      bootstrap_and_metadata_residual_bytes: 0 });
  }
  const plan = { schema: 'pllm.private_boundary_byte_dot_plan.v1', tiny: false, suite: structuredClone(suite),
    source_lock_digest: report.configuration.source_lock_digest, public_input_counts: { '150': 150 },
    preflight: names.map((name) => ({ case: name, admission: { admitted: true } })),
    frozen_sources: { ...Object.fromEntries(sourceFiles.map((file) => [file, hash(read(file))])),
      'SYNTHETIC-TEST-ONLY/prompt.txt': hash('SYNTHETIC public fixture pin'),
      'SYNTHETIC-TEST-ONLY/indexed.json': hash('SYNTHETIC public tokenizer pin') } };
  const spec = { id: 'SYNTHETIC-TEST-ONLY-byte-dot', adapter: 'private-boundary-byte-dot-v1', evidenceTier: 'confirmation',
    report: 'test-only/report.json', plan: 'test-only/plan.json', summary: 'test-only/summary.json',
    replay_file: driver, replay_sha256: hash(read(driver)), frozen_factory: { file: factory, sha256: hash(read(factory)) } };
  return { document, summary, plan, spec };
}

function refreshComparisons(summary) {
  const a = summary.groups[names[0]], b = summary.groups[names[1]];
  summary.comparisons[names[1]][names[0]] = Object.fromEntries(priorities.map((id) => {
    const left = a.medians[id], right = b.medians[id], known = left !== null && right !== null, min = id !== 'response.decode_tps';
    return [id, { control: left, candidate: right, delta: known ? right - left : null,
      improvement_fraction: known && left ? (min ? 1 - right / left : right / left - 1) : null,
      candidate_wins: known ? (min ? right < left : right > left) : null,
      paired_deltas: a.samples.map((v, i) => v[id] === null || b.samples[i][id] === null ? null : b.samples[i][id] - v[id]) }];
  }));
}

function evaluate(f, changeRead = null, mutateSpec = null) {
  const files = new Map(), spec = { ...f.spec };
  for (const [field, value, pin] of [['report', f.document, 'sha256'], ['plan', f.plan, 'plan_sha256'], ['summary', f.summary, 'summary_sha256']]) {
    const raw = Buffer.from(JSON.stringify(value)); files.set(spec[field], raw); spec[pin] = hash(raw);
  }
  mutateSpec?.(spec);
  const reader = (file) => changeRead?.(file) ?? files.get(file) ?? read(file);
  return privateBoundaryByteDotCohorts(spec, f.document, f.plan, reader, helpers);
}
const candidate = (f) => f.document.results.at(-1);

test('synthetic six-client two-arm reader is read-only; never historical candidate evidence', () => {
  const f = fixture(), before = structuredClone(f), [view] = evaluate(f);
  assert.deepEqual(f, before); assert.equal(view.baseline, names[0]); assert.equal(view.rows.length, 2);
  assert.equal(view.identity.cap, 8); assert.equal(view.identity.inputs, 150);
  assert.match(view.measurement.cpu, /five role.*cold first response; cleanup excluded/);
  for (const row of view.rows) {
    assert.equal(row.sampleCount, 3);
    assert.equal(row.metrics.aggregateCpu, f.summary.groups[row.id].medians['topology.cold_cpu']);
    assert.equal(row.clientMemory.maxPeakBytes, f.summary.groups[row.id].maximum_client_peak_bytes);
    assert.equal(row.derivedMeasurementContracts[0].cleanup_included, false);
    assert.equal(row.metrics.wire, null); assert.equal(row.metrics.baseNetwork, null);
  }
});

const mutations = {
  incomplete: (f) => { f.document.complete = false; },
  missing: (f) => { f.document.results.pop(); },
  duplicate: (f) => { f.document.results[5] = structuredClone(f.document.results[4]); },
  cohort: (f) => { candidate(f).execution.cohort = 'd'.repeat(64); },
  historicalCell: (f) => { candidate(f).execution.case.name = 'coupled-parallel-uniform_residues'; },
  tiny: (f) => { f.plan.tiny = true; },
  cap: (f) => { f.summary.output_cap = 32; },
  suite: (f) => { f.plan.suite.cases.pop(); },
  preflight: (f) => { f.plan.preflight[1].admission.admitted = false; },
  frozenSource: (f) => { f.plan.frozen_sources[factory] = 'f'.repeat(64); },
  missingSource: (f) => { delete f.plan.frozen_sources['examples/prepared_sdk.py']; },
  missingArtifactPin: (f) => { delete f.plan.frozen_sources['SYNTHETIC-TEST-ONLY/prompt.txt']; },
  snapshotEscape: (f) => { f.spec.source_snapshots = {}; },
  buildAttributes: (f) => { candidate(f).execution.build.attributes.forged = true; },
  caseDigest: (f) => { candidate(f).execution.case_digest = '0'.repeat(64); },
  suiteDigest: (f) => { candidate(f).execution.suite_digest = '0'.repeat(64); },
  targetDigest: (f) => { candidate(f).configuration_digest = '0'.repeat(64); },
  pipeline: (f) => { reportOf(candidate(f)).experiment.pipeline_digest = '0'.repeat(64); },
  kernel: (f) => { candidate(f).execution.case.target.parameters.model.experiment.pipeline.components.boundary.params.head_kernel = 'existing'; },
  source: (f) => { reportOf(candidate(f)).configuration.source_lock_digest = 'f'.repeat(64); },
  state: (f) => { candidate(f).execution.details.response_state_audit.bindings[0].selections[0].logits_and_active_kv = 'f'.repeat(64); },
  rng: (f) => { candidate(f).execution.details.response_state_audit.bindings[0].selections[0].rng_after = 'f'.repeat(64); },
  history: (f) => { candidate(f).execution.details.response_state_audit.bindings[0].selections[0].cache_lengths[0]--; },
  selection: (f) => { candidate(f).execution.details.response_state_audit.bindings[0].selections[0].selected_token = 'f'.repeat(64); },
  output: (f) => { reportOf(candidate(f)).runs[0].generation.output_text_digest = 'f'.repeat(64); },
  numeric: (f) => { candidate(f).execution.comparison.numeric_digest = 'f'.repeat(64); },
  cpuOwner: (f) => { delete reportOf(candidate(f)).process_cpu_accounting.cold_first_response_cpu_seconds_by_role.boundary_b; },
  cpuSum: (f) => { reportOf(candidate(f)).process_cpu_accounting.aggregate_cold_first_response_cpu_seconds++; },
  cpuCache: (f) => { f.summary.groups[names[1]].cold_first_cpu_seconds_by_role[0].boundary_b++; },
  cpuMedian: (f) => { f.summary.groups[names[1]].medians['topology.cold_cpu']--; },
  cleanupCpu: (f) => { f.summary.groups[names[1]].samples[2]['topology.cold_cpu'] = metric(candidate(f), 'process.cpu').value; },
  metricWindow: (f) => { for (const r of f.document.results) metric(r, 'request.0.preparation').window = 'execution'; },
  nonPriorityValue: (f) => { metric(candidate(f), 'target.cpu').value++; },
  epochRaw: (f) => { reportOf(candidate(f)).runs[0].privacy.decode_epoch_pause_ns = 1000; },
  unknownMetric: (f) => { candidate(f).metrics.push({ ...candidate(f).metrics[0], id: 'forged.metric' }); },
  tpsAliases: (f) => { for (const id of ['request.0.decode_tps', 'response.decode_tps']) metric(candidate(f), id).value *= 2; },
  cachedSamples: (f) => { f.summary.groups[names[1]].samples[0]['target.cpu']++; },
  hiddenPeak: (f) => { f.summary.groups[names[1]].maximum_client_peak_bytes--; },
  smallPeak: (f) => { metric(candidate(f), 'process.peak_rss').value = 1; },
  swap: (f) => { reportOf(candidate(f)).memory_guard.maximum_host_swap_growth_bytes = 4096; },
  tcp: (f) => { reportOf(candidate(f)).native_network_accounting.samples.runs[0].after.wan_emulation.total_stream_bytes--; },
  directedTcp: (f) => { reportOf(candidate(f)).native_network_accounting.samples.runs[0].after.wan_emulation.directed_stream_bytes['inference->client']--; },
  wanRate: (f) => { reportOf(candidate(f)).native_network_accounting.samples.runs[0].after.wan_emulation.parties.boundary_b.upload.bytes_per_second++; },
  body: (f) => { reportOf(candidate(f)).topology_accounting.runs[0].all_link_serialized_body_bytes--; },
  stage: (f) => { Object.values(reportOf(candidate(f)).topology_accounting.stages.runs[0].body_bytes_by_stage_and_edge)[0]['client->inference']--; },
  material: (f) => { reportOf(candidate(f)).prepared_material_accounting.counts.claimed--; },
  headWork: (f) => { reportOf(candidate(f)).runs[0].privacy.private_boundary_head_calls--; },
  headPayload: (f) => { reportOf(candidate(f)).runs[0].privacy.private_boundary_head_download_bytes++; },
  residual: (f) => { f.summary.body_reconciliation[names[1]][0].bootstrap_and_metadata_residual_bytes++; },
  trafficResidual: (f) => { reportOf(candidate(f)).traffic_decomposition.covered_lifecycle.unattributed_all_link_body_bytes--; },
  fabricatedWin: (f) => { f.summary.comparisons[names[1]][names[0]]['topology.cold_cpu'].candidate_wins = true; },
};
for (const [name, mutate] of Object.entries(mutations)) test(`reject rehashed synthetic tamper: ${name}`, () => {
  const f = fixture(); mutate(f); assert.throws(() => evaluate(f));
});

test('report, plan, summary, driver and factory pins bind exact bytes', () => {
  const f = fixture();
  for (const field of ['sha256', 'plan_sha256', 'summary_sha256', 'replay_sha256']) {
    assert.throws(() => evaluate(f, null, (spec) => { spec[field] = '0'.repeat(64); }));
  }
  for (const file of [factory, driver]) assert.throws(() => evaluate(f, (path) => path === file ? Buffer.from('forged source') : null));
});

test('largest finalized whole-worker outlier controls tier, not median', () => {
  const f = fixture(), row = candidate(f), group = f.summary.groups[names[1]], rep = row.execution.replicate;
  metric(row, 'process.peak_rss').value = 300 * 2 ** 20;
  group.samples[rep]['process.peak_rss'] = 300 * 2 ** 20;
  group.medians['process.peak_rss'] = median(group.samples.map((s) => s['process.peak_rss']));
  group.maximum_client_peak_bytes = 300 * 2 ** 20; group.observed_tiers.low = false;
  refreshComparisons(f.summary);
  const [view] = evaluate(f);
  assert.equal(view.rows[1].observedTiers.low, false);
  assert.equal(view.rows[1].clientMemory.maxPeakBytes, 300 * 2 ** 20);
});

function changeServerNsEnvelope(f, delta) {
  const row = candidate(f), report = reportOf(row), rep = row.execution.replicate;
  report.runs[0].privacy.inference_download_bytes += delta;
  const ledger = report.topology_accounting.runs[0];
  ledger.body_bytes_by_edge.find((e) => e.source === 'inference' && e.destination === 'client' && e.phase === 'online').serialized_body_bytes += delta;
  for (const key of ['all_link_serialized_body_bytes', 'client_serialized_body_bytes',
    'online_all_link_serialized_body_bytes', 'online_client_serialized_body_bytes']) ledger[key] += delta;
  const stages = report.topology_accounting.stages.runs[0];
  Object.values(stages.body_bytes_by_stage_and_edge)[0]['inference->client'] += delta;
  stages.measured_body_bytes_by_edge['inference->client'] += delta;
  stages.expected_stage_body_bytes_by_edge['inference->client'] += delta;
  const wan = report.native_network_accounting.samples.runs[0].after.wan_emulation;
  wan.directed_stream_bytes['inference->client'] += delta; wan.total_stream_bytes += delta; wan.client_stream_bytes += delta;
  wan.parties.inference.upload.admitted_bytes += delta; wan.parties.client.download.admitted_bytes += delta;
  const traffic = report.traffic_decomposition;
  for (const t of [ledger.traffic_attribution, traffic.runs[0], traffic.runs[0].attribution,
    traffic.cold_first_request, traffic.covered_lifecycle]) {
    t.total_all_link_body_bytes += delta; t.unattributed_all_link_body_bytes += delta;
  }
  ledger.traffic_attribution.audit_counters.inference_download_bytes += delta;
  traffic.runs[0].attribution.audit_counters.inference_download_bytes += delta;
  row.execution.details.traffic_decomposition = structuredClone(traffic);
  const group = f.summary.groups[names[1]];
  for (const id of ['network.covered_body_bytes', 'network.all_link_stream_bytes', 'network.unattributed_body_bytes']) {
    metric(row, id).value += delta; group.samples[rep][id] += delta;
    if (priorities.includes(id)) group.medians[id] = median(group.samples.map((s) => s[id]));
  }
  const cache = f.summary.body_reconciliation[names[1]][rep];
  cache.covered_body_delta_bytes += delta; cache.prepared_stage_reply_delta_bytes += delta;
  refreshComparisons(f.summary);
}

test('small numeric MessagePack server_ns changes remain charged; stage-level bound enforced', () => {
  const f = fixture(); changeServerNsEnvelope(f, 2);
  assert.equal(evaluate(f)[0].rows.length, 2);
  const row = candidate(f), rep = row.execution.replicate, cached = f.summary.body_reconciliation[names[1]][rep];
  assert.equal(cached.prepared_stage_reply_delta_bytes, 2);
  assert.equal(cached.bootstrap_and_metadata_residual_bytes, 0);
  Object.values(reportOf(row).topology_accounting.stages.runs[0].body_bytes_by_stage_and_edge)[0]['inference->client']--;
  assert.throws(() => evaluate(f));
  const excessive = fixture();
  changeServerNsEnvelope(excessive, excessive.summary.body_reconciliation[names[1]][2].prepared_stage_reply_metadata_compatibility_bound_bytes + 1);
  assert.throws(() => evaluate(excessive), /server_ns bound/);
});
