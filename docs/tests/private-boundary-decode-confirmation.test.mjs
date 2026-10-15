/** In-memory protocol controls only: transformed archives are NOT measurements. */
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { executionPlacement, metrics, providerPeak } from '../scripts/research-scorecard.mjs';
import { adaptedCohorts } from '../scripts/research-scorecard-adapters.mjs';
import { memoryBudgetRows, leaders } from '../lib/research-rankings.mjs';

const root = fileURLToPath(new URL('../../', import.meta.url));
const read = (file) => fs.readFileSync(root + file);
const hash = (value) => createHash('sha256').update(value).digest('hex');
const names = ['maintained', 'coupled-serial-u32', 'coupled-parallel-uniform_residues'];
const priority = ['client.peak_rss', 'client.cold_cpu', 'preparation.cold_cpu', 'preparation.peak_rss',
  'response.ttft', 'response.decode_tps', 'network.all_link_stream_bytes', 'network.covered_body_bytes',
  'request.0.full', 'target.latency', 'process.peak_rss'];
const old = JSON.parse(read('docs/data/research/benchmark-cohorts.json')).adapted_cohorts
  .find((entry) => entry.adapter === 'private-boundary-decode-v1');
const archive = JSON.parse(read(old.report)), archivePlan = JSON.parse(read(old.plan));
const factory = 'benchmarks/research/private_boundary_decode_confirmation.py';
const driver = 'scripts/benchmark_private_boundary_decode_confirmation.py';
const helpers = { executionPlacement, metrics, providerPeak };
const canonical = (v, spaced = false, key = '') => {
  const comma = spaced ? ', ' : ',', colon = spaced ? ': ' : ':';
  if (Array.isArray(v)) return `[${v.map((x) => canonical(x, spaced)).join(comma)}]`;
  if (v && typeof v === 'object') return `{${Object.keys(v).sort().map((k) => `${JSON.stringify(k)}${colon}${canonical(v[k], spaced, k)}`).join(comma)}}`;
  if (Number.isInteger(v) && ['download_mbps', 'upload_mbps'].includes(key)) return `${v}.0`;
  return JSON.stringify(v);
};
const digest = (v) => hash(canonical(v));
const median = (v) => v.includes(null) ? null : [...v].sort((a, b) => a - b)[1];
const values = (r) => Object.fromEntries(r.metrics.filter((m) => priority.includes(m.id)).map((m) => [m.id, m.value]));
const fields = ['preparation_rows', 'preparation_prefill_rows_issued', 'preparation_decode_rows_issued',
  'preparation_padding_rows_issued', 'inference_stage_calls', 'prepared_stage_rows_reserved',
  'prepared_stage_rows_claimed', 'prepared_stage_rows_burned', 'prepared_stage_rows_discarded'];
const pick = (v, keys) => Object.fromEntries(keys.map((k) => [k, v[k]]));
const work = (r) => {
  const run = r.execution.details.report.runs[0], p = run.privacy, count = r.execution.details.response_state_audit.bindings[0].selections.length;
  return { output_tokens: run.tokens.output_tokens, sampled_selections: count, executed_decode_steps: count - 1,
    terminated_before_cap: run.tokens.output_tokens < run.max_output_tokens, ...pick(p, fields) };
};

function fixture(cap = 8, output = cap, burned = 0, discarded = 0) {
  const document = structuredClone(archive);
  document.suite.name = 'private-boundary-decode-confirmation-v1';
  document.suite.sampling.timeout_seconds = 3600;
  document.suite.cases = document.suite.cases.filter((c) => names.includes(c.name));
  for (const c of document.suite.cases) {
    c.target.parameters.scenario.max_output_tokens = cap;
    c.target.parameters.model.experiment.budget.max_new_tokens = cap;
  }
  document.results = document.results.filter((r) => names.includes(r.execution.case.name));
  const suiteHash = digest(document.suite), count = output + Number(output < cap);
  for (const row of document.results) {
    const e = row.execution;
    e.case = structuredClone(document.suite.cases.find((c) => c.name === e.case.name));
    e.sampling = structuredClone(document.suite.sampling);
    e.suite_digest = suiteHash; e.case_digest = digest(e.case);
    row.configuration_digest = hash('pllm.benchmark_target.v1\0' + canonical(e.case.target));
    const report = e.details.report, run = report.runs[0], p = run.privacy, binding = e.details.response_state_audit.bindings[0];
    report.configuration.max_output_tokens = cap; run.max_output_tokens = cap;
    run.tokens.output_tokens = output; run.tokens.total_tokens = 150 + output;
    run.generation.response_status = output < cap ? 'completed' : 'incomplete';
    binding.selections = Array.from({ length: count }, (_, i) => ({ ...structuredClone(binding.selections[Math.min(i, 7)]),
      position: 150 + i, cache_lengths: binding.selections[0].cache_lengths.map(() => 150 + i) }));
    e.oracle.checked_values = binding.selections.reduce((sum, s) => sum + s.checked_values, 0);
    p.generation_output_tokens = output; p.generation_requested_output_tokens = cap;
    p.generation_executed_decode_steps = count - 1;
    if (e.case.name !== 'maintained') { p.private_boundary_head_calls = count; p.private_boundary_lookup_rows = 150 + count - 1; }
    p.prepared_stage_rows_claimed -= burned + discarded; p.prepared_stage_rows_burned += burned;
    p.prepared_stage_rows_reserved -= discarded; p.prepared_stage_rows_discarded += discarded;
    report.prepared_material_accounting.counts.claimed -= burned + discarded;
    report.prepared_material_accounting.counts.burned += burned;
    report.prepared_material_accounting.counts.reserved -= discarded;
    report.prepared_material_accounting.counts.discarded += discarded;
    for (const m of row.metrics.filter((m) => ['request.0.decode_tps', 'response.decode_tps'].includes(m.id))) {
      m.denominator.value = output - 1;
      if (output === 1) Object.assign(m, { value: null, origin: 'not_available', sample_count: 0, unavailable_reason: 'measurement unavailable' });
      else { m.value = (output - 1) / run.durations.generation_seconds; run.durations.tokens_per_second = m.value; }
    }
  }
  const first = document.results[0].execution, report = first.details.report, binding = first.details.response_state_audit.bindings[0];
  const summary = { schema: 'pllm.private_boundary_decode_confirmation.v1', tiny: false, complete: true,
    cohort: document.cohort, clients: 9, replicates: 3, output_cap: cap,
    all_logits_active_kv_positions_selections_rng_and_outputs_match: true,
    identity: { source_lock_digest: report.configuration.source_lock_digest, prompt_digest: report.configuration.prompt_digest,
      wan_emulation_digest: report.configuration.wan_emulation_digest, sampling: report.configuration.sampling,
      body: binding.numeric_body_fingerprint, tokens: report.runs[0].tokens,
      output: report.runs[0].generation.output_text_digest, build: first.build },
    checked_values: document.results.reduce((sum, r) => sum + r.execution.oracle.checked_values, 0),
    state_commitment: hash(canonical(first.details.response_state_audit.bindings, true)),
    maximum_observed_swap_growth_bytes: 0, long_output_cap_reached: output === cap, groups: {},
    comparisons: { [names[2]]: {} }, body_reconciliation: { [names[2]]: [] } };
  const samples = Object.fromEntries(names.map((name) => [name, document.results.filter((r) => r.execution.case.name === name)
    .sort((a, b) => a.execution.replicate - b.execution.replicate)]));
  for (const name of names) {
    const rows = samples[name], observations = rows.map(values), peak = Math.max(...observations.map((v) => v['process.peak_rss']));
    summary.groups[name] = { samples: rows.map((r) => Object.fromEntries(r.metrics.map((m) => [m.id, m.value]))),
      medians: Object.fromEntries(priority.map((id) => [id, median(observations.map((v) => v[id]))])),
      maximum_client_peak_bytes: peak, observed_tiers: { low: peak <= 256 * 2 ** 20, medium: peak <= 2 * 2 ** 30, high: peak <= 16 * 2 ** 30 },
      executed_work: rows.map(work) };
  }
  for (const control of [names[1], names[0]]) {
    summary.comparisons[names[2]][control] = Object.fromEntries(priority.map((id) => {
      const a = summary.groups[control].medians[id], b = summary.groups[names[2]].medians[id], known = a !== null && b !== null;
      return [id, { control: a, candidate: b, delta: known ? b - a : null,
        improvement_fraction: known && a ? (id === 'response.decode_tps' ? b / a - 1 : 1 - b / a) : null,
        candidate_wins: known ? (id === 'response.decode_tps' ? b > a : b < a) : null,
        paired_deltas: samples[control].map((r, i) => known ? values(samples[names[2]][i])[id] - values(r)[id] : null) }];
    }));
  }
  for (let rep = 0; rep < 3; rep++) {
    const a = samples[names[1]][rep], b = samples[names[2]][rep], pa = a.execution.details.report.runs[0].privacy, pb = b.execution.details.report.runs[0].privacy;
    const total = values(b)['network.covered_body_bytes'] - values(a)['network.covered_body_bytes'];
    const head = pb.private_boundary_head_download_bytes - pa.private_boundary_head_download_bytes;
    const controls = (r) => r.execution.details.report.topology_accounting.startup.lifecycle_control_serialized_body_bytes +
      r.execution.details.report.topology_accounting.runs[0].lifecycle_control_serialized_body_bytes;
    const control = controls(b) - controls(a), metadata = pb.inference_download_bytes - pa.inference_download_bytes;
    summary.body_reconciliation[names[2]].push({ replicate: rep, covered_body_delta_bytes: total, head_reply_delta_bytes: head,
      lifecycle_control_delta_bytes: control, prepared_stage_reply_delta_bytes: metadata,
      prepared_stage_reply_metadata_compatibility_bound_bytes: 8 * (Math.ceil((output < cap ? pa.prepared_stage_rows_claimed : pa.prepared_stage_rows_issued) / 16) + pa.inference_stage_calls - 1),
      bootstrap_and_metadata_residual_bytes: total - head - control - metadata });
  }
  const plan = { schema: 'pllm.private_boundary_decode_confirmation_plan.v1', tiny: false, output_cap: cap,
    source_lock_digest: report.configuration.source_lock_digest, suite: structuredClone(document.suite),
    public_input_counts: { '150': 150 }, preflight: names.map((name) => ({ case: name, admission: { admitted: true } })),
    frozen_sources: { ...archivePlan.frozen_sources, [driver]: hash(read(driver)), [factory]: hash(read(factory)) } };
  const spec = { ...old, id: 'test-only-confirmation', adapter: 'private-boundary-decode-confirmation-v1', evidenceTier: 'confirmation',
    report: 'in-memory/report.json', plan: 'in-memory/plan.json', summary: 'in-memory/summary.json',
    replay_file: driver, replay_sha256: hash(read(driver)), frozen_factory: { file: factory, sha256: hash(read(factory)) } };
  return { document, plan, summary, spec };
}

function evaluate(f, corruptRead = null) {
  const spec = { ...f.spec }, files = new Map();
  for (const [field, value, digestField] of [['report', f.document, 'sha256'], ['plan', f.plan, 'plan_sha256'], ['summary', f.summary, 'summary_sha256']]) {
    const raw = Buffer.from(JSON.stringify(value)); files.set(spec[field], raw); spec[digestField] = hash(raw);
  }
  return adaptedCohorts(spec, (path) => corruptRead?.(path) ?? files.get(path) ?? read(path), helpers);
}
const boundary = (f) => f.document.results.findLast((r) => r.execution.case.name === names[2]);

const snapshot = 'docs/evidence/private-boundary-confirmation-initial-auditor.py';
const initialHash = '2720640b8928f044cc65b6f2b361b5d88882202a323de4fe34a28f2fd007772c';
function archivedSourceFixture() {
  const f = fixture();
  f.plan.frozen_sources[factory] = initialHash;
  f.spec.frozen_factory = { file: snapshot, sha256: initialHash };
  f.spec.source_snapshots = { [factory]: { file: snapshot, sha256: initialHash } };
  return f;
}

test('explicit original-byte snapshot preserves replay without reading hardened live factory', () => {
  const f = archivedSourceFixture(), before = structuredClone(f);
  assert.equal(hash(read(snapshot)), initialHash);
  const [view] = evaluate(f, (path) => {
    assert.notEqual(path, factory, 'must not substitute/read hardened live factory');
    return null;
  });
  assert.deepEqual(f, before);
  for (const row of view.rows) {
    assert.equal(row.file, snapshot);
    assert.equal(row.fileSha256, initialHash);
    assert.deepEqual(row.replay.sourceSnapshots, f.spec.source_snapshots);
    assert.equal(row.replay.requiresRegisteredSourceSnapshots, true);
    assert.equal(row.replay.originalDriver, driver);
  }
});

const snapshotMutations = {
  implicitArchive: (f) => { delete f.spec.source_snapshots; },
  missingEntry: (f) => { f.spec.source_snapshots = {}; },
  extraEntry: (f) => { f.spec.source_snapshots[driver] = { file: snapshot, sha256: initialHash }; },
  unlistedSource: (f) => { f.spec.source_snapshots = { 'benchmarks/research/private_boundary_decode.py': { file: snapshot, sha256: initialHash } }; },
  unlistedDestination: (f) => { f.spec.source_snapshots[factory].file = driver; },
  changedHash: (f) => { f.spec.source_snapshots[factory].sha256 = '0'.repeat(64); },
  extraEntryField: (f) => { f.spec.source_snapshots[factory].ignored = true; },
  unusedEntry: (f) => { delete f.plan.frozen_sources[factory]; },
  wrongPlanHash: (f) => { f.plan.frozen_sources[factory] = hash(read(factory)); },
  mismatchedFactory: (f) => { f.spec.frozen_factory = { file: factory, sha256: hash(read(factory)) }; },
};
for (const [name, mutate] of Object.entries(snapshotMutations)) test(`snapshot map rejects ${name}`, () => {
  const f = archivedSourceFixture(); mutate(f);
  assert.throws(() => evaluate(f));
});
test('snapshot bytes must match original plan, not merely registered mapping text', () => {
  assert.throws(() => evaluate(archivedSourceFixture(), (path) => path === snapshot ? Buffer.from('changed bytes') : null));
});

test('cached samples reconcile every SDK metric, not only scorecard priorities', () => {
  const f = fixture(), row = f.document.results.find((r) => r.execution.case.name === names[0] && r.execution.replicate === 0);
  assert.deepEqual(Object.keys(f.summary.groups[names[0]].samples[0]).sort(), row.metrics.map((m) => m.id).sort());
  const nonPriority = row.metrics.find((m) => !priority.includes(m.id));
  assert.ok(nonPriority);
  f.summary.groups[names[0]].samples[0][nonPriority.id] = nonPriority.value === null ? 1 : nonPriority.value + 1;
  assert.throws(() => evaluate(f), /cached samples changed/);
});

for (const cap of [8, 32, 64]) test(`three-arm ${cap} cap, recorded cases and original owner/TCP evidence`, () => {
  const f = fixture(cap), before = structuredClone(f), [view] = evaluate(f);
  assert.deepEqual(f, before);
  assert.equal(view.rows.length, 3); assert.equal(view.identity.outputs, cap); assert.equal(view.identity.cap, cap);
  assert.deepEqual(view.rows[2].experiment, view.rows[2].recordedCase.target.parameters.model.experiment);
  assert.equal(view.rows[2].sourceLock, f.summary.identity.source_lock_digest);
  assert.equal(view.rows[2].metrics.networkPerToken, f.summary.groups[names[2]].medians['network.all_link_stream_bytes'] / 1e6 / cap);
  for (const row of view.rows) {
    assert.equal(row.sampleCount, 3); assert.equal(row.replay.callable, 'pllm.metrics.BenchmarkCase.from_spec');
    assert.equal(row.metrics.baseNetwork, null); assert.equal(row.metrics.tokenNetwork, null); assert.equal(row.metrics.wire, null);
    assert.equal(row.clientMemory.maxPeakBytes, f.summary.groups[row.id].maximum_client_peak_bytes);
  }
});

for (const output of [1, 7, 31]) test(`natural EOS: ${output} emitted, final EOS selection separate, unused rows charged`, () => {
  const f = fixture(32, output, 20), [view] = evaluate(f), row = view.rows[2];
  assert.equal(view.identity.outputs, output); assert.equal(view.identity.cap, 32);
  assert.deepEqual(view.outputWorkload, { emittedTokens: output, outputCap: 32, sampledSelections: output + 1,
    executedDecodeSteps: output, terminatedBeforeCap: true });
  assert.equal(row.executedWork[0].prepared_stage_rows_burned, 20);
  assert.equal(row.metrics.networkPerToken, f.summary.groups[names[2]].medians['network.all_link_stream_bytes'] / 1e6 / output);
  assert.equal(row.metrics.requestTps, median(f.document.results.filter((r) => r.execution.case.name === names[2])
    .map((r) => output / r.execution.details.report.runs[0].durations.full_seconds)));
  if (output === 1) { assert.equal(row.metrics.decodeTps, null); assert.deepEqual(leaders(view.rows, metrics.find((m) => m.id === 'decodeTps')), []); }
});

test('natural EOS also conserves issued-but-unreserved discarded rows', () => {
  const f = fixture(64, 7, 20, 5), [view] = evaluate(f);
  assert.equal(view.rows[2].executedWork[0].prepared_stage_rows_discarded, 5);
  assert.equal(view.outputWorkload.outputCap, 64);
});

test('locally conserved EOS inventory still must match between boundary arms', () => {
  const f = fixture(32, 7, 20), r = boundary(f), report = r.execution.details.report;
  report.runs[0].privacy.prepared_stage_rows_claimed--;
  report.runs[0].privacy.prepared_stage_rows_burned++;
  report.prepared_material_accounting.counts.claimed--;
  report.prepared_material_accounting.counts.burned++;
  f.summary.groups[names[2]].executed_work[r.execution.replicate] = work(r);
  assert.throws(() => evaluate(f), /fixed boundary-arm work/);
});

test('server_ns stage-byte variation stays charged, independently reconciled, and bounded', () => {
  const f = fixture(32, 7, 20), row = boundary(f), report = row.execution.details.report, rep = row.execution.replicate;
  const p = report.runs[0].privacy, ledger = report.topology_accounting.runs[0];
  p.inference_download_bytes += 2;
  ledger.body_bytes_by_edge.find((e) => e.source === 'inference' && e.destination === 'client' && e.phase === 'online').serialized_body_bytes += 2;
  for (const field of ['all_link_serialized_body_bytes', 'client_serialized_body_bytes',
    'online_all_link_serialized_body_bytes', 'online_client_serialized_body_bytes']) ledger[field] += 2;
  const stage = report.topology_accounting.stages.runs[0];
  Object.values(stage.body_bytes_by_stage_and_edge)[0]['inference->client'] += 2;
  stage.measured_body_bytes_by_edge['inference->client'] += 2;
  stage.expected_stage_body_bytes_by_edge['inference->client'] += 2;
  const wan = report.native_network_accounting.samples.runs[0].after.wan_emulation;
  wan.directed_stream_bytes['inference->client'] += 2; wan.total_stream_bytes += 2; wan.client_stream_bytes += 2;
  wan.parties.inference.upload.admitted_bytes += 2; wan.parties.client.download.admitted_bytes += 2;
  for (const id of ['network.covered_body_bytes', 'network.all_link_stream_bytes']) {
    row.metrics.find((m) => m.id === id).value += 2;
    const group = f.summary.groups[names[2]];
    group.samples[rep][id] += 2; group.medians[id] = median(group.samples.map((v) => v[id]));
    for (const control of [names[1], names[0]]) {
      const a = f.summary.groups[control].medians[id], b = group.medians[id];
      f.summary.comparisons[names[2]][control][id] = { control: a, candidate: b, delta: b - a,
        improvement_fraction: 1 - b / a, candidate_wins: b < a,
        paired_deltas: group.samples.map((v, i) => v[id] - f.summary.groups[control].samples[i][id]) };
    }
  }
  const cached = f.summary.body_reconciliation[names[2]][rep];
  cached.covered_body_delta_bytes += 2; cached.prepared_stage_reply_delta_bytes += 2;
  assert.equal(evaluate(f)[0].rows.length, 3);
  Object.values(stage.body_bytes_by_stage_and_edge)[0]['inference->client'] -= 2;
  assert.throws(() => evaluate(f), /stage/); // No aggregate-only metadata excuse.
});

test('whole-worker maximum, not median or child peak, controls Low eligibility', () => {
  const f = fixture(), r = f.document.results.filter((r) => r.execution.case.name === names[2])
    .sort((a, b) => values(a)['process.peak_rss'] - values(b)['process.peak_rss']).at(-1),
    m = r.metrics.find((m) => m.id === 'process.peak_rss');
  m.value = 300 * 2 ** 20;
  const group = f.summary.groups[names[2]];
  group.samples[r.execution.replicate]['process.peak_rss'] = m.value;
  group.maximum_client_peak_bytes = m.value; group.observed_tiers.low = false;
  // Median and its derived comparisons do not change when only largest sample rises.
  f.summary.comparisons[names[2]][names[1]]['process.peak_rss'].paired_deltas[r.execution.replicate] =
    m.value - f.summary.groups[names[1]].samples[r.execution.replicate]['process.peak_rss'];
  f.summary.comparisons[names[2]].maintained['process.peak_rss'].paired_deltas[r.execution.replicate] =
    m.value - f.summary.groups.maintained.samples[r.execution.replicate]['process.peak_rss'];
  const [view] = evaluate(f);
  assert.equal(view.rows[2].clientMemory.maxPeakBytes, m.value);
  assert.equal(view.rows[2].observedTiers.low, false);
  assert.equal(memoryBudgetRows(view.rows, 256 * 2 ** 20).rows.length, 2);
});

test('exact report/plan/summary/driver/factory bytes cannot be substituted', () => {
  const f = fixture();
  for (const file of [f.spec.report, f.spec.plan, f.spec.summary, driver, factory]) {
    assert.throws(() => evaluate(f, (path) => path === file ? Buffer.from('{}') : null));
  }
});

const mutations = {
  missing: (f) => f.document.results.pop(),
  duplicate: (f) => { boundary(f).execution.replicate = 0; },
  cohort: (f) => { boundary(f).execution.cohort = '0'.repeat(64); },
  suiteDigest: (f) => { boundary(f).execution.suite_digest = '0'.repeat(64); },
  caseDigest: (f) => { boundary(f).execution.case_digest = '0'.repeat(64); },
  targetDigest: (f) => { boundary(f).configuration_digest = '0'.repeat(64); },
  build: (f) => { boundary(f).execution.build.digest = '0'.repeat(64); },
  buildAttributes: (f) => { boundary(f).execution.build.attributes.version = 'different'; },
  planSuite: (f) => { f.plan.suite.sampling.processes = 4; },
  frozenSource: (f) => { f.plan.frozen_sources[factory] = '0'.repeat(64); },
  sourceLock: (f) => { f.plan.source_lock_digest = '0'.repeat(64); },
  bounds: (f) => { boundary(f).execution.case.target.parameters.model.experiment.budget.max_new_tokens = 64; },
  state: (f) => { boundary(f).execution.details.response_state_audit.bindings[0].selections[0].logits_and_active_kv = '0'.repeat(64); },
  position: (f) => { boundary(f).execution.details.response_state_audit.bindings[0].selections[0].position++; },
  history: (f) => { boundary(f).execution.details.response_state_audit.bindings[0].selections[0].selected_token = '0'.repeat(64); },
  rng: (f) => { boundary(f).execution.details.response_state_audit.bindings[0].selections[0].rng_after = '0'.repeat(64); },
  sampler: (f) => { boundary(f).execution.details.report.configuration.sampling = { mode: 'greedy' }; },
  environment: (f) => { boundary(f).environment.digest = '0'.repeat(64); },
  numeric: (f) => { boundary(f).execution.comparison.numeric_digest = '0'.repeat(64); },
  measurement: (f) => { boundary(f).execution.comparison.measurement_digest = '0'.repeat(64); },
  oracle: (f) => { boundary(f).execution.oracle.checked_values--; },
  admission: (f) => { boundary(f).execution.details.topology_admission.admitted = false; },
  reportCheck: (f) => { delete boundary(f).execution.details.report.checks.no_plaintext_token_ids_sent; },
  swap: (f) => { boundary(f).execution.details.report.memory_guard.maximum_host_swap_growth_bytes = 1; },
  owner: (f) => { boundary(f).execution.details.response_state_audit.client_memory_owners = ['sdk']; },
  metricContract: (f) => { boundary(f).metrics.find((m) => m.id === 'response.ttft').window = 'including_setup'; },
  denominator: (f) => { boundary(f).metrics.find((m) => m.id === 'response.decode_tps').denominator.value = 32; },
  rawBody: (f) => { boundary(f).execution.details.report.runs[0].privacy.correction_push_bytes++; },
  cachedBody: (f) => { const r = boundary(f); r.execution.details.report.topology_accounting.runs[0].all_link_serialized_body_bytes--;
    r.metrics.find((m) => m.id === 'network.covered_body_bytes').value--; },
  rawStartup: (f) => { delete boundary(f).execution.details.report.topology_accounting.startup.traffic_attribution.audit_counters['role_link.boundary_a.setup_upload_bytes']; },
  tcp: (f) => { boundary(f).execution.details.report.native_network_accounting.samples.runs[0].after.wan_emulation.directed_stream_bytes['inference->client'] = 0; },
  wanRate: (f) => { boundary(f).execution.details.report.native_network_accounting.samples.runs[0].after.wan_emulation.parties.client.upload.bytes_per_second = 1e9; },
  stage: (f) => { const stage = boundary(f).execution.details.report.topology_accounting.stages.runs[0];
    Object.values(stage.body_bytes_by_stage_and_edge)[0]['inference->client']++; },
  material: (f) => { boundary(f).execution.details.report.prepared_material_accounting.counts.claimed--; },
  rows: (f) => { boundary(f).execution.details.report.runs[0].privacy.prepared_stage_rows_burned++; },
  cachedTiming: (f) => { f.summary.groups[names[2]].medians['response.decode_tps'] = 100; },
  rawTiming: (f) => { boundary(f).execution.details.report.runs[0].durations.generation_seconds /= 2; },
  sampleMirror: (f) => { boundary(f).samples[0].duration_seconds++; },
  requestDecodeAlias: (f) => { boundary(f).metrics.find((m) => m.id === 'request.0.decode_tps').value = 123456789; },
  bothDecodeAliases: (f) => { for (const r of f.document.results.filter((r) => r.execution.case.name === names[2])) {
    for (const m of r.metrics) if (['response.decode_tps', 'request.0.decode_tps'].includes(m.id)) m.value = 123456789;
  } },
  allTtftWindows: (f) => { for (const r of f.document.results) for (const m of r.metrics) {
    if (['response.ttft', 'request.0.ttft'].includes(m.id)) m.window = 'after_first_output';
  } },
  fabricatedWin: (f) => { f.summary.comparisons[names[2]].maintained['response.decode_tps'].candidate_wins = true; },
  codecSavings: (f) => { f.summary.body_reconciliation[names[2]][0].head_reply_delta_bytes--; },
  hiddenPeak: (f) => { f.summary.groups[names[2]].maximum_client_peak_bytes--; },
  tier: (f) => { f.summary.groups[names[2]].observed_tiers.low = false; },
  stateTotal: (f) => { f.summary.checked_values--; },
  work: (f) => { f.summary.groups[names[2]].executed_work[0].output_tokens = 32; },
};
for (const [name, mutate] of Object.entries(mutations)) test(`reject rehashed negative control: ${name}`, () => {
  const f = fixture(32, 7, 20); mutate(f); assert.throws(() => evaluate(f));
});

test('one-token null TPS and zero-output telemetry cannot become scorecard wins', () => {
  const f = fixture(32, 1, 20);
  f.summary.comparisons[names[2]].maintained['response.decode_tps'].candidate_wins = true;
  assert.throws(() => evaluate(f));
  const g = fixture(32, 1, 20);
  for (const r of g.document.results) r.execution.details.report.runs[0].tokens.output_tokens = 0;
  assert.throws(() => evaluate(g));
});

test('one-output request alias cannot claim measured zero instead of unavailable', () => {
  const f = fixture(32, 1, 20);
  for (const row of f.document.results) {
    Object.assign(row.metrics.find((m) => m.id === 'request.0.decode_tps'),
      { value: 0, origin: 'measured', sample_count: 1, unavailable_reason: null });
  }
  assert.throws(() => evaluate(f));
});
