import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { cohortRows, buildScorecard, executionPlacement, providerPeak, metrics, clientAndTotalResources, baselineKind, reportMetrics } from '../scripts/research-scorecard.mjs';
import { leaders, improvement, normalizedAxes, measuredSegments, defaultMetrics } from '../lib/research-rankings.mjs';
import { buildResults, qualityResults } from '../scripts/research-results.mjs';

const root = fileURLToPath(new URL('../../', import.meta.url));
const cohort = JSON.parse(fs.readFileSync(`${root}docs/data/research/benchmark-cohorts.json`)).cohorts
  .find((item) => item.id === 'qwen25-prepared-150-8');
const report = JSON.parse(fs.readFileSync(root + cohort.report));

test('published metrics conserve output denominators and unknown measurement scope', () => {
  const card = buildScorecard();
  const measured = card.cohorts.find((item) => item.id === cohort.id);
  const rows = measured.rows;
  assert.equal(rows.length, 3);
  assert.equal(measured.identity.outputs, 8);
  assert.equal(measured.identity.inputs, 150);
  for (const row of rows) {
    assert.equal(row.metrics.clientPeak, null);
    assert.equal(row.metrics.wire, null);
    assert.equal(row.metrics.quality, null);
    assert.equal(row.metrics.energy, null);
    assert.equal(row.fileSha256.length, 64);
    const source = report.candidates.find((item) => item.name === row.id).report;
    assert.equal(row.metrics.decodeTps, 7 / source.runs[0].durations.generation_seconds);
    assert.equal(row.metrics.covered, source.communication_per_token.summary.setup_inclusive_mb_per_output_token);
  }
  assert.deepEqual(leaders(rows, metrics.find((m) => m.id === 'full')), ['prepared-baseline']);
  assert.deepEqual(leaders(rows, metrics.find((m) => m.id === 'covered')), ['prepared-public-prefix']);
  assert.deepEqual(leaders(rows.filter((row) => !row.publicPrefixTokens), metrics[0]), ['prepared-exact-stack']);
});

test('incompatible sources, numerics, workloads, output identity and lifecycle cannot rank', () => {
  const mutations = [
    (c) => { c.report.configuration.prompt_digest = 'different invocation'; },
    (c) => { c.report.configuration.source_lock_digest = 'different source'; },
    (c) => { c.pipeline.components.quantization.params.activation_bits = 4; },
    (c) => { c.pipeline.components.verification = { component: 'pllm/freivalds-verify/v1', params: { target_failure_bits: 40 } }; },
    (c) => { c.report.runs[0].tokens.input_tokens++; },
    (c) => { c.report.runs[0].tokens.output_tokens++; },
    (c) => { c.report.runs[0].generation.output_text_digest = 'different output'; },
    (c) => { c.report.runs[0].model_fingerprint = 'different body'; },
    (c) => { c.report.runs[0].model_id = 'different model'; },
    (c) => { c.report.runs[0].warm = true; },
    (c) => { c.report.configuration.provider_backend = 'docker'; },
    (c) => { c.report.configuration.wan_emulation_digest = 'different network'; },
    (c) => { c.report.configuration.client_process_isolated = true; },
    (c) => { c.report.checks.passed = false; },
    (c) => { c.report.runs[0].status = 'failed'; },
    (c) => { c.report.runs[0].tokens.authoritative = false; },
    (c) => { c.report.warmup_runs.push(c.report.runs[0]); },
  ];
  for (const change of mutations) {
    const copy = structuredClone(report);
    change(copy.candidates[1]);
    assert.throws(() => cohortRows(cohort, copy));
  }
  const unknown = structuredClone(cohort);
  delete unknown.candidates['prepared-exact-stack'];
  assert.throws(() => cohortRows(unknown, report), /Python configuration/);
  assert.throws(() => cohortRows(cohort, report, (file) =>
    file === 'benchmarks/research/common.py' ? 'changed' : fs.readFileSync(root + file)), /source changed/);
  assert.throws(() => cohortRows(cohort, report, (file) =>
    file === 'benchmarks/research/sota.py' ? 'changed' : fs.readFileSync(root + file)), /configuration changed/);
});

test('zero, unknown, ties and metric direction retain their meaning', () => {
  const rows = [
    { id: 'unknown', metrics: { x: null } },
    { id: 'zero', metrics: { x: 0 } },
    { id: 'one', metrics: { x: 1 } },
    { id: 'tie', metrics: { x: 1 } },
  ];
  assert.deepEqual(leaders(rows, { id: 'x', direction: 'min' }), ['zero']);
  assert.deepEqual(leaders(rows, { id: 'x', direction: 'max' }), ['one', 'tie']);
  assert.deepEqual(leaders(rows.slice(0, 1), { id: 'x', direction: 'min' }), []);
  assert.equal(improvement(0, 2, 'min'), 100);
  assert.equal(improvement(null, 2, 'min'), null);
  assert.equal(improvement(2, 0, 'min'), null);
  assert.equal(improvement(3, 2, 'max'), 50);
});

test('SDK finalists retain measured placement, filters and original observations', () => {
  const card = buildScorecard();
  const finalists = card.cohorts.find((item) => item.id === 'qwen25-sdk-finalists-150-8');
  const throughput = metrics.find((item) => item.id === 'requestTps');
  assert.equal(finalists.rows.length, 16);
  assert.deepEqual(leaders(finalists.rows, throughput), ['searched-client-ee35d158335a']);
  assert.equal(finalists.rows.find((row) => row.placement.topology === 'Client-only').placement.clientMacPercent, 100);
  const remote = finalists.rows.filter((row) => !row.placement.clientBody);
  assert.equal(remote.length, 12);
  assert.deepEqual(leaders(remote, throughput), ['searched-public_prefix-a4e0c08b4380']);
  assert.deepEqual(leaders(remote.filter((row) => !row.publicPrefixTokens), throughput), ['searched-prepared-755220a8dcd9']);
  const raw = JSON.parse(fs.readFileSync(root + finalists.report));
  const registered = JSON.parse(fs.readFileSync(`${root}docs/data/research/benchmark-cohorts.json`)).cohorts.find((item) => item.id === finalists.id);
  const mutated = structuredClone(raw);
  mutated.candidates[0].report.runs[0].durations.full_seconds++;
  assert.throws(() => cohortRows(registered, mutated), /original measurements/);
  const rejected = structuredClone(raw);
  rejected.checks.passed = false;
  assert.throws(() => cohortRows(registered, rejected), /policy/);
});

test('parallel coordinates invert cost, preserve ties and break lines at unknowns', () => {
  const rows = [{ id: 'a', metrics: { speed: 2, cost: 0, tied: 3, unknown: null } },
    { id: 'b', metrics: { speed: 1, cost: 5, tied: 3, unknown: null } },
    { id: 'c', metrics: { speed: 1.5, cost: null, tied: 3, unknown: null } }];
  const axes = normalizedAxes(rows, [{ id: 'speed', direction: 'max' }, { id: 'cost', direction: 'min' },
    { id: 'tied', direction: 'min' }, { id: 'unknown', direction: 'min' }]);
  assert.deepEqual(axes[0].scores, { a: 100, b: 0, c: 50 });
  assert.deepEqual(axes[1].scores, { a: 100, b: 0, c: null });
  assert.deepEqual(axes[2].scores, { a: 100, b: 100, c: 100 });
  assert.deepEqual(axes[3].scores, { a: null, b: null, c: null });
  assert.deepEqual(measuredSegments([20, 0, null, 100]), [[{ index: 0, value: 20 }, { index: 1, value: 0 }], [{ index: 3, value: 100 }]]);
  assert.deepEqual(defaultMetrics, ['requestTps', 'networkPerToken', 'bodyOffload', 'clientCpu', 'clientPeak', 'clientNetwork', 'aggregateCpu', 'totalPeak', 'totalNetwork']);
});

test('declared subcohorts preserve their original salted observations exactly', () => {
  const selected = report.candidates.slice(0, 2).map((candidate) => candidate.name);
  const bytes = fs.readFileSync(root + cohort.report);
  const spec = { ...cohort, candidates: Object.fromEntries(selected.map((name) => [name, cohort.candidates[name]])) };
  const subcohort = structuredClone(report);
  subcohort.candidates = subcohort.candidates.slice(0, 2);
  subcohort.analysis_of = { file: cohort.report, sha256: buildScorecard().cohorts.find((item) => item.id === cohort.id).reportSha256,
    candidates: selected };
  assert.equal(cohortRows(spec, subcohort).rows.length, 2);
  const edited = structuredClone(subcohort);
  edited.candidates[0].report.runs[0].durations.full_seconds++;
  assert.throws(() => cohortRows(spec, edited), /original measurements/);
  for (const selection of [[selected[0], selected[0]], [selected[0], 'unknown'], [selected[0]]]) {
    assert.throws(() => cohortRows(spec, { ...subcohort, analysis_of: { ...subcohort.analysis_of, candidates: selection } }));
  }
  assert.throws(() => cohortRows(spec, subcohort, (file) => file === cohort.report
    ? Buffer.concat([bytes, Buffer.from(' ')]) : fs.readFileSync(root + file)), /original measurements/);
});

test('paper scores keep verifier contracts and calibrated quality evidence separate', () => {
  const card = buildScorecard();
  const verified = card.cohorts.find((item) => item.id === 'qwen25-slalom-50-8');
  assert.equal(verified.rows.length, 2);
  assert.ok(verified.identity.verification);
  assert.ok(verified.rows.every((row) => row.papers.some((paper) => paper.id === 'slalom' && paper.kind === 'adaptation')));
  const equalized = card.cohorts.find((item) => item.id === 'qwen25-smoothquant-50-8');
  const plain = card.cohorts.find((item) => item.id === 'qwen25-smoothquant-plain-50-8');
  assert.equal(equalized.rows.length, 2);
  assert.equal(equalized.identity.prompt, plain.identity.prompt);
  assert.equal(equalized.identity.output, plain.identity.output);
  assert.notEqual(equalized.identity.body, plain.identity.body);
  assert.ok(equalized.rows.every((row) => row.metrics.artifactBytes === 0.729527 &&
    row.papers.some((paper) => paper.id === 'smoothquant' && paper.kind === 'adaptation')));
  const { data, text } = buildResults();
  assert.equal(data.quality.length, 1);
  assert.deepEqual(data.quality[0].rows.map((row) => row.top1_agreement), [11 / 12, 10 / 12]);
  assert.equal(data.quality[0].calibration.profile_bytes, 729527);
  assert.match(text, /Paper numeric and offline evidence/);
  assert.ok(card.cohorts.flatMap((cohort) => cohort.rows).every((row) => row.metrics.quality === null));
});

test('equalized cost evidence requires its locked public calibration and priced artifact', () => {
  const spec = JSON.parse(fs.readFileSync(`${root}docs/data/research/benchmark-cohorts.json`)).cohorts
    .find((item) => item.id === 'qwen25-smoothquant-50-8');
  const document = JSON.parse(fs.readFileSync(root + spec.report));
  assert.throws(() => cohortRows({ ...spec, calibration_report: undefined }, document), /calibration artifact/);
  const changed = structuredClone(spec);
  changed.candidates['paper-qwen-smoothquant'].artifact_bytes = 0;
  assert.throws(() => cohortRows(changed, document), /artifact cost/);
  assert.throws(() => cohortRows(spec, document, (file) => file === spec.calibration_report
    ? Buffer.concat([fs.readFileSync(root + file), Buffer.from(' ')]) : fs.readFileSync(root + file)), /source changed/);
});

test('numeric evidence binds factories, source and calibration without hiding regressions', () => {
  const spec = JSON.parse(fs.readFileSync(`${root}docs/data/research/benchmark-cohorts.json`)).quality_cohorts[0];
  const original = JSON.parse(fs.readFileSync(root + spec.report));
  assert.throws(() => qualityResults(spec, (file) => file === spec.candidates['paper-qwen-smoothquant'].file
    ? 'changed factory' : fs.readFileSync(root + file)), /configuration/);
  for (const change of [
    (d) => { d.candidates[1].public_equalization_profile_digest = 'other profile'; },
    (d) => { d.model.source_lock_digest = 'other checkpoint'; },
    (d) => { d.candidates[1].configuration_digest = 'other configuration'; },
    (d) => { d.candidates[1].sample_count++; },
    (d) => { d.candidates[1].top1_agreement = 1.1; },
  ]) {
    const altered = structuredClone(original);
    change(altered);
    const raw = JSON.stringify(altered);
    const bound = { ...spec, sha256: createHash('sha256').update(raw).digest('hex') };
    assert.throws(() => qualityResults(bound, (file) => file === spec.report ? raw : fs.readFileSync(root + file)));
  }
});

test('offload counts plan work once without rewarding provider overhead or rescaling zero', () => {
  const source = structuredClone(report.candidates[0].report);
  source.client_body_placement = { schema: 'pllm.client_body_placement.v1',
    declared_body_linear_macs_per_row_client: 70, declared_body_linear_macs_per_row_remote: 30 };
  assert.equal(reportMetrics(source).bodyOffload, 30);
  source.process_cpu_accounting.aggregate_cold_first_response_cpu_seconds *= 100;
  assert.equal(reportMetrics(source).bodyOffload, 30);
  source.configuration.roles = ['client', 'worker_a', 'worker_b'];
  assert.equal(reportMetrics(source).bodyOffload, 30);
  source.client_body_placement.declared_body_linear_macs_per_row_client = 0;
  assert.equal(reportMetrics(source).bodyOffload, 100);
  source.client_body_placement.declared_body_linear_macs_per_row_remote = 0;
  assert.equal(reportMetrics(source).bodyOffload, null);
  delete source.client_body_placement;
  assert.equal(reportMetrics(source).bodyOffload, null);
  source.configuration.roles = ['client'];
  assert.equal(reportMetrics(source).bodyOffload, 0);
  const metric = metrics.find((item) => item.id === 'bodyOffload');
  const rows = [{ id: 'local', metrics: { bodyOffload: 0 } },
    { id: 'partial', metrics: { bodyOffload: 30 } }, { id: 'missing', metrics: { bodyOffload: null } }];
  assert.deepEqual(normalizedAxes(rows, [metric])[0].scores, { local: 0, partial: 30, missing: null });
  assert.deepEqual(normalizedAxes(rows.slice(0, 1), [metric])[0].scores, { local: 0 });
  rows[1].metrics.bodyOffload = 101;
  assert.throws(() => normalizedAxes(rows, [metric]), /score bounds/);
});

test('native scores distinguish isolated client, same-window total and conserved traffic', () => {
  const report = { configuration: { client_process_isolated: true, roles: ['client', 'inference', 'preparation'], wan_emulation_digest: 'caps' },
    runs: [{ tokens: { output_tokens: 2 } }], client_process_memory: { after: { lifetime_peak_rss_bytes: 100e6 } },
    native_process_memory: { schema: 'pllm.native_process_memory.v1', complete: true, samples: 10, sampled_total_peak_rss_bytes: 240e6 },
    wan_readiness: { emulation: { native_stream_rate_snapshots_checked: true } },
    native_network_accounting: { samples: { runs: [{ after: { wan_emulation: {
      schema: 'pllm.native_wan.v1', enforced: true, conditions_digest: 'caps',
      directed_stream_bytes: { 'client->inference': 2e6, 'inference->client': 3e6, 'preparation->inference': 7e6 },
      total_stream_bytes: 12e6, client_stream_bytes: 5e6,
    } } }] } } };
  assert.deepEqual(clientAndTotalResources(report), { clientPeak: 100, totalPeak: 240, clientNetwork: 5, totalNetwork: 12, networkPerToken: 6 });
  report.native_process_memory.sampled_total_peak_rss_bytes = null;
  assert.equal(clientAndTotalResources(report).totalPeak, null);
  report.configuration.client_process_isolated = false;
  assert.equal(clientAndTotalResources(report).clientPeak, null);
  assert.equal(clientAndTotalResources(report).totalPeak, null);
  report.native_network_accounting.samples.runs[0].after.wan_emulation.total_stream_bytes++;
  assert.throws(() => clientAndTotalResources(report), /conservation/);
});

test('optimized offset finalists cannot be relabelled as the naive baseline', () => {
  const pipeline = { components: { linear: { component: 'pllm/two-online-offset-linear/v1', params: {} } } };
  const placement = { topology: 'Two workers', kernel: 'CPU' };
  assert.equal(baselineKind({ reference: 'offset' }, pipeline, placement, false), 'offset');
  pipeline.components.linear.params.input_encoding = 'seeded';
  assert.throws(() => baselineKind({ reference: 'offset' }, pipeline, placement, false), /unoptimized/);
});

test('native 100/40 frontier measures client and total for every matched finalist', () => {
  const card = buildScorecard();
  const frontier = card.cohorts[0];
  assert.equal(frontier.id, 'qwen25-native-wan-frontier-150-8');
  assert.equal(frontier.rows.length, 17);
  assert.equal(frontier.identity.wanBackend, 'native-shared-tcp-pacer');
  assert.equal(frontier.identity.clientIsolation, true);
  assert.deepEqual(frontier.rows.filter((row) => row.reference).map((row) => row.reference).sort(),
    ['client', 'offset', 'prepared']);
  const raw = JSON.parse(fs.readFileSync(root + frontier.report));
  assert.equal(frontier.rows.find((row) => row.reference === 'client').metrics.bodyOffload, 0);
  assert.equal(frontier.rows.find((row) => row.reference === 'offset').metrics.bodyOffload, 100);
  assert.equal(frontier.rows.find((row) => row.reference === 'prepared').metrics.bodyOffload, 100);
  assert.ok(frontier.rows.filter((row) => row.placement.ownership === 'Selected body linear stages at client')
    .every((row) => Math.abs(row.metrics.bodyOffload - 12.307692307692308) < 1e-10));
  for (const row of frontier.rows) {
    for (const metric of defaultMetrics) assert.ok(Number.isFinite(row.metrics[metric]), `${row.id}: ${metric}`);
    assert.ok(row.metrics.clientCpu <= row.metrics.aggregateCpu);
    assert.ok(row.metrics.clientNetwork <= row.metrics.totalNetwork);
    if (row.metrics.covered !== null) assert.ok(row.metrics.networkPerToken >= row.metrics.covered);
    const report = raw.candidates.find((item) => item.name === row.id).report;
    const sample = report.native_network_accounting.samples.runs[0].after.wan_emulation;
    for (const party of Object.values(sample.parties)) {
      assert.equal(party.download.bytes_per_second, 12_500_000);
      assert.equal(party.upload.bytes_per_second, 5_000_000);
    }
  }
});

test('research data retains every exploratory attempt without merging cohort rankings', () => {
  const { data, text } = buildResults();
  assert.equal(data.searches.reduce((sum, search) => sum + search.attempted, 0), 108);
  assert.equal(data.searches.reduce((sum, search) => sum + search.trials.length, 0), 72);
  assert.equal(data.searches.reduce((sum, search) => sum + search.rejections.length, 0), 36);
  assert.match(text, /Unmeasured/);
  assert.match(text, /cannot be ranked together/);
  assert.match(text, /trial_ee35d158335adf1b.py/);
  for (const search of data.searches) for (const trial of search.trials) {
    assert.ok(trial.observations.clientCpu > 0);
    assert.ok(trial.observations.aggregateCpu >= trial.observations.clientCpu);
    assert.equal(trial.observations.clientPeak, null);
    assert.equal(trial.observations.totalPeak, null);
    assert.equal(trial.observations.totalNetwork, null);
  }
});

test('artifact cost and public-prefix scope bind the measured Pipeline and publisher', () => {
  for (const [key, value] of [['artifact_bytes', 0], ['public_prefix_tokens', 0]]) {
    const copy = structuredClone(cohort);
    copy.candidates['prepared-public-prefix'][key] = value;
    assert.throws(() => cohortRows(copy, report), /artifact cost or prefix scope/);
  }
  for (const change of [
    (p) => { p.tokenizer.bytes++; },
    (p) => { p.capsule.digest = 'different capsule'; },
    (p) => { p.body_fingerprint = 'different body'; },
    (p) => { p.output_cap++; },
    (p) => { p.public_prefix_tokens++; },
  ]) {
    const publisher = JSON.parse(fs.readFileSync(root + cohort.publisher_report));
    change(publisher);
    assert.throws(() => cohortRows(cohort, report, (file) => file === cohort.publisher_report
      ? JSON.stringify(publisher) : fs.readFileSync(root + file)), /artifact/i);
  }
});

test('placement distinguishes local work, provider graphs and CPU/Metal', () => {
  const pipeline = structuredClone(report.candidates[0].pipeline);
  const body = { schema: 'pllm.client_body_placement.v1',
    declared_body_linear_macs_per_row_client: 70,
    declared_body_linear_macs_per_row_remote: 30 };
  pipeline.components.kernels = { component: 'pllm/apple-metal-int8/v1', params: { min_rows: 8 } };
  pipeline.components.placement = { component: 'pllm/client-owned-linear-roles/v1' };
  const mixed = executionPlacement(pipeline, ['client', 'inference', 'preparation'], body);
  assert.equal(mixed.topology, 'Prepared');
  assert.equal(mixed.kernel, 'CPU + Metal');
  assert.equal(mixed.clientBody, true);
  assert.equal(mixed.clientMacPercent, 70);
  delete pipeline.components.placement;
  assert.equal(executionPlacement(pipeline, ['client']).ownership, 'Full decoder at client');
  const workers = executionPlacement(pipeline, ['worker_b', 'client', 'worker_a']);
  assert.equal(workers.topology, 'Two workers');
  assert.equal(workers.clientBody, false);
  assert.equal(workers.clientMacPercent, null);
  assert.throws(() => executionPlacement(pipeline, ['unrecognised-role']), /role graph/);
  assert.throws(() => executionPlacement(pipeline, ['client'], {
    ...body, declared_body_linear_macs_per_row_client: -1,
  }), /body placement/);
});

test('provider memory requires every role sample and excludes cumulative client memory', () => {
  const processes = { client: { rss_peak_bytes: 900e6 }, worker_a: { rss_peak_bytes: 100e6 },
    worker_b: { rss_peak_bytes: 120e6 } };
  assert.equal(providerPeak(['client', 'worker_a', 'worker_b'], processes), 120);
  assert.equal(providerPeak(['client'], processes), 0);
  delete processes.worker_b;
  assert.equal(providerPeak(['client', 'worker_a', 'worker_b'], processes), null);
  processes.worker_a.rss_peak_bytes = -1;
  assert.throws(() => providerPeak(['client', 'worker_a'], processes), /provider peak RSS/);
});

test('chronology exposes real implementation coverage without promoting planned classes', () => {
  const registry = JSON.parse(fs.readFileSync(`${root}docs/data/research/implementations.json`));
  const index = fs.readFileSync(`${root}docs/content/docs/research/papers/index.mdx`, 'utf8');
  assert.ok(index.indexOf('<ResearchScorecard />') < index.indexOf('<PaperTimeline>'));
  assert.equal([...index.matchAll(/data-implemented="true"/g)].length, Object.keys(registry.papers).length);
  assert.equal([...index.matchAll(/data-implemented="false"/g)].length, 85 - Object.keys(registry.papers).length);
  assert.match(index, /No reimplementation performance score/);
  for (const item of Object.values(registry.papers)) {
    for (const file of [...item.code, item.configuration, item.evidence].filter(Boolean)) {
      assert.ok(fs.existsSync(root + file), file);
    }
  }
});
