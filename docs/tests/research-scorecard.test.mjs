import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import { cohortRows, buildScorecard, executionPlacement, providerPeak, metrics } from '../scripts/research-scorecard.mjs';
import { leaders, improvement } from '../lib/research-rankings.mjs';

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
    (c) => { c.report.runs[0].tokens.input_tokens++; },
    (c) => { c.report.runs[0].tokens.output_tokens++; },
    (c) => { c.report.runs[0].generation.output_text_digest = 'different output'; },
    (c) => { c.report.runs[0].model_fingerprint = 'different body'; },
    (c) => { c.report.runs[0].model_id = 'different model'; },
    (c) => { c.report.runs[0].warm = true; },
    (c) => { c.report.configuration.provider_backend = 'docker'; },
    (c) => { c.report.configuration.wan_emulation_digest = 'different network'; },
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
  const registered = JSON.parse(fs.readFileSync(`${root}docs/data/research/benchmark-cohorts.json`)).cohorts[0];
  const mutated = structuredClone(raw);
  mutated.candidates[0].report.runs[0].durations.full_seconds++;
  assert.throws(() => cohortRows(registered, mutated), /original measurements/);
  const rejected = structuredClone(raw);
  rejected.checks.passed = false;
  assert.throws(() => cohortRows(registered, rejected), /policy/);
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
