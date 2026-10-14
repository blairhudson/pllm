import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { buildScorecard, executionPlacement, metrics, providerPeak } from '../scripts/research-scorecard.mjs';
import { adaptedCohorts } from '../scripts/research-scorecard-adapters.mjs';
import { leaders, memoryBudgetRows, sotaTable } from '../lib/research-rankings.mjs';

const root = fileURLToPath(new URL('../../', import.meta.url));
const read = (file) => fs.readFileSync(root + file);
const hash = (value) => createHash('sha256').update(value).digest('hex');
const spec = JSON.parse(read('docs/data/research/benchmark-cohorts.json')).adapted_cohorts
  .find((item) => item.adapter === 'private-boundary-tiers-v1');
const helpers = { metrics, executionPlacement, providerPeak };

test('fresh complete-worker evidence qualifies budgets without promoting historical child peaks', () => {
  const card = buildScorecard(), view = card.cohorts.find((item) => item.id === spec.id);
  const raw = JSON.parse(read(spec.report));
  assert.equal(view.evidenceTier, 'confirmation');
  assert.equal(view.rows.length, 8);
  for (const row of view.rows) {
    const samples = raw.results.filter((item) => item.execution.case.name === row.id);
    const peaks = samples.map((item) => item.metrics.find((metric) => metric.id === 'process.peak_rss').value);
    assert.equal(row.clientMemory.maxPeakBytes, Math.max(...peaks));
    assert.equal(row.sampleCount, 3);
    assert.equal(row.clientMemory.scope, 'process_birth_through_target');
    assert.equal(row.recordedCase.target.parameters.state_audit, true);
    assert.equal(row.replay.callable, 'pllm.metrics.BenchmarkCase.from_spec');
    assert.ok(row.scopedTimings.completeTargetSeconds > row.metrics.full);
    assert.equal(row.metrics.baseNetwork, null);
    assert.equal(row.metrics.tokenNetwork, null);
    assert.equal(row.metrics.covered, null);
    assert.equal(row.metrics.wire, null);
  }
  const candidate = view.rows.find((row) => row.id === 'coupled-boundary');
  assert.equal(candidate.clientMemory.maxPeakBytes, 220430336);
  assert.equal(candidate.metrics.providerPeak, null); // Boundary peaks missing from this particular provider ledger.
  assert.ok(view.rows.find((row) => row.id === 'maintained').metrics.providerPeak > 1000);
  for (const maximum of [256 * 2 ** 20, 2 * 2 ** 30]) {
    const eligible = memoryBudgetRows(view.rows, maximum).rows;
    assert.equal(eligible.length, 8);
    for (const id of ['totalNetwork', 'full', 'clientCpu']) {
      assert.deepEqual(leaders(eligible, metrics.find((metric) => metric.id === id)), ['coupled-boundary']);
    }
    assert.deepEqual(leaders(eligible, metrics.find((metric) => metric.id === 'decodeTps')), ['tiling-digit8-placement']);
    assert.deepEqual(leaders(eligible, metrics.find((metric) => metric.id === 'clientPeak')), ['tiling-digit8']);
    assert.equal(sotaTable(view, 'Prepared', metrics, maximum).rows.length, 7);
    assert.equal(sotaTable(view, 'Prepared + private boundary', metrics, maximum).rows.length, 1);
  }
  for (const old of card.cohorts.filter((item) => item.adapter === 'private-boundary-v1')) {
    assert.equal(memoryBudgetRows(old.rows, 2 * 2 ** 30).rows.length, 0);
  }
});

test('tier reader rejects changed sources, incomplete replicates, state and measurement windows', () => {
  for (const file of [spec.report, spec.plan, spec.summary, spec.replay_file, spec.frozen_factory.file]) {
    assert.throws(() => adaptedCohorts(spec, (path) => path === file ? Buffer.from('{}') : read(path), helpers));
  }
  for (const change of [
    (document) => { document.results.pop(); },
    (document) => { document.results[8].execution.replicate = 0; },
    (document) => { document.results[0].execution.build.digest = '0'.repeat(64); },
    (document) => { document.results[0].execution.details.response_state_audit.bindings[0].selections[0].logits_and_active_kv = '0'.repeat(64); },
    (document) => { document.results[0].execution.details.response_state_audit.client_memory_owners = ['sdk']; },
    (document) => { document.results[0].metrics.find((metric) => metric.id === 'process.peak_rss').window = 'client_child_only'; },
    (document) => { document.results[0].metrics.find((metric) => metric.id === 'response.ttft').window = 'including_setup'; },
  ]) {
    const document = JSON.parse(read(spec.report)); change(document);
    const raw = Buffer.from(JSON.stringify(document));
    assert.throws(() => adaptedCohorts({ ...spec, sha256: hash(raw) },
      (path) => path === spec.report ? raw : read(path), helpers));
  }
});

test('cached summaries cannot hide an outlier, change tier eligibility or manufacture a metric win', () => {
  for (const change of [
    (summary) => { summary.groups['coupled-boundary'].maximum_client_peak_bytes = summary.groups['coupled-boundary'].medians['process.peak_rss']; },
    (summary) => { summary.groups['coupled-boundary'].medians['network.all_link_stream_bytes'] = 1; },
    (summary) => { delete summary.groups['coupled-boundary'].medians['process.peak_rss']; },
    (summary) => { summary.groups['coupled-boundary'].observed_tiers.low = false; },
    (summary) => { summary.checked_values--; },
  ]) {
    const summary = JSON.parse(read(spec.summary)); change(summary);
    const raw = Buffer.from(JSON.stringify(summary));
    assert.throws(() => adaptedCohorts({ ...spec, summary_sha256: hash(raw) },
      (path) => path === spec.summary ? raw : read(path), helpers));
  }
});
