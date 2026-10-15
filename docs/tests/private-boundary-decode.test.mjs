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
  .find((entry) => entry.adapter === 'private-boundary-decode-v1');
const helpers = { metrics, executionPlacement, providerPeak };
const boundaryRow = (document) => document.results.find((entry) =>
  entry.execution.case.name === 'coupled-parallel-uniform_residues');

test('matched head choices keep observed Low ownership and topology-local screen leaders', () => {
  const card = buildScorecard(), view = card.cohorts.find((entry) => entry.id === spec.id);
  const raw = JSON.parse(read(spec.report));
  assert.equal(view.evidenceTier, 'screen');
  assert.equal(view.rows.length, 5);
  assert.equal(view.identity.inputs, 150);
  assert.equal(view.identity.outputs, 8);
  for (const row of view.rows) {
    const samples = raw.results.filter((entry) => entry.execution.case.name === row.id);
    const peaks = samples.map((entry) => entry.metrics.find((metric) => metric.id === 'process.peak_rss').value);
    assert.equal(row.clientMemory.maxPeakBytes, Math.max(...peaks));
    assert.equal(row.clientMemory.scope, 'process_birth_through_target');
    assert.equal(row.sampleCount, 3);
    assert.equal(row.recordedCase.target.parameters.state_audit, true);
    assert.equal(row.replay.callable, 'pllm.metrics.BenchmarkCase.from_spec');
    assert.equal(row.metrics.baseNetwork, null);
    assert.equal(row.metrics.tokenNetwork, null);
    assert.equal(row.metrics.wire, null);
  }
  const combined = view.rows.find((row) => row.id === 'coupled-parallel-uniform_residues');
  const serial = view.rows.find((row) => row.id === 'coupled-serial-u32');
  assert.equal(combined.clientMemory.maxPeakBytes, 239779840);
  assert.ok(combined.metrics.decodeTps / serial.metrics.decodeTps > 1.52);
  for (const maximum of [256 * 2 ** 20, 2 * 2 ** 30]) {
    assert.equal(memoryBudgetRows(view.rows, maximum).rows.length, 5);
    const boundary = sotaTable(view, 'Prepared + private boundary', metrics, maximum);
    assert.equal(boundary.rows.length, 4);
    assert.deepEqual(boundary.leaders.decodeTps, [combined.id]);
    assert.equal(sotaTable(view, 'Prepared', metrics, maximum).rows.length, 1);
    assert.deepEqual(leaders(view.rows, metrics.find((metric) => metric.id === 'decodeTps')), ['maintained']);
  }
  assert.equal(card.cohorts.find((entry) => entry.adapter === 'private-boundary-tiers-v1').rows.length, 8);
});

test('decode reader rejects changed sources, raw bodies and jointly rewritten cached totals', () => {
  for (const file of [spec.report, spec.plan, spec.summary, spec.replay_file, spec.frozen_factory.file]) {
    assert.throws(() => adaptedCohorts(spec, (path) => path === file ? Buffer.from('{}') : read(path), helpers));
  }
  for (const change of [
    (document) => { document.results.pop(); },
    (document) => { boundaryRow(document).execution.details.report.runs[0].privacy.correction_push_bytes++; },
    (document) => {
      const row = boundaryRow(document);
      row.execution.details.report.topology_accounting.runs[0].all_link_serialized_body_bytes--;
      row.metrics.find((metric) => metric.id === 'network.covered_body_bytes').value--;
    },
    (document) => {
      delete boundaryRow(document).execution.details.report.topology_accounting.startup
        .traffic_attribution.audit_counters['role_link.boundary_a.setup_upload_bytes'];
    },
    (document) => { boundaryRow(document).execution.details.response_state_audit.client_memory_owners = ['sdk']; },
  ]) {
    const document = JSON.parse(read(spec.report)); change(document);
    const raw = Buffer.from(JSON.stringify(document));
    assert.throws(() => adaptedCohorts({ ...spec, sha256: hash(raw) },
      (path) => path === spec.report ? raw : read(path), helpers));
  }
});

test('decode summary cannot manufacture a throughput winner or hide a memory outlier', () => {
  for (const change of [
    (summary) => { summary.groups['coupled-parallel-uniform_residues'].medians['response.decode_tps'] = 100; },
    (summary) => { summary.groups['coupled-parallel-uniform_residues'].maximum_client_peak_bytes = 1; },
    (summary) => { summary.checked_values--; },
  ]) {
    const summary = JSON.parse(read(spec.summary)); change(summary);
    const raw = Buffer.from(JSON.stringify(summary));
    assert.throws(() => adaptedCohorts({ ...spec, summary_sha256: hash(raw) },
      (path) => path === spec.summary ? raw : read(path), helpers));
  }
});
