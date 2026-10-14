import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { buildScorecard, executionPlacement, metrics, providerPeak } from '../scripts/research-scorecard.mjs';
import { adaptedCohorts } from '../scripts/research-scorecard-adapters.mjs';
import { componentCombination, memoryBudgetRows, scopedMetric, sotaTable } from '../lib/research-rankings.mjs';

const root = fileURLToPath(new URL('../../', import.meta.url));
const read = (file) => fs.readFileSync(root + file);
const hash = (value) => createHash('sha256').update(value).digest('hex');
const specs = JSON.parse(read('docs/data/research/benchmark-cohorts.json')).adapted_cohorts.filter((spec) => spec.adapter === 'private-boundary-v1');
const helpers = { metrics, executionPlacement, providerPeak };

test('both independent boundary cohorts expose exact SDK recipes with honest owner and timing scopes', () => {
  const card = buildScorecard();
  assert.equal(specs.length, 2);
  const views = specs.map((spec) => card.cohorts.find((cohort) => cohort.id === spec.id));
  assert.notEqual(views[0].identity.prompt, views[1].identity.prompt);
  assert.deepEqual(views.map((view) => view.identity.inputs), [48, 150]);
  for (const view of views) {
    assert.equal(view.evidenceTier, 'screen');
    assert.equal(view.rows.length, 2);
    assert.deepEqual(view.rows.map((row) => row.placement.topology), ['Prepared', 'Prepared + private boundary']);
    for (const row of view.rows) {
      assert.equal(row.sampleCount, 3);
      assert.ok(row.metrics.clientPeak > 0);
      assert.equal(row.clientMemory.maxPeakBytes, null);
      assert.ok(row.observedClientSubprocessPeakBytes > row.metrics.clientPeak * 1e6);
      for (const tier of card.clientMemoryTiers) assert.equal(memoryBudgetRows([row], tier.maxBytes).rows.length, 0);
      assert.equal(row.metrics.totalPeak, null);
      assert.equal(row.metrics.baseNetwork, null);
      assert.equal(row.metrics.tokenNetwork, null);
      assert.equal(row.metrics.wire, null);
      assert.equal(row.metrics.quality, null);
      assert.equal(row.replay.callable, 'recorded_contract');
      assert.ok(row.scopedTimings.completeCellSeconds > row.metrics.full);
    }
    const [control, coupled] = view.rows;
    assert.ok(coupled.metrics.totalNetwork < control.metrics.totalNetwork);
    assert.ok(coupled.metrics.decodeTps < control.metrics.decodeTps);
    assert.ok(coupled.metrics.aggregateCpu > control.metrics.aggregateCpu);
    assert.ok(componentCombination(coupled.pipeline).some((component) => component.name === 'PrivateLookupAndFullHead'));
    assert.equal(sotaTable(view, 'Prepared', metrics).rows.length, 1);
    assert.equal(sotaTable(view, 'Prepared + private boundary', metrics).rows.length, 1);
    assert.match(scopedMetric(metrics.find((metric) => metric.id === 'ttft'), view.measurement).label, /setup included/);
    assert.match(scopedMetric(metrics.find((metric) => metric.id === 'clientPeak'), view.measurement).label, /relay excluded/);
  }
});

test('boundary rows reject changed raw sources, summaries, recipes, drivers and tier promotion', () => {
  const spec = specs[0];
  for (const file of [spec.report, spec.summary, spec.control, spec.summary_source, spec.replay_file, spec.frozen_factory.file]) {
    assert.throws(() => adaptedCohorts(spec, (path) => path === file ? Buffer.from('{}') : read(path), helpers));
  }
  for (const change of [
    (s) => { s.groups.coupled.medians.all_link_tcp_stream_bytes = 1; },
    (s) => { s.groups.coupled.whole_client_peak_rss_bytes = 200000000; },
    (s) => { s.groups.coupled.whole_client_memory_tier_qualified = true; },
    (s) => { s.groups.coupled.experiment.pipeline.components.kernels.params.threads = 2; },
    (s) => { s.groups.coupled.sample_count = 1; },
    (s) => { s.cohort_commitment = '0'.repeat(64); },
  ]) {
    const summary = JSON.parse(read(spec.summary)); change(summary);
    const raw = Buffer.from(JSON.stringify(summary));
    assert.throws(() => adaptedCohorts({ ...spec, summary_sha256: hash(raw) },
      (path) => path === spec.summary ? raw : read(path), helpers));
  }
  assert.throws(() => adaptedCohorts({ ...spec, evidenceTier: 'confirmation' }, read, helpers));
  const pipeline = JSON.parse(read(spec.summary)).groups.coupled.experiment.pipeline;
  delete pipeline.components.boundary;
  assert.throws(() => executionPlacement(pipeline, ['client', 'preparation', 'inference', 'boundary_a', 'boundary_b']));
});
