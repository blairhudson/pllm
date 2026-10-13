import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { clientMemoryEvidence } from '../scripts/research-memory.mjs';
import { buildScorecard } from '../scripts/research-scorecard.mjs';
import { defaultResearchCohort, memoryBudgetRows, sotaTable } from '../lib/research-rankings.mjs';

test('default view finds newly registered Low evidence and preserves one cold cohort', () => {
  const card = JSON.parse(fs.readFileSync(new URL('../data/research/scorecard.json', import.meta.url)));
  const cap = card.clientMemoryTiers.find((tier) => tier.id === 'low').maxBytes;
  const cohorts = card.cohorts.filter((cohort) => [
    'docs/evidence/research-frontier-native-wan-qwen25.json',
    'docs/evidence/research-sota-combinations-qwen25.json',
  ].includes(cohort.report));
  const selected = defaultResearchCohort(cohorts, cap);
  assert.equal(selected.id, 'qwen25-sota-combinations-screen-0-cold');
  assert.equal(selected.evidenceTier, 'screen');
  assert.equal(sotaTable(selected, 'Prepared', [], cap).rows.length, 5);
  assert.equal(sotaTable(cohorts[0], 'Prepared', [], cap).rows.length, 1);
  const qwen3 = defaultResearchCohort(card.cohorts.filter((cohort) => [
    'qwen3-sdk-tokenizer-16-8', 'qwen3-client-memory-16-8',
  ].includes(cohort.id)), cap);
  assert.equal(qwen3.id, 'qwen3-client-memory-16-8');
  assert.equal(sotaTable(qwen3, 'Prepared', [], cap).rows.length, 1);
});

test('cohort defaults respect peak bounds, missing evidence, topology and empty tiers', () => {
  const cap = 256 * 2 ** 20;
  const row = (id, peak, topology = 'Prepared') => ({ id, metrics: { clientPeak: 1 },
    clientMemory: { maxPeakBytes: peak }, placement: { topology } });
  const cohort = (id, report, warm, rows) => ({ id, report, identity: { warm }, rows });
  const old = cohort('old', 'old.json', false, [row('old', cap)]);
  const cold = cohort('cold', 'new.json', false, [row('bounded', cap)]);
  const repeat = cohort('repeat', 'new.json', true, [row('bounded', cap)]);
  const over = cohort('over', 'over.json', false, [row('outlier', cap + 1), row('local', 1, 'Client-only')]);
  const unknown = cohort('unknown', 'unknown.json', false, [row('unknown', null)]);
  const cohorts = [old, repeat, cold, over, unknown];
  assert.equal(defaultResearchCohort(cohorts, cap), cold);
  assert.equal(defaultResearchCohort(cohorts), unknown);
  assert.equal(defaultResearchCohort([over], cap), over);
  assert.equal(sotaTable(defaultResearchCohort([over], cap), 'Prepared', [], cap).rows.length, 0);
  const local = cohort('local', 'local.json', false, [row('local', cap, 'Client-only')]);
  assert.equal(defaultResearchCohort([...cohorts, local], cap), local);
  assert.equal(defaultResearchCohort([], cap), undefined);
  assert.deepEqual(cohorts.map((value) => value.id), ['old', 'repeat', 'cold', 'over', 'unknown']);
});

test('RAM ceilings are inclusive, nested and independent of median or unknown RSS', () => {
  const cap = 256 * 2 ** 20;
  const rows = [
    { id: 'boundary', clientMemory: clientMemoryEvidence([cap - 2, cap], 'process_lifetime'),
      metrics: { clientPeak: cap / 1e6, speed: 2 }, placement: { topology: 'Prepared' } },
    { id: 'outlier', clientMemory: clientMemoryEvidence([200e6, 201e6, cap + 1], 'process_lifetime'),
      metrics: { clientPeak: 201, speed: 10 }, placement: { topology: 'Prepared' } },
    { id: 'unknown', metrics: { clientPeak: 100, speed: 30 }, placement: { topology: 'Prepared' } },
    { id: 'missing-sample', clientMemory: clientMemoryEvidence([1, null], 'process_lifetime'),
      metrics: { clientPeak: 1, speed: 40 }, placement: { topology: 'Prepared' } },
    { id: 'local', clientMemory: clientMemoryEvidence([cap], 'process_lifetime'),
      metrics: { clientPeak: cap / 1e6, speed: 100 }, placement: { topology: 'Client-only' } },
  ];
  const metrics = [{ id: 'speed', direction: 'max' }];
  const low = sotaTable({ rows }, 'Prepared', metrics, cap);
  assert.deepEqual(low.rows.map((row) => row.id), ['boundary']);
  assert.deepEqual(low.leaders.speed, ['boundary']);
  assert.equal(low.unknownCount, 2);
  assert.equal(low.overBudgetCount, 1);
  const medium = sotaTable({ rows }, 'Prepared', metrics, 2 ** 31);
  assert.deepEqual(medium.rows.map((row) => row.id), ['boundary', 'outlier']);
  assert.deepEqual(medium.leaders.speed, ['outlier']);
  assert.equal(memoryBudgetRows(rows).rows.length, 5);
  assert.deepEqual(sotaTable({ rows }, 'Prepared', metrics, 1).leaders.speed, []);
  for (const value of [-1, 0, 1.5, NaN, 'low']) assert.throws(() => memoryBudgetRows(rows, value));
  for (const peaks of [[], [0], [-1], [NaN], [1.5], ['1']]) assert.throws(() => clientMemoryEvidence(peaks, 'test'));
});

test('published budgets share the installed SDK definition and all evidence windows survive', () => {
  const card = buildScorecard();
  const sdk = JSON.parse(fs.readFileSync(new URL('../../python/pllm/search/client_memory_tiers.json', import.meta.url)));
  assert.deepEqual(card.clientMemoryTiers, sdk);
  assert.deepEqual(sdk.map(({ id, maxBytes }) => [id, maxBytes]), [
    ['low', 256 * 2 ** 20], ['medium', 2 * 2 ** 30], ['high', 16 * 2 ** 30],
  ]);
  for (const cohort of card.cohorts) for (const row of cohort.rows) {
    assert.ok(row.clientMemory.scope);
    assert.equal(row.clientMemory.sampleCount, row.sampleCount ?? cohort.replicates ?? 1);
    if (row.metrics.clientPeak === null) assert.equal(row.clientMemory.maxPeakBytes, null);
    else assert.ok(row.clientMemory.maxPeakBytes >= Math.round(row.metrics.clientPeak * 1e6));
    if (row.observations) assert.equal(row.clientMemory.maxPeakBytes,
      Math.max(...row.observations.map((sample) => Math.round(sample.metrics.clientPeak * 1e6))));
  }
  const unknown = card.cohorts.find((cohort) => cohort.id === 'qwen25-prepared-150-8');
  assert.equal(memoryBudgetRows(unknown.rows, 16 * 2 ** 30).rows.length, 0);
  const qwen3 = card.cohorts.find((cohort) => cohort.id === 'qwen3-sdk-tokenizer-16-8');
  // 262.73 decimal MB fits 256 binary MiB; its resident-tokenizer control does not.
  assert.equal(memoryBudgetRows(qwen3.rows, 256 * 2 ** 20).rows.length, 1);
});
