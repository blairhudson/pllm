import test from 'node:test';
import assert from 'node:assert/strict';
import { buildScorecard } from '../scripts/research-scorecard.mjs';
import { metricRelationship, relationshipMetrics, relationshipTable } from '../lib/research-relationships.mjs';

const metrics = [{ id: 'x', direction: 'min' }, { id: 'y', direction: 'max' }];
const rows = (x, y) => x.map((value, i) => ({ id: String(i), label: `Choice ${i}`, placement: { topology: 'Prepared' },
  metrics: { x: value, y: y[i] }, clientMemory: { maxPeakBytes: 128 * 2 ** 20 } }));
const near = (value, expected) => assert.ok(Math.abs(value - expected) < 1e-12, `${value} != ${expected}`);
const pair = (data) => metricRelationship(data, ...metrics);

test('raw Pearson and rank Spearman differ for a nonlinear monotonic relationship', () => {
  const result = pair(rows([1, 2, 3, 4, 5], [1, 4, 9, 16, 25]));
  near(result.pearson, 0.9811049102515929);
  near(result.spearman, 1);
  near(result.alignedSpearman, -1);
  assert.equal(result.n, 5);
  assert.equal(result.status, 'measured');
  assert.equal(result.leaveOneOut.pearson.defined, 5);
});

test('inverse relationship, exact ties, zero and units retain their meanings', () => {
  const data = rows([0, 1, 1, 3, 4], [8, 6, 6, 2, 0]);
  const before = structuredClone(data);
  const result = pair(data);
  near(result.pearson, -1);
  near(result.spearman, -1);
  near(result.alignedPearson, 1);
  const scaled = pair(rows([0, 1e300, 1e300, 3e300, 4e300], [8, 6, 6, 2, 0]));
  near(scaled.pearson, -1);
  assert.deepEqual(data, before);
  assert.equal(result.points[0].x, 0);
});

test('average tie ranks match a hand-computed nontrivial example', () => {
  // Centered ranks: (-1.5,-1.5,0,1.5,1.5), (1.5,-2,-.5,-.5,1.5).
  const result = pair(rows([1, 1, 2, 3, 3], [3, 1, 2, 2, 3]));
  near(result.spearman, 2.25 / 9);
});

test('missing observations are excluded pairwise, never converted to zero', () => {
  const data = rows([0, 1, 2, 3, 4, 5, null, 7, NaN], [0, 2, 4, 6, 8, null, 12, Infinity, 18]);
  const result = pair(data);
  assert.equal(result.n, 5);
  assert.equal(result.missing, 4);
  near(result.pearson, 1);
  assert.deepEqual(result.points.map((point) => point.id), ['0', '1', '2', '3', '4']);
});

test('small and constant samples remain undefined, including nonzero floating constants', () => {
  for (let n = 0; n < 5; n++) {
    const result = pair(rows(Array.from({ length: n }, (_, i) => i), Array.from({ length: n }, (_, i) => i)));
    assert.equal(result.status, 'insufficient');
    assert.equal(result.pearson, null);
    assert.equal(result.spearman, null);
    assert.equal(result.points.length, n);
  }
  for (const constant of [0, 0.3, 1e20]) {
    const result = pair(rows([constant, constant, constant, constant, constant], [1, 2, 3, 4, 5]));
    assert.equal(result.status, 'constant');
    assert.equal(result.pearson, null);
    assert.equal(result.spearman, null);
  }
});

test('leave-one-configuration-out exposes outliers and undefined constant omissions', () => {
  const result = pair(rows([1, 2, 3, 4, 5], [1, 2, 3, 4, -20]));
  assert.ok(result.pearson < 0);
  near(result.leaveOneOut.pearson.range.max, 1);
  assert.ok(result.leaveOneOut.pearson.range.min < 0);
  const singleton = pair(rows([1, 1, 1, 1, 2], [1, 2, 3, 4, 5]));
  assert.equal(singleton.leaveOneOut.pearson.omitted, 1);
  assert.equal(singleton.leaveOneOut.pearson.defined, 4);
});

test('RAM ceilings use maximum replicate RSS and topology mixing is explicit', () => {
  const data = rows([0, 1, 2, 3, 4, 5, 6, 7], [7, 6, 5, 4, 3, 2, 1, 0]);
  data[5].clientMemory.maxPeakBytes = 256 * 2 ** 20;
  data[6].clientMemory.maxPeakBytes = 256 * 2 ** 20 + 1;
  data[6].metrics.clientPeak = 1; // A tiny median does not excuse a larger replicate.
  data[7].clientMemory.maxPeakBytes = null;
  data[4].placement.topology = 'Prepared + private boundary';
  const cohort = { id: 'one-cohort', rows: data };
  const low = relationshipTable(cohort, 'Prepared', metrics, 256 * 2 ** 20);
  assert.equal(low.rows.length, 5);
  assert.equal(low.overBudgetCount, 1);
  assert.equal(low.unknownCount, 1);
  assert.equal(low.pairs[0].n, 5);
  const mixed = relationshipTable(cohort, 'Prepared', metrics, 256 * 2 ** 20, true);
  assert.equal(mixed.rows.length, 6);
  assert.equal(mixed.topology, null);
  const medium = relationshipTable(cohort, 'Prepared', metrics, 2 ** 31);
  assert.equal(medium.rows.length, 6);
  assert.equal(relationshipTable(cohort, 'Prepared', metrics).rows.length, 7);
  assert.throws(() => relationshipTable([cohort, cohort], 'Prepared', metrics), /one cohort/);
  assert.throws(() => relationshipTable({ ...cohort, rows: [...data, data[0]] }, 'Prepared', metrics), /distinct/);
});

test('replicates and missing third-axis values cannot inflate or shrink a pair', () => {
  const data = rows([1, 2, 3, 4, 5], [1, 2, 3, 5, 4]);
  for (const row of data) {
    row.sampleCount = 20;
    row.observations = Array(20).fill({ metrics: { x: 999, y: 0 } });
    row.metrics.z = null;
  }
  const result = relationshipTable({ id: 'cohort', rows: data }, 'Prepared', [...metrics, { id: 'z', direction: 'min' }]);
  assert.equal(result.pairs.length, 3);
  assert.equal(result.pairs[0].n, 5);
  near(result.pairs[0].spearman, 0.9);
  assert.equal(result.pairs[1].n, 0);
  assert.equal(result.pairs[2].n, 0);
});

test('checked real evidence preserves Low/Medium populations and exposes topology reversal', () => {
  const card = buildScorecard();
  const cohort = card.cohorts.find((cohort) => cohort.id === 'qwen25-private-boundary-tiers-150-8');
  const definitions = relationshipMetrics.map((id) => card.metrics.find((metric) => metric.id === id));
  const low = relationshipTable(cohort, 'Prepared', definitions, 256 * 2 ** 20);
  const medium = relationshipTable(cohort, 'Prepared', definitions, 2 ** 31);
  assert.deepEqual(low.rows.map((row) => row.id), medium.rows.map((row) => row.id));
  assert.deepEqual(low.pairs, medium.pairs);
  assert.equal(low.rows.length, 7);
  assert.equal(low.pairs.length, 36);
  const decode = (table) => table.pairs.find((pair) => pair.x === 'decodeTps' && pair.y === 'totalNetwork');
  near(decode(low).pearson, -0.9993870292789795);
  near(decode(low).spearman, -0.8928571428571429);
  const mixed = relationshipTable(cohort, 'Prepared', definitions, 256 * 2 ** 20, true);
  near(decode(mixed).pearson, 0.8248074686212913);
  near(decode(mixed).leaveOneOut.pearson.range.min, decode(low).pearson);
  const finalists = card.cohorts.find((cohort) => cohort.id === 'qwen25-private-boundary-decode-confirmation-150-8');
  const sparse = relationshipTable(finalists, 'Prepared', definitions, 256 * 2 ** 20, true);
  assert.ok(sparse.pairs.every((pair) => pair.n === 3 && pair.pearson === null));
});
