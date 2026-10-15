/** Descriptive associations between configurations, never pooled benchmark trials. */
import { defaultMetrics, memoryBudgetRows } from './research-rankings.mjs';

export const relationshipMetrics = [...defaultMetrics, 'full', 'aggregateCpu'];
// A display floor, not a significance test. Two-point correlations are tautologies.
export const minimumConfigurations = 5;

function pearson(x, y) {
  const center = (values) => {
    const scale = Math.max(...values.map(Math.abs));
    if (scale === 0) return values;
    const scaled = values.map((value) => value / scale);
    const mean = scaled.reduce((sum, value) => sum + value, 0) / values.length;
    return scaled.map((value) => value - mean);
  };
  const a = center(x), b = center(y);
  const xx = a.reduce((sum, value) => sum + value * value, 0);
  const yy = b.reduce((sum, value) => sum + value * value, 0);
  if (xx === 0 || yy === 0) return null;
  const xy = a.reduce((sum, value, index) => sum + value * b[index], 0);
  return Math.max(-1, Math.min(1, xy / Math.sqrt(xx * yy)));
}

function ranks(values) {
  const sorted = values.map((value, index) => ({ value, index })).sort((a, b) => a.value - b.value);
  const result = Array(values.length);
  for (let start = 0; start < sorted.length;) {
    let end = start + 1;
    while (end < sorted.length && sorted[end].value === sorted[start].value) end++;
    for (let i = start; i < end; i++) result[sorted[i].index] = (start + 1 + end) / 2;
    start = end;
  }
  return result;
}

function coefficients(points) {
  const x = points.map((point) => point.x), y = points.map((point) => point.y);
  // Exact equality matters: a nonzero constant must not gain rounding variance.
  if (new Set(x).size < 2 || new Set(y).size < 2) return { pearson: null, spearman: null };
  return { pearson: pearson(x, y), spearman: pearson(ranks(x), ranks(y)) };
}

const extent = (values) => values.length ? { min: Math.min(...values), max: Math.max(...values) } : null;

export function metricRelationship(rows, xMetric, yMetric) {
  const points = rows.filter((row) => Number.isFinite(row.metrics[xMetric.id]) && Number.isFinite(row.metrics[yMetric.id]))
    .map((row) => ({ id: row.id, label: row.label, topology: row.placement.topology,
      x: row.metrics[xMetric.id], y: row.metrics[yMetric.id] }));
  const n = points.length;
  const status = n < minimumConfigurations ? 'insufficient' :
    new Set(points.map((point) => point.x)).size < 2 || new Set(points.map((point) => point.y)).size < 2 ? 'constant' : 'measured';
  const values = status === 'measured' ? coefficients(points) : { pearson: null, spearman: null };
  const omissions = status === 'measured' ? points.map((_, index) => coefficients(points.filter((_, i) => i !== index))) : [];
  const sensitivity = (method) => {
    const defined = omissions.map((sample) => sample[method]).filter(Number.isFinite);
    return { range: extent(defined), defined: defined.length, omitted: omissions.length - defined.length };
  };
  return { x: xMetric.id, y: yMetric.id, n, missing: rows.length - n, status, points, ...values,
    xRange: extent(points.map((point) => point.x)), yRange: extent(points.map((point) => point.y)),
    // Positive = improvements align; raw coefficients above preserve physical units/direction.
    alignedPearson: values.pearson === null ? null : values.pearson * (xMetric.direction === yMetric.direction ? 1 : -1),
    alignedSpearman: values.spearman === null ? null : values.spearman * (xMetric.direction === yMetric.direction ? 1 : -1),
    leaveOneOut: { pearson: sensitivity('pearson'), spearman: sensitivity('spearman') } };
}

/** Call with one checked scorecard cohort; filters also define the correlation population. */
export function relationshipTable(cohort, topology, metrics, maxBytes = null, includeTopologies = false) {
  if (!cohort?.id || !Array.isArray(cohort.rows) || new Set(cohort.rows.map((row) => row.id)).size !== cohort.rows.length) {
    throw new Error('Relationships require one cohort with distinct configuration rows');
  }
  if (typeof topology !== 'string' || typeof includeTopologies !== 'boolean') throw new Error('Explicit topology required');
  if (new Set(metrics.map((metric) => metric.id)).size !== metrics.length ||
      metrics.some((metric) => !['min', 'max'].includes(metric.direction))) throw new Error('Invalid relationship metrics');
  const budget = memoryBudgetRows(cohort.rows.filter((row) => includeTopologies || row.placement.topology === topology), maxBytes);
  const pairs = metrics.flatMap((x, i) => metrics.slice(i + 1).map((y) => metricRelationship(budget.rows, x, y)));
  return { ...budget, cohortId: cohort.id, topology: includeTopologies ? null : topology,
    includeTopologies, maxBytes, minimumConfigurations, pairs };
}
