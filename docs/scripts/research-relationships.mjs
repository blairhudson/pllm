/** Read-only replay from hash-checked scorecard evidence; no model execution. */
import { parseArgs } from 'node:util';
import { buildScorecard } from './research-scorecard.mjs';
import { relationshipMetrics, relationshipTable } from '../lib/research-relationships.mjs';
import { scopedMetric } from '../lib/research-rankings.mjs';

const { values } = parseArgs({ options: {
  cohort: { type: 'string' }, memory: { type: 'string', default: 'low' },
  topology: { type: 'string', default: 'Prepared' }, 'include-topologies': { type: 'boolean', default: false },
} });
const card = buildScorecard();
const cohort = card.cohorts.find((item) => item.id === values.cohort);
if (!cohort) throw new Error('Select --cohort with an exact registered cohort ID');
const tier = card.clientMemoryTiers.find((item) => item.id === values.memory);
if (!tier && values.memory !== 'all') throw new Error('Choose --memory low, medium, high or all');
if (!cohort.rows.some((row) => row.placement.topology === values.topology)) throw new Error('Topology absent from cohort');
const metrics = relationshipMetrics.map((id) => scopedMetric(card.metrics.find((metric) => metric.id === id), cohort.measurement));
const result = relationshipTable(cohort, values.topology, metrics, tier?.maxBytes ?? null, values['include-topologies']);
console.log(JSON.stringify({ schema: 'pllm.research_metric_relationships.v1',
  title: cohort.title, model: cohort.model, identity: cohort.identity, evidenceTier: cohort.evidenceTier,
  report: cohort.report, reportSha256: cohort.reportSha256, measurement: cohort.measurement ?? null,
  method: 'One existing configuration summary per point; pairwise-complete raw Pearson and average-tie-rank Spearman. No cross-cohort pooling, causal inference or significance test. Leave-one-out ranges are sensitivity, not confidence intervals.',
  metrics, ...result,
  rows: result.rows.map((row) => ({ id: row.id, label: row.label, metrics: row.metrics,
    placement: row.placement, clientMemory: row.clientMemory, sampleCount: row.sampleCount ?? cohort.measurement?.sampleCount ?? null })),
}, null, 2));
