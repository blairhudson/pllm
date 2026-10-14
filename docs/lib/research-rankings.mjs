/** Zero and unknown remain distinct; ties survive. Rankings never cross cohorts. */
export function leaders(rows, metric) {
  const measured = rows.filter((row) => Number.isFinite(row.metrics[metric.id]));
  if (!measured.length) return [];
  const values = measured.map((row) => row.metrics[metric.id]);
  const best = metric.direction === 'min' ? Math.min(...values) : Math.max(...values);
  return measured.filter((row) => row.metrics[metric.id] === best).map((row) => row.id);
}

/** Sort numeric observations without mutating the cohort; unknowns stay last. */
export function sortMetricRows(rows, metricId, direction) {
  return [...rows].sort((a, b) => {
    const left = a.metrics[metricId];
    const right = b.metrics[metricId];
    if (!Number.isFinite(left)) return Number.isFinite(right) ? 1 : 0;
    if (!Number.isFinite(right)) return -1;
    return direction === 'ascending' ? left - right : right - left;
  });
}

export function improvement(value, baseline, direction) {
  if (!Number.isFinite(value) || !Number.isFinite(baseline) || value < 0 || baseline <= 0) return null;
  return (direction === 'min' ? 1 - value / baseline : value / baseline - 1) * 100;
}

// Research objectives and their available chart metrics. An empty list means the
// objective needs attribution absent from the historical reports, not zero cost.
export const priorities = [
  { label: 'Client memory', anchor: 'client-memory', metrics: ['clientPeak'],
    why: 'Fit the complete client on an ordinary device while keeping substantial work remote. Paging and tiling can trade RAM for CPU, I/O or exchanges.' },
  { label: 'Client CPU', anchor: 'client-cpu', metrics: ['clientCpu'],
    why: 'Leave device capacity for the application. Compression, verification and low-memory storage can increase local work.' },
  { label: 'Preparation CPU', anchor: 'preparation-cpu', metrics: ['preparationCpu'],
    why: 'Produce fresh material affordably, including unused reservations. Batching may save CPU while increasing memory and waiting.' },
  { label: 'Preparation memory', anchor: 'preparation-memory', metrics: ['preparationPeak'],
    why: 'Make Preparation practical on hardware under your control. More concurrent issuance can require larger live workspaces.' },
  { label: 'Time to first token', anchor: 'time-to-first-token', metrics: ['ttft'],
    why: 'Reduce the wait for an answer. This chart measures online TTFT; cold setup, admission and queueing need separate lifecycle accounting.' },
  { label: 'Decode throughput', anchor: 'decode-throughput', metrics: ['decodeTps'],
    why: 'Keep later tokens responsive. Count epoch pauses in N−1 decode speed; lower material waste can mean slower generation.' },
  { label: 'Base all-link traffic', anchor: 'base-all-link-traffic', metrics: [],
    why: 'Reduce fixed setup, public delivery and prompt-prefill bytes for stated cold or warm conditions. Retained caches can lower this base but cost storage.' },
  { label: 'Per-token all-link traffic', anchor: 'per-token-all-link-traffic', metrics: [],
    why: 'Reduce output-dependent bytes across every link, including prepaid and unused output material. Whole-response MB/token mixes this cost with the base.' },
];
// Keep the measured total visible until a reconciled base/generation split exists.
export const defaultMetrics = [...priorities.flatMap((priority) => priority.metrics), 'totalNetwork'];

// Always show the eight objectives, even when attribution is unavailable.
// The measured total is a separate reconciliation column, never a proxy split.
export const sotaMetrics = [...priorities.flatMap((priority) => priority.metrics),
  'baseNetwork', 'tokenNetwork', 'totalNetwork'];

/** A cohort may measure a different, explicitly named window on the same axis. */
export function scopedMetric(metric, measurement) {
  const label = measurement?.metricLabels?.[metric.id] ?? ({
    clientPeak: measurement?.clientRssLabel, clientCpu: measurement?.clientCpuLabel,
    preparationCpu: measurement?.preparationCpuLabel,
  })[metric.id];
  return label ? { ...metric, shortLabel: label, label } : metric;
}

// Extra decimals prevent nearby observations from looking like a tied result.
function metricPrecision(rows, metric) {
  const values = [...new Set(rows.map((row) => row.metrics[metric.id]).filter(Number.isFinite))];
  for (let digits = 2; digits < 9; digits++) {
    if (new Set(values.map((value) => value.toLocaleString('en', { maximumFractionDigits: digits }))).size === values.length) return digits;
  }
  return 9;
}

/** Nested maximum budgets use all observed process peaks, never median RSS. */
export function memoryBudgetRows(rows, maxBytes = null) {
  if (maxBytes !== null && (!Number.isSafeInteger(maxBytes) || maxBytes <= 0)) throw new Error('Invalid client memory maximum');
  const known = (row) => Number.isSafeInteger(row.clientMemory?.maxPeakBytes) && row.clientMemory.maxPeakBytes > 0;
  return { rows: maxBytes === null ? rows : rows.filter((row) => known(row) && row.clientMemory.maxPeakBytes <= maxBytes),
    unknownCount: rows.filter((row) => !known(row)).length,
    overBudgetCount: maxBytes === null ? 0 : rows.filter((row) => known(row) && row.clientMemory.maxPeakBytes > maxBytes).length };
}

/** Open the newest registered matching report, cold-first, without pooling cohorts. */
export function defaultResearchCohort(cohorts, maxBytes = null) {
  const eligible = cohorts.filter((cohort) => {
    const topologies = cohort.rows.map((row) => row.placement.topology);
    const topology = topologies.includes('Prepared') ? 'Prepared' : topologies[0];
    return memoryBudgetRows(cohort.rows.filter((row) => row.placement.topology === topology), maxBytes).rows.length > 0;
  });
  const choices = eligible.length ? eligible : cohorts;
  const report = choices.at(-1)?.report;
  return choices.find((cohort) => cohort.report === report && cohort.identity.warm === false) ??
    choices.find((cohort) => cohort.report === report);
}

/** A table compares one checked cohort, topology and optional maximum at a time. */
export function sotaTable(cohort, topology, metrics, maxBytes = null) {
  const budget = memoryBudgetRows(cohort.rows.filter((row) => row.placement.topology === topology), maxBytes);
  const { rows } = budget;
  return { ...budget, leaders: Object.fromEntries(metrics.map((metric) => [metric.id, leaders(rows, metric)])),
    precision: Object.fromEntries(metrics.map((metric) => [metric.id, metricPrecision(rows, metric)])) };
}

const componentNames = {
  'pllm/apple-metal-int8/v1': 'AppleMetal',
  'pllm/cleartext-linear': 'CleartextLinear',
  'pllm/client-bundle-transport/v1': 'ClientBundleTransport',
  'pllm/client-only/v1': 'ClientOnlyRoles',
  'pllm/client-owned-linear-roles/v1': 'ClientLinearRoles',
  'pllm/client-prefix-reuse/v1': 'ClientPrefixReuse',
  'pllm/cpu': 'Cpu',
  'pllm/freivalds-verify/v1': 'FreivaldsVerify',
  'pllm/indexed-tokenizer/v1': 'IndexedTokenizer',
  'pllm/inference': 'Inference',
  'pllm/masked-linear': 'MaskedLinear',
  'pllm/masked-wire-gated-multiply-q12/v1': 'MaskedWireGatedMultiplyQ12',
  'pllm/mlp-prefill-tiling/v1': 'MlpPrefillTiling',
  'pllm/model-aware-corrections': 'ModelAwareCorrections',
  'pllm/prepared-inventory-policy/v1': 'PreparedInventory',
  'pllm/prepared-body-two-online-boundary/v1': 'PreparedBoundaryRoles',
  'pllm/private-lookup-full-head/v1': 'PrivateLookupAndFullHead',
  'pllm/public-per-channel-equalized/v1': 'PublicPerChannelEqualized',
  'pllm/public-prefix-capsule/v1': 'PublicPrefixCapsule',
  'pllm/symmetric-per-row-quantization/v1': 'SymmetricPerRow',
  'pllm/two-online-offset-linear/v1': 'TwoOnlineOffsetLinear',
  'pllm/two-online-offset-workers/v1': 'TwoOnlineOffsetRoles',
};

/** Names improve readability; immutable component IDs and every parameter survive. */
export function componentCombination(pipeline) {
  return Object.entries(pipeline.components).filter(([, component]) => component !== undefined).map(([slot, component]) => ({
    slot, name: componentNames[component.component] ?? component.component, ...component,
  }));
}

export function groupedMetrics(metrics) {
  return ['Client', 'Preparation', 'Performance', 'Network', 'Agg', 'Providers'].map((label) => ({
    label, title: label === 'Agg' ? 'Aggregate across all roles' : `${label} metrics`,
    metrics: metrics.filter((metric) => metric.group === label),
  })).filter((group) => group.metrics.length);
}

/** Costs use observed ranges; bounded shares retain their absolute scale. */
export function normalizedAxes(rows, metrics) {
  return metrics.map((metric) => {
    const values = rows.map((row) => row.metrics[metric.id]).filter(Number.isFinite);
    const min = values.length ? Math.min(...values) : null;
    const max = values.length ? Math.max(...values) : null;
    const [low, high] = metric.scoreBounds ?? [min, max];
    if (metric.scoreBounds && (metric.scoreBounds.length !== 2 || !Number.isFinite(low) ||
        !Number.isFinite(high) || low >= high || values.some((value) => value < low || value > high))) {
      throw new Error('Metric outside score bounds');
    }
    return { ...metric, min, max, scores: Object.fromEntries(rows.map((row) => {
      const value = row.metrics[metric.id];
      const score = !Number.isFinite(value) ? null : low === high ? 100 :
        100 * (metric.direction === 'min' ? high - value : value - low) / (high - low);
      return [row.id, score];
    })) };
  });
}

/** Contiguous runs only: a line must not imply a measurement across an unknown. */
export function measuredSegments(values) {
  const segments = [];
  let current = [];
  for (const [index, value] of values.entries()) {
    if (Number.isFinite(value)) current.push({ index, value });
    else if (current.length) { segments.push(current); current = []; }
  }
  if (current.length) segments.push(current);
  return segments;
}
