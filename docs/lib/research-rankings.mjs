/** Zero and unknown remain distinct; ties survive. Rankings never cross cohorts. */
export function leaders(rows, metric) {
  const measured = rows.filter((row) => Number.isFinite(row.metrics[metric.id]));
  if (!measured.length) return [];
  const values = measured.map((row) => row.metrics[metric.id]);
  const best = metric.direction === 'min' ? Math.min(...values) : Math.max(...values);
  return measured.filter((row) => row.metrics[metric.id] === best).map((row) => row.id);
}

export function improvement(value, baseline, direction) {
  if (!Number.isFinite(value) || !Number.isFinite(baseline) || value < 0 || baseline <= 0) return null;
  return (direction === 'min' ? 1 - value / baseline : value / baseline - 1) * 100;
}

// One source for the chart's default axes and their visible rationale.
export const priorities = [
  { label: 'Client memory', metrics: ['clientPeak'],
    why: 'Peak RSS shows the client memory footprint, helping assess device capacity and swap pressure.' },
  { label: 'Client CPU', metrics: ['clientCpu'],
    why: 'Less local computation leaves more device capacity for the application using the model.' },
  { label: 'Preparation cost', metrics: ['preparationCpu', 'preparationPeak'],
    why: 'Low offline CPU and RAM make Preparation easier to run under your control, separate from online Inference.' },
  { label: 'WAN throughput', metrics: ['requestTps', 'decodeTps'],
    why: 'Request speed includes request-bound setup and first-output wait; decode speed measures later tokens. Capped cohorts show how both behave on constrained links.' },
  { label: 'All-link traffic', metrics: ['totalNetwork'],
    why: 'Every link consumes bandwidth. Including correction delivery and setup keeps work shifted away from the client visible.' },
];
export const defaultMetrics = priorities.flatMap((priority) => priority.metrics);

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
