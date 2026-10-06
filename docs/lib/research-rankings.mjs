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
