'use client';

import { useId, useState, type CSSProperties } from 'react';
import { metricRelationship, minimumConfigurations, relationshipTable,
  type RelationshipMetric, type RelationshipRow } from '@/lib/research-relationships.mjs';
import styles from './research-relationships.module.css';

type Metric = RelationshipMetric & { label: string; shortLabel: string; unit: string };
type Props = { cohort: { id: string; rows: RelationshipRow[] }; topology: string;
  metrics: Metric[]; maxBytes: number | null };
const coefficient = (value: number | null) => value === null ? '—' : `${value > 0 ? '+' : ''}${value.toFixed(3)}`;
const number = (value: number, range: { min: number; max: number } | null) => value.toLocaleString('en', {
  maximumSignificantDigits: range && range.max > range.min && (range.max - range.min) < Math.abs(value) * 0.0001 ? 12 : 6,
});
const axisLabels: Record<string, string> = { clientPeak: 'Client RAM', clientCpu: 'Client CPU', preparationCpu: 'Prep CPU' };
const axisLabel = (metric: Metric) => axisLabels[metric.id] ?? metric.shortLabel;

export function ResearchRelationships({ cohort, topology, metrics, maxBytes }: Props) {
  const id = useId();
  const [method, setMethod] = useState<'pearson' | 'spearman'>('pearson');
  const [crossTopology, setCrossTopology] = useState(false);
  const [selected, setSelected] = useState(['totalNetwork', 'decodeTps']);
  const table = relationshipTable(cohort, topology, metrics, maxBytes, crossTopology);
  const x = metrics.find((metric) => metric.id === selected[0]) ?? metrics[0];
  const y = metrics.find((metric) => metric.id === selected[1]) ?? metrics[1];
  const pair = metricRelationship(table.rows, x, y);
  const sensitivity = pair.leaveOneOut[method];
  const topologies = [...new Set(cohort.rows.map((row) => row.placement.topology))];
  const displayTopologies = [...new Set(table.rows.map((row) => row.placement.topology))];
  const aligned = method === 'pearson' ? pair.alignedPearson : pair.alignedSpearman;
  const position = (value: number, range: { min: number; max: number } | null, start: number, length: number) =>
    range === null || range.min === range.max ? start + length / 2 : start + length * (value - range.min) / (range.max - range.min);
  const color = (topology: string) => `var(--series-${displayTopologies.indexOf(topology) % 5})`;

  return <section className={styles.relationships} aria-labelledby={`${id}-title`}>
    <div className={styles.heading}>
      <div><h3 id={`${id}-title`}>How metrics move together</h3>
        <p>One point per configuration summary in this matched cohort, after the RAM filter.
          Replicates are not extra points. Coefficients describe the tested choices; they do not establish causation.</p></div>
      <label htmlFor={`${id}-method`}>Association<select id={`${id}-method`} value={method}
        onChange={(event) => setMethod(event.target.value as 'pearson' | 'spearman')}>
        <option value="pearson">Pearson r · linear</option><option value="spearman">Spearman ρ · ranks</option>
      </select></label>
    </div>
    {topologies.length > 1 && <label className={styles.toggle}><input type="checkbox" checked={crossTopology}
      onChange={(event) => setCrossTopology(event.target.checked)} />Compare topologies within this cohort</label>}
    <p className={styles.note} role="status">{table.rows.length} configurations · {crossTopology ? 'cross-topology comparison' : topology}
      {' '}· {table.overBudgetCount} over RAM budget · {table.unknownCount} with unknown RAM
      {maxBytes !== null ? ' (excluded)' : ' (included)'}.
      {crossTopology && ' Topology changes ownership and trust assumptions; pooled associations can reverse within a topology.'}
      {' '}At least {minimumConfigurations} paired configurations required to display a coefficient; this is a display floor, not statistical significance.</p>
    <div className={styles.matrixWrap} tabIndex={0} role="region" aria-label="Metric correlation matrix">
      <table className={styles.matrix}>
        <caption>{method === 'pearson' ? 'Pearson r' : 'Spearman ρ'} · raw values · click a pair for its scatter plot</caption>
        <thead><tr><th scope="col">Metric</th>{metrics.map((metric) => <th scope="col" key={metric.id} title={metric.label}>{metric.shortLabel}</th>)}</tr></thead>
        <tbody>{metrics.map((a) => <tr key={a.id}><th scope="row" title={a.label}>{a.shortLabel}</th>{metrics.map((b) => {
          if (a.id === b.id) return <td key={b.id} className={styles.diagonal} aria-label="Same metric">·</td>;
          const entry = table.pairs.find((pair) => pair.x === a.id && pair.y === b.id || pair.x === b.id && pair.y === a.id)!;
          const value = entry[method];
          const reason = entry.status === 'constant' ? 'constant values' : entry.status === 'insufficient' ? 'too few configurations' : '';
          const active = (x.id === a.id && y.id === b.id) || (x.id === b.id && y.id === a.id);
          return <td key={b.id}><button type="button" aria-pressed={active}
            aria-label={`${a.shortLabel} versus ${b.shortLabel}: ${coefficient(value)}, n=${entry.n}${reason ? `, ${reason}` : ''}`}
            title={`${a.label} / ${b.label}: ${value ?? reason}; n=${entry.n}`}
            data-sign={value === null ? 'unknown' : value >= 0 ? 'positive' : 'negative'}
            style={{ '--strength': value === null ? 0 : `${Math.abs(value) * 24}%` } as CSSProperties}
            onClick={() => setSelected([a.id, b.id])}>
            <strong>{coefficient(value)}</strong><small>n={entry.n}</small>
          </button></td>;
        })}</tr>)}</tbody>
      </table>
    </div>
    <p className={styles.note}>+1: rise together. −1: move oppositely. 0: little {method === 'pearson' ? 'linear' : 'rank'} association.
      A positive sign is not automatically good: RAM/CPU/latency/traffic should fall, decode throughput should rise.
      Unknown and constant columns have no coefficient. Pair counts may differ.</p>
    <div className={styles.plotControls}>
      {([['x', x], ['y', y]] as const).map(([axis, metric], i) => <label key={axis} htmlFor={`${id}-${axis}`}>{axis.toUpperCase()} axis
        <select id={`${id}-${axis}`} value={metric.id} onChange={(event) => setSelected(i === 0 ? [event.target.value, y.id] : [x.id, event.target.value])}>
          {metrics.filter((option) => option.id !== (i === 0 ? y.id : x.id)).map((option) => <option key={option.id} value={option.id}>{option.shortLabel}</option>)}
        </select></label>)}
    </div>
    <p className={styles.stats} aria-live="polite">
      <strong>Pearson r {coefficient(pair.pearson)} · Spearman ρ {coefficient(pair.spearman)} · n={pair.n}</strong>
      {aligned !== null && <span>{Math.abs(aligned) < 0.1 ? 'Little association' : aligned > 0 ? 'Improvement directions align' : 'Improvement directions oppose'} in this filtered sample.</span>}
      {pair.status === 'insufficient' && <span>Too few configurations for a coefficient. Raw points remain visible.</span>}
      {pair.status === 'constant' && <span>At least one metric is constant.</span>}
      {sensitivity.range && <span>Leave one configuration out · {method === 'pearson' ? 'Pearson r' : 'Spearman ρ'}: {coefficient(sensitivity.range.min)} to {coefficient(sensitivity.range.max)}
        {' '}({sensitivity.defined} defined; {sensitivity.omitted} constant). Sensitivity range, not a confidence interval.</span>}
    </p>
    {pair.points.length > 0 ? <>
      <svg className={styles.plot} viewBox="0 0 640 335" role="img" aria-labelledby={`${id}-plot-title ${id}-plot-description`}>
        <title id={`${id}-plot-title`}>{`${x.label} versus ${y.label}`}</title>
        <desc id={`${id}-plot-description`}>{pair.n} configuration summaries. Point numbers match the table below. Axes show observed ranges, not zero-based bars.</desc>
        <path d="M 88 30 V 260 H 592" className={styles.axis} />
        {[0, 0.5, 1].map((fraction) => <g key={fraction}>
          <path d={`M 88 ${248 - fraction * 200} H 592`} className={styles.grid} />
          {pair.yRange && <text x="80" y={252 - fraction * 200} textAnchor="end">{number(pair.yRange.min + fraction * (pair.yRange.max - pair.yRange.min), pair.yRange)}</text>}
          {pair.xRange && <text x={104 + fraction * 472} y="280" textAnchor="middle">{number(pair.xRange.min + fraction * (pair.xRange.max - pair.xRange.min), pair.xRange)}</text>}
        </g>)}
        <text x="340" y="317" textAnchor="middle">{axisLabel(x)} ({x.unit}) · {x.direction === 'min' ? 'lower' : 'higher'} better</text>
        <text x="18" y="150" textAnchor="middle" transform="rotate(-90 18 150)">{axisLabel(y)} ({y.unit})</text>
        {pair.points.map((point, index) => <g key={point.id} className={styles.point} tabIndex={0} role="img"
          aria-label={`${index + 1}. ${point.label}: ${x.shortLabel} ${point.x} ${x.unit}, ${y.shortLabel} ${point.y} ${y.unit}`}>
          <title>{`${point.label} · ${point.topology}: ${point.x} ${x.unit}, ${point.y} ${y.unit}`}</title>
          <circle cx={position(point.x, pair.xRange, 104, 472)} cy={position(point.y, pair.yRange, 248, -200)} r="7" fill={color(point.topology)} />
          <text x={position(point.x, pair.xRange, 104, 472) + 10} y={position(point.y, pair.yRange, 248, -200) - 9}>{index + 1}</text>
        </g>)}
      </svg>
      <div className={styles.legend}>{displayTopologies.map((value) => <span key={value}><i style={{ backgroundColor: color(value) }} />{value}</span>)}</div>
      <details className={styles.points}><summary>Exact plotted values · {pair.missing} missing pairs excluded</summary>
        <div className={styles.matrixWrap}><table><thead><tr><th>Point / configuration</th><th>{x.label} ({x.unit})</th><th>{y.label} ({y.unit})</th></tr></thead>
          <tbody>{pair.points.map((point, index) => <tr key={point.id}><th scope="row">{index + 1}. {point.label}<small>{point.topology}</small></th>
            <td>{String(point.x)}</td><td>{String(point.y)}</td></tr>)}</tbody></table></div>
      </details>
    </> : <p className={styles.note}>No configurations measure both selected axes under this filter.</p>}
    <p className={styles.note}>Pearson measures linear association; Spearman uses average ranks for exact ties.
      Ranks can exaggerate tiny framing-byte differences; compare both coefficients and the raw ranges.
      CPU totals contain role CPU, so their correlation is partly built in. Request throughput and whole-request MB/token averages are derived duplicates.
      Complete base/per-token attribution remains unknown.
      Different lifecycle windows remain visible in metric labels and the cohort description above. RAM filtering can change the relationship.</p>
  </section>;
}
