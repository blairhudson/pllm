'use client';

import { useEffect, useId, useState, type CSSProperties } from 'react';
import data from '@/data/research/scorecard.json';
import { leaders, normalizedAxes, measuredSegments, groupedMetrics } from '@/lib/research-rankings.mjs';
import { withBasePath } from '@/lib/paths.mjs';
import styles from './research-scorecard.module.css';

const repository = 'https://github.com/blairhudson/pllm/blob/main/';
const number = (value: number | null) => value === null ? 'Unmeasured' :
  value.toLocaleString('en', { maximumFractionDigits: value < 1 ? 4 : 2 });
type Row = typeof data.cohorts[number]['rows'][number];
type MetricKey = keyof Row['metrics'];
const referenceNames = { client: 'Client-only baseline', offset: 'Naive offset baseline', prepared: 'Prepared baseline' };
const referenceName = (row: Row) => row.reference && row.paper ? `${row.label} baseline`
  : referenceNames[row.reference as keyof typeof referenceNames] ?? null;
const color = (row: Row) => row.reference === 'client' ? '#b66b08' : row.reference === 'offset' ? '#8961d8'
  : row.reference === 'prepared' ? '#748293' : `hsl(${[...row.id].reduce((sum, c) => sum * 31 + c.charCodeAt(0) >>> 0, 0) % 360} 52% 46%)`;
const paperAnchor = (paper: string) => `paper-${paper}`;

export function ResearchScorecard() {
  const id = useId();
  const models = [...new Set(data.cohorts.map((cohort) => cohort.model))];
  const [model, setModel] = useState(models[0]);
  const [cohortId, setCohort] = useState(data.cohorts[0].id);
  const [metricIds, setMetrics] = useState(data.defaultMetrics);
  const [publicPrefix, setPublicPrefix] = useState(true);
  const [clientWeights, setClientWeights] = useState(true);
  const [selected, setSelected] = useState<string | null>(null);
  const [hovered, setHovered] = useState<string | null>(null);
  const [paperId, setPaper] = useState('');
  useEffect(() => {
    const select = (event: Event) => {
      const paper = (event as CustomEvent<string>).detail;
      const adapted = (cohort: typeof data.cohorts[number]) => cohort.rows.some((row) =>
        row.papers.some((link) => link.id === paper && link.kind === 'adaptation'));
      const cohort = data.cohorts.find((item) => item.id === cohortId && adapted(item)) ?? data.cohorts.find(adapted);
      if (cohort) { setModel(cohort.model); setCohort(cohort.id); }
      setPaper(paper);
      setSelected(null);
    };
    window.addEventListener('pllm-paper-select', select);
    return () => window.removeEventListener('pllm-paper-select', select);
  }, [cohortId]);
  const available = data.cohorts.filter((cohort) => cohort.model === model);
  const cohort = available.find((candidate) => candidate.id === cohortId) ?? available[0];
  const groups = groupedMetrics(metricIds.map((key) => data.metrics.find((metric) => metric.id === key)!));
  const metrics = groups.flatMap((group) => group.metrics);
  const rows = cohort.rows.filter((row) => Boolean(row.reference) ||
    ((publicPrefix || row.publicPrefixTokens === 0) && (clientWeights || !row.placement.clientBody)));
  const axes = normalizedAxes(rows, metrics);
  const active = rows.find((row) => row.id === (hovered ?? selected));
  const paper = data.papers.find((item) => item.id === paperId);
  const related = rows.filter((row) => row.papers.some((link) => link.id === paperId));
  const width = Math.max(640, metrics.length * 112), top = 138, bottom = 426;
  const x = (index: number) => metrics.length === 1 ? width / 2 : 70 + index * (width - 140) / (metrics.length - 1);
  const y = (score: number) => bottom - (bottom - top) * score / 100;
  const ordered = [...rows].sort((a, b) => Number(a.id === active?.id) - Number(b.id === active?.id) ||
    Number(Boolean(a.reference)) - Number(Boolean(b.reference)));
  const isMuted = (row: Row) => active ? row.id !== active.id : Boolean(paperId && !related.some((item) => item.id === row.id) && !row.reference);
  const goToPaper = (paperId: string) => {
    window.dispatchEvent(new CustomEvent('pllm-paper-focus', { detail: paperId }));
    window.location.hash = paperAnchor(paperId);
  };

  return <section className={styles.scorecard} aria-label="Measured SDK performance">
    <div className={styles.heading}>
      <div><span className={styles.eyebrow}>The measured frontier</span><h2>One model.<br />Every trade-off.</h2></div>
      <p>Follow each experiment across the measures. Higher always means better.
        Body offload rewards moving linear work off the client. Distinct baselines show local execution and naive offloading.</p>
    </div>
    <div className={styles.controls}>
      <label htmlFor={`${id}-model`}>Model<select id={`${id}-model`} value={model} onChange={(event) => setModel(event.target.value)}>
        {models.map((value) => <option key={value}>{value}</option>)}
      </select></label>
      <label htmlFor={`${id}-cohort`}>Matched conditions<select id={`${id}-cohort`} value={cohort.id}
        onChange={(event) => { setCohort(event.target.value); setSelected(null); }}>
        {available.map((value) => <option key={value.id} value={value.id}>{value.title}</option>)}
      </select></label>
      <label htmlFor={`${id}-paper`}>Highlight a paper<select id={`${id}-paper`} value={paperId}
        onChange={(event) => { setPaper(event.target.value); setSelected(null); }}>
        <option value="">All experiments</option>
        {data.papers.map((value) => <option key={value.id} value={value.id}>{value.title}</option>)}
      </select></label>
    </div>
    <details className={styles.metricPicker}><summary>Choose metrics <span>{metrics.length} selected · speed, efficiency, CPU, memory and network</span></summary>
      <div>{groupedMetrics(data.metrics).map((group) => <fieldset key={group.label}><legend title={group.title}>{group.label}</legend>
        {group.metrics.map((metric) => <label key={metric.id}><input type="checkbox"
        checked={metricIds.includes(metric.id)} disabled={metricIds.length === 1 && metricIds.includes(metric.id)}
        onChange={(event) => setMetrics(event.target.checked ? [...metricIds, metric.id] : metricIds.filter((key) => key !== metric.id))} />
        {metric.label}</label>)}</fieldset>)}</div>
      <button type="button" onClick={() => setMetrics(data.defaultMetrics)}>Restore defaults</button>
    </details>
    <div className={styles.filters}>
      {cohort.rows.some((row) => row.publicPrefixTokens > 0) && <label><input type="checkbox" checked={publicPrefix}
        onChange={(event) => setPublicPrefix(event.target.checked)} /> Public-prefix configurations</label>}
      {cohort.rows.some((row) => row.placement.clientBody) && <label><input type="checkbox" checked={clientWeights}
         onChange={(event) => setClientWeights(event.target.checked)} /> Client-owned decoder candidates</label>}
      <span>Baselines stay visible.</span>
    </div>
    <p className={styles.scope}>{cohort.scope}</p>
    {paper && <div className={styles.paperNotice} aria-live="polite"><strong>{paper.title}</strong>
      <p>{related.length ? `${related.length} related experiment lines highlighted. Component attribution is not a full-system reproduction.`
        : `No matched whole-response score in this cohort. ${paper.status}.`}</p>
      <button type="button" onClick={() => goToPaper(paper.id)}>Scroll to paper ↓</button></div>}
    <figure className={styles.chart}>
      <div className={styles.chartScroll} tabIndex={0} role="region" aria-label="Normalized experiment chart">
        <svg viewBox={`0 0 ${width} 502`} style={{ minWidth: Math.max(640, metrics.length * 84) }} aria-labelledby={`${id}-title ${id}-desc`}>
          <title id={`${id}-title`}>SDK experiment frontier — every metric normalized higher is better</title>
          <desc id={`${id}-desc`}>Each line is one experiment. Costs and speed run from worst observed at zero to best at one hundred.
            Body offload uses its absolute zero-to-one-hundred percentage of plan body-linear work delegated to providers.
            Lower costs are inverted. Missing measurements leave gaps. Dashed amber is client-only; dotted purple is naive offset.</desc>
          {groups.map((group) => {
            const first = metrics.indexOf(group.metrics[0]), last = first + group.metrics.length - 1;
            const left = first === 0 ? 48 : (x(first - 1) + x(first)) / 2;
            const right = last === metrics.length - 1 ? width - 45 : (x(last) + x(last + 1)) / 2;
            return <g key={group.label} className={styles.metricGroup} aria-label={group.title}>
              <title>{group.title}</title>
              <text x={(left + right) / 2} y={22} textAnchor="middle">{group.label}</text>
              <line x1={left + 8} x2={right - 8} y1={34} y2={34} />
              {first > 0 && <line x1={left} x2={left} y1={44} y2={bottom + 34} className={styles.groupBoundary} />}
            </g>;
          })}
          {[0, 25, 50, 75, 100].map((score) => <g key={score}>
            <line x1={48} x2={width - 45} y1={y(score)} y2={y(score)} className={score === 100 ? styles.frontier : styles.grid} />
            <text x={36} y={y(score) + 4} textAnchor="end" className={styles.chartNote}>{score}</text>
          </g>)}
          {axes.map((axis, index) => <g key={axis.id}>
            <text x={x(index)} y={59} textAnchor="middle" className={styles.axisLabel}>{axis.shortLabel}</text>
            <text x={x(index)} y={78} textAnchor="middle" className={styles.chartNote}>{axis.id === 'bodyOffload' ? '% offloaded' : axis.direction === 'min' ? 'Efficiency score' : 'Performance score'}</text>
            <text x={x(index)} y={94} textAnchor="middle" className={styles.chartNote}>Higher is better</text>
            <text x={x(index)} y={top - 12} textAnchor="middle" className={styles.chartValue}>{axis.min === null ? 'Unmeasured' : '100'}</text>
            <line x1={x(index)} x2={x(index)} y1={top} y2={bottom} className={styles.axis} />
            <text x={x(index)} y={bottom + 24} textAnchor="middle" className={styles.chartNote}>{axis.min === null ? '—' : '0'}</text>
          </g>)}
          {ordered.map((row) => {
            const values = axes.map((axis) => axis.scores[row.id]);
            const dash = row.reference === 'client' ? '9 5' : row.reference === 'offset' ? '3 4' : undefined;
            return <g key={row.id} className={styles.experimentLine} style={{ opacity: isMuted(row) ? .12 : active || row.reference ? 1 : .58 }}
              onMouseEnter={() => setHovered(row.id)} onMouseLeave={() => setHovered(null)} onClick={() => setSelected(row.id)}>
              <title>{`${row.label} — ${axes.map((axis) => `${axis.shortLabel}: ${axis.scores[row.id] === null ? 'Unmeasured' : `${number(axis.scores[row.id])}/100 (${number(row.metrics[axis.id as MetricKey])} ${axis.unit})`}`).join('; ')}`}</title>
              {measuredSegments(values).map((segment, index) => {
                const points = segment.map((point) => `${x(point.index)},${y(point.value)}`).join(' ');
                return <g key={index}><polyline points={points} fill="none" stroke="transparent" strokeWidth={14} />
                  <polyline points={points} fill="none" stroke={color(row)} strokeDasharray={dash}
                    strokeWidth={row.id === active?.id ? 3.5 : row.reference ? 2.5 : 1.6} pointerEvents="none" /></g>;
              })}
              {values.map((value, index) => value === null ? null : <circle key={index} cx={x(index)} cy={y(value)} r={value === 100 ? 4.5 : 2.6}
                fill={color(row)} stroke={value === 100 ? 'var(--panel)' : 'none'} strokeWidth={1.5} />)}
            </g>;
          })}
          <text x={48} y={487} className={styles.chartNote}>↑ Higher is better · speed/costs use observed ranges · body offload uses absolute %</text>
        </svg>
      </div>
      <figcaption>Every axis runs 0–100, higher is better. Body offload is the plan's percentage of body-linear work assigned to providers: client-only scores 0%. It excludes client attention, nonlinear and token-boundary work. Speed and costs normalize by observed range; lower CPU, memory and network costs earn higher scores. Hover or select for actual values. Axes do not form a composite ranking.</figcaption>
    </figure>
    <div className={styles.legend} aria-label="Experiment lines">
      {rows.map((row) => <button type="button" key={row.id} aria-pressed={selected === row.id}
        className={isMuted(row) ? styles.muted : undefined} style={{ '--line-color': color(row) } as CSSProperties}
        onMouseEnter={() => setHovered(row.id)} onMouseLeave={() => setHovered(null)}
        onClick={() => setSelected(selected === row.id ? null : row.id)}>
        <span className={styles.swatch} data-reference={row.reference} /><span>{referenceName(row) ?? row.label}
          <small>{row.placement.topology} · {row.placement.kernel}{row.publicPrefixTokens ? ` · ${row.publicPrefixTokens} public tokens` : ''}</small></span>
      </button>)}
    </div>
    <div className={styles.selection} aria-live="polite">
      {active ? <><div><strong>{active.label}</strong><span>{active.placement.ownership}</span>
        <a href={repository + active.file}>Python configuration ↗</a></div>
        <dl>{metrics.map((metric) => <div key={metric.id}><dt>{metric.shortLabel}</dt><dd>{number(active.metrics[metric.id as MetricKey])} <small>{metric.unit}</small></dd></div>)}</dl>
        {active.papers.map((link) => <p key={link.id}><button type="button" onClick={() => goToPaper(link.id)}>
          {data.papers.find((item) => item.id === link.id)?.title} ↓</button> · {link.scope}</p>)}</>
        : <p>Select an experiment to keep its values and paper links open. Amber dashed: client-only baseline. Purple dotted: naive offset baseline.</p>}
    </div>
    <details className={styles.details}><summary>Every result, actual units and rerunnable configurations</summary>
      <div className={styles.tableWrap} tabIndex={0} role="region" aria-label="Performance values and configurations">
        <table><caption>All {rows.length} visible experiments · selected metrics · decimal MB</caption>
          <thead><tr><th rowSpan={2}>Experiment</th>{groups.map((group) => <th key={group.label} title={group.title} colSpan={group.metrics.length} scope="colgroup">{group.label}</th>)}<th rowSpan={2}>Reproduce / papers</th></tr>
            <tr>{metrics.map((metric) => <th key={metric.id}>{metric.shortLabel}<small>{metric.unit}</small></th>)}</tr></thead>
          <tbody>{rows.map((row) => <tr key={row.id}><th scope="row">{row.label}<small>{referenceName(row)}</small>
            <small>{row.placement.topology} · {row.placement.kernel}</small>
            {row.placement.clientMacPercent !== null && <small>{number(row.placement.clientMacPercent)}% body linear MACs at client</small>}</th>
            {metrics.map((metric) => <td key={metric.id} className={leaders(rows, metric).includes(row.id) ? styles.leadingValue : undefined}>
              {number(row.metrics[metric.id as MetricKey])}</td>)}
            <td><a href={repository + row.file}>Python ↗</a>{row.papers.map((link) => <div key={link.id}><button type="button" onClick={() => goToPaper(link.id)}>
              {data.papers.find((item) => item.id === link.id)?.title} ↓</button></div>)}</td></tr>)}</tbody>
        </table>
      </div>
    </details>
    <details className={styles.details}><summary>Measurement scope and evidence</summary>
      <p>Single-response observations, not confidence intervals. Native 100/40 cohorts cap each party's combined TCP traffic across peers.
        This is local bandwidth emulation; no Internet RTT is implied. Network counts each directed stream once, including HTTP/WebSocket framing,
        excluding kernel headers, retransmissions, telemetry and checkpoint distribution. Public artifacts are pre-positioned.</p>
      <p>Agg means aggregate across all roles. Client peaks require fresh processes. Aggregate RAM is the largest sampled sum of simultaneously live client/provider RSS, not summed lifetime peaks;
        it may double-count shared pages and miss short spikes. CPU includes startup and native relay overhead; GPU compute and uncharged GPU memory remain unmeasured.
        Historical shared-client runs leave incomparable memory fields empty. Client-only holds the full decoder; remote placements still keep attention and nonlinear work at the client.</p>
      <p><a href={repository + cohort.report}>Canonical report ↗</a> · <a href={withBasePath('/research/scorecard/')}>Commands and methodology</a></p>
      <p>Report SHA-256: <code>{cohort.reportSha256}</code></p>
      {rows.map((row) => <details key={row.id}><summary>{row.label}</summary><p>Configuration: <code>{row.configurationDigest}</code><br />Python SHA-256: <code>{row.fileSha256}</code></p>
        <dl>{data.metrics.map((metric) => <div key={metric.id}><dt>{metric.label}</dt><dd>{number(row.metrics[metric.id as MetricKey])} {metric.unit}</dd></div>)}</dl>
        <pre>{JSON.stringify(row.pipeline, null, 2)}</pre></details>)}
    </details>
    <p className={styles.scope}><a href={withBasePath('/research/results/')}>All research data: confirmed results and every search trial →</a></p>
    <p className={styles.scope}>Next model: Qwen3-4B. The model selector grows when matched evidence is available.</p>
  </section>;
}
