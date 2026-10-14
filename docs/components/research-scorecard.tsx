'use client';

import { useEffect, useId, useState, type CSSProperties } from 'react';
import data from '@/data/research/scorecard.json';
import { leaders, normalizedAxes, measuredSegments, groupedMetrics, memoryBudgetRows, defaultResearchCohort, scopedMetric } from '@/lib/research-rankings.mjs';
import { withBasePath } from '@/lib/paths.mjs';
import styles from './research-scorecard.module.css';
import { ResearchSotaTable } from './research-sota';

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
  const [cohortId, setCohort] = useState(() => defaultResearchCohort(
    data.cohorts.filter((cohort) => cohort.model === models[0]),
    data.clientMemoryTiers.find((tier) => tier.id === 'low')!.maxBytes,
  )!.id);
  const [memoryTier, setMemoryTier] = useState<string | null>('low');
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
  const measurement = 'measurement' in cohort ? cohort.measurement : null;
  const definitions = data.metrics.map((metric) => scopedMetric(metric, measurement));
  const groups = groupedMetrics(metricIds.map((key) => definitions.find((metric) => metric.id === key)!));
  const metrics = groups.flatMap((group) => group.metrics);
  const maxBytes = data.clientMemoryTiers.find((tier) => tier.id === memoryTier)?.maxBytes ?? null;
  const rows = memoryBudgetRows<Row>(cohort.rows, maxBytes).rows.filter((row) => Boolean(row.reference) ||
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
      <div><span className={styles.eyebrow}>The measured frontier</span><h2>A lighter client.<br />Practical private inference.</h2></div>
      <p>Our priority is a small, efficient client with affordable offline Preparation and responsive use over constrained links.
        The preferred deployment pairs Preparation under your control with one online Inference provider.</p>
    </div>
    <section className={styles.priorities} aria-labelledby={`${id}-priorities`}>
      <h3 id={`${id}-priorities`}>Eight objectives, competing costs</h3>
      <dl>{data.priorities.map((priority) => <div key={priority.label}>
        <dt><a href={withBasePath(`/research/problems/#${priority.anchor}`)}>{priority.label}</a></dt><dd>{priority.why}</dd>
      </div>)}</dl>
      <p>Quality and privacy are requirements for useful improvements. Compare matching numeric, workload and output contracts;
        inspect each deployment's trust assumptions. CPU and RAM measure resource cost, not price or energy.</p>
      <p>The historical chart has seven default axes: six resource/responsiveness metrics and measured all-link traffic totals.
        Base and output-dependent traffic need separate attribution; whole-response averages and decode-online counters do not establish that split.
        {' '}<a href={withBasePath('/research/problems/')}>Read the research questions, accounting definitions and measured tradeoffs →</a></p>
    </section>
    <div className={styles.controls}>
      <label htmlFor={`${id}-model`}>Model<select id={`${id}-model`} value={model} onChange={(event) => {
        setModel(event.target.value);
        setCohort(defaultResearchCohort(data.cohorts.filter((cohort) => cohort.model === event.target.value), maxBytes)!.id);
        setSelected(null);
      }}>
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
    <ResearchSotaTable key={cohort.id} cohort={cohort} memoryTier={memoryTier} onMemoryTierChange={setMemoryTier} />
    <details className={styles.metricPicker}><summary>Choose chart metrics <span>{metrics.length} selected · client, Preparation, responsiveness and all-link traffic</span></summary>
      <div>{groupedMetrics(definitions).map((group) => <fieldset key={group.label}><legend title={group.title}>{group.label}</legend>
        {group.metrics.map((metric) => <label key={metric.id}><input type="checkbox"
        checked={metricIds.includes(metric.id)} disabled={metricIds.length === 1 && metricIds.includes(metric.id)}
        onChange={(event) => setMetrics(event.target.checked ? [...metricIds, metric.id] : metricIds.filter((key) => key !== metric.id))} />
        {metric.label}</label>)}</fieldset>)}</div>
      <button type="button" onClick={() => setMetrics(data.defaultMetrics)}>Restore priority metrics</button>
    </details>
    <div className={styles.filters}>
      {cohort.rows.some((row) => row.publicPrefixTokens > 0) && <label><input type="checkbox" checked={publicPrefix}
        onChange={(event) => setPublicPrefix(event.target.checked)} /> Public-prefix configurations</label>}
      {cohort.rows.some((row) => row.placement.clientBody) && <label><input type="checkbox" checked={clientWeights}
         onChange={(event) => setClientWeights(event.target.checked)} /> Client-owned decoder candidates</label>}
       <span>Baselines within the RAM maximum stay visible. Chart includes all topologies.</span>
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
            {metricIds.includes('bodyOffload') && ' Body offload uses its absolute zero-to-one-hundred percentage of plan body-linear work delegated to providers. '}
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
          <text x={48} y={487} className={styles.chartNote}>↑ Higher is better · lower costs score higher · missing measurements leave gaps</text>
        </svg>
      </div>
      <figcaption>Every axis runs 0–100, higher is better. Latency, speed and costs normalize within the selected cohort; lower latency, CPU, memory and traffic earn higher scores.
        Hover or select for actual values. Missing Preparation measurements and topologies without that role leave gaps.
        {metricIds.includes('bodyOffload') && " Body offload uses the plan's absolute percentage of body-linear work assigned to providers; it excludes client attention, nonlinear and token-boundary work."}
        {' '}Axes do not form a composite ranking.</figcaption>
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
      <p>Matched observations, not confidence intervals. Native 100/40 cohorts cap each party's combined TCP traffic across peers.
        This is local bandwidth emulation; no Internet RTT is implied. Network counts each directed stream once, including HTTP/WebSocket framing,
        excluding kernel headers, retransmissions, telemetry and checkpoint distribution. Public artifacts are pre-positioned.</p>
      <p>Agg means aggregate across all roles. Client peaks require fresh processes. Aggregate RAM is the largest sampled sum of simultaneously live client/provider RSS, not summed lifetime peaks;
        it may double-count shared pages and miss short spikes. CPU includes startup and native relay overhead; GPU compute and uncharged GPU memory remain unmeasured.
        Historical shared-client runs leave incomparable memory fields empty. Client-only holds the full decoder; remote placements still keep attention and nonlinear work at the client.</p>
      {'measurement' in cohort && cohort.measurement ? <p>{cohort.measurement.requestScope}; median of {cohort.measurement.sampleCount} fresh cells.
        {' '}{cohort.measurement.cpu}. {cohort.measurement.rss}. {cohort.measurement.timing}. {cohort.measurement.traffic}.</p>
        : <p>Client CPU includes the benchmark/dashboard and native relay. Preparation CPU runs from that role's process birth through the first cold response;
        its RAM is the largest request-window telemetry sample, including the opening sample. TTFT measures online start to first output;
        decode throughput uses the later N−1 outputs. All-link traffic covers the recorded cold lifecycle across every directed role link.
        Whole-response MB/token includes that lifecycle's base cost; it is not incremental generation traffic.</p>}
      <p><a href={repository + cohort.report}>Canonical report ↗</a> · <a href={withBasePath('/research/scorecard/')}>Commands and methodology</a></p>
      <p>Report SHA-256: <code>{cohort.reportSha256}</code></p>
      {rows.map((row) => <details key={row.id}><summary>{row.label}</summary><p>Configuration: <code>{row.configurationDigest}</code><br />Python SHA-256: <code>{row.fileSha256}</code></p>
        <dl>{data.metrics.map((metric) => <div key={metric.id}><dt>{metric.label}</dt><dd>{number(row.metrics[metric.id as MetricKey])} {metric.unit}</dd></div>)}</dl>
        <pre>{JSON.stringify(row.pipeline, null, 2)}</pre></details>)}
    </details>
    <p className={styles.scope}><a href={withBasePath('/research/results/')}>All research data: confirmed results and every search trial →</a></p>
    <p className={styles.scope}>Model and condition selectors include only registered, matched evidence.</p>
  </section>;
}
