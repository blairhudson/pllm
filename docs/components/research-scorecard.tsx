'use client';

import { useId, useState } from 'react';
import data from '@/data/research/scorecard.json';
import { leaders, improvement } from '@/lib/research-rankings.mjs';
import { withBasePath } from '@/lib/paths.mjs';
import styles from './research-scorecard.module.css';

const repository = 'https://github.com/blairhudson/pllm/blob/main/';
const number = (value: number | null) => value === null ? 'Unmeasured' :
  value.toLocaleString('en', { maximumFractionDigits: value < 1 ? 4 : 2 });

export function ResearchScorecard() {
  const id = useId();
  const models = [...new Set(data.cohorts.map((cohort) => cohort.model))];
  const [model, setModel] = useState(models[0]);
  const [cohortId, setCohort] = useState(data.cohorts[0].id);
  const [metricId, setMetric] = useState('covered');
  const [publicPrefix, setPublicPrefix] = useState(true);
  const available = data.cohorts.filter((cohort) => cohort.model === model);
  const cohort = available.find((candidate) => candidate.id === cohortId) ?? available[0];
  const metric = data.metrics.find((candidate) => candidate.id === metricId)!;
  const key = metric.id as keyof typeof cohort.rows[number]['metrics'];
  const rows = cohort.rows.filter((row) => publicPrefix || row.publicPrefixTokens === 0);
  const baseline = rows.find((row) => row.id === cohort.baseline)!;
  const best = leaders(rows, metric);
  const max = Math.max(1, ...rows.map((row) => row.metrics[key] ?? 0));
  const baselineValue = baseline.metrics[key];

  return <section className={styles.scorecard} aria-label="Measured SDK performance">
    <div className={styles.heading}>
      <div><span className={styles.eyebrow}>The measured frontier</span>
        <h2>What works. What it costs.</h2></div>
      <p>SOTA here means the best <em>measured SDK composition</em> for each metric,
        model and matched workload. There is no single winner across every measure.</p>
    </div>
    <div className={styles.controls}>
      <label htmlFor={`${id}-model`}>Model<select id={`${id}-model`} value={model}
        onChange={(event) => setModel(event.target.value)}>
        {models.map((value) => <option key={value}>{value}</option>)}
      </select></label>
      <label htmlFor={`${id}-cohort`}>Matched experiment<select id={`${id}-cohort`} value={cohort.id}
        onChange={(event) => setCohort(event.target.value)}>
        {available.map((value) => <option key={value.id} value={value.id}>{value.title}</option>)}
      </select></label>
      <label htmlFor={`${id}-metric`}>Measure<select id={`${id}-metric`} value={metric.id}
        onChange={(event) => setMetric(event.target.value)}>
        {data.metrics.map((value) => <option key={value.id} value={value.id}>{value.label}</option>)}
      </select></label>
    </div>
    {cohort.rows.some((row) => row.publicPrefixTokens > 0) &&
      <label className={styles.toggle}><input type="checkbox" checked={publicPrefix}
        onChange={(event) => setPublicPrefix(event.target.checked)} /> Include explicitly public-prefix configurations</label>}
    <p className={styles.scope}>{cohort.scope}</p>
    <div className={styles.winner} aria-live="polite">
      <span>{metric.direction === 'min' ? 'Lower is better' : 'Higher is better'} · {metric.unit}</span>
      <strong>{best.length ? `Best measured: ${rows.filter((row) => best.includes(row.id)).map((row) => row.label).join(' / ')}`
        : 'No matched measurement available'}</strong>
    </div>
    {best.length > 0 && <figure className={styles.chart}>
      <svg viewBox={`0 0 760 ${rows.length * 66 + 35}`} role="img" aria-labelledby={`${id}-chart-title ${id}-chart-desc`}>
        <title id={`${id}-chart-title`}>{`${metric.label}: measured SDK configurations`}</title>
        <desc id={`${id}-chart-desc`}>{rows.map((row) => `${row.label}: ${number(row.metrics[key])} ${metric.unit}`).join('; ')}</desc>
        {rows.map((row, index) => {
          const value = row.metrics[key];
          return <g key={row.id} transform={`translate(0 ${index * 66 + 10})`}>
            <text x="0" y="17" className={styles.chartLabel}>{row.label}</text>
            <rect x="230" y="0" width="430" height="25" className={styles.track} rx="2" />
            {value !== null && <rect x="230" y="0" width={430 * value / max} height="25" rx="2"
              className={best.includes(row.id) ? styles.bestBar : styles.bar} />}
            <text x="755" y="17" textAnchor="end" className={styles.chartValue}>{number(value)}</text>
            {row.publicPrefixTokens > 0 && <text x="0" y="37" className={styles.chartNote}>{row.publicPrefixTokens} public prefix tokens</text>}
          </g>;
        })}
      </svg>
      <figcaption>Single-sample observations, not confidence intervals. Decimal MB. Unknown measurements remain empty.</figcaption>
    </figure>}
    <div className={styles.tableWrap} tabIndex={0} role="region" aria-label="Performance values and configurations">
      <table><caption>{metric.label} · matched baseline comparison</caption>
        <thead><tr><th>Configuration</th><th>{metric.unit}</th><th>vs baseline</th><th>Reproduce</th></tr></thead>
        <tbody>{rows.map((row) => {
          const value = row.metrics[key], gain = improvement(value, baselineValue, metric.direction);
          return <tr key={row.id}>
            <th scope="row">{row.label}{row.id === cohort.baseline && <small>Baseline</small>}
              {row.publicPrefixTokens > 0 && <small>{row.publicPrefixTokens} public tokens</small>}</th>
            <td>{number(value)}</td><td>{row.id === cohort.baseline ? '—' : gain === null ? 'Unavailable' :
              `${number(Math.abs(gain))}% ${gain >= 0 ? 'better' : 'worse'}`}</td>
            <td><a href={repository + row.file}>Python configuration ↗</a></td>
          </tr>;
        })}</tbody>
      </table>
    </div>
    <details className={styles.details}><summary>Scope, identities and every measured value</summary>
      <p>These local CPU runs do not establish Internet latency, operator independence or representative model quality.
        Client lifetime RSS is cumulative across candidates and cannot rank their peaks.
        Pre-positioned public artifact sizes are shown separately; checkpoint distribution and full physical wire are unmeasured.</p>
      <p><a href={repository + cohort.report}>Canonical report ↗</a> · <a href={withBasePath('/research/scorecard/')}>Commands and methodology</a></p>
      <p>Report SHA-256: <code>{cohort.reportSha256}</code></p>
      {rows.map((row) => <details key={row.id}><summary>{row.label}</summary>
        <p>Configuration: <code>{row.configurationDigest}</code><br />Python file SHA-256: <code>{row.fileSha256}</code></p>
        <dl>{data.metrics.map((item) => <div key={item.id}><dt>{item.label}</dt>
          <dd>{number(row.metrics[item.id as typeof key])} {item.unit}</dd></div>)}</dl>
        <pre>{JSON.stringify(row.pipeline, null, 2)}</pre>
      </details>)}
    </details>
    <p className={styles.scope}>Next model: Qwen3-4B. It will appear in the selector once a matched cohort is available.</p>
  </section>;
}
