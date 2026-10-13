'use client';

import { useId, useState } from 'react';
import data from '@/data/research/scorecard.json';
import { componentCombination, defaultResearchCohort, memoryBudgetRows, sortMetricRows, sotaMetrics, sotaTable } from '@/lib/research-rankings.mjs';
import { withBasePath } from '@/lib/paths.mjs';
import styles from './research-sota.module.css';

type Cohort = typeof data.cohorts[number];
type Row = Cohort['rows'][number];
type MetricKey = keyof Row['metrics'];
type MetricSort = { id: string; direction: 'ascending' | 'descending' };
type BodyAttributionDisclosure = { replicate: number; status: string; window: string; scope: string; byteUnit: string;
  generatedOutputTokens: number; knownBaseBytes: number; knownOutputDependentBytes: number; residualBytes: number; totalBytes: number };
const repository = 'https://github.com/blairhudson/pllm/blob/main/';
const number = (value: number, digits = value < 1 ? 4 : 2) => value.toLocaleString('en', { maximumFractionDigits: digits });
const metricDefinitions = sotaMetrics.map((id) => data.metrics.find((metric) => metric.id === id)!);

function Combination({ row }: { row: Row }) {
  const components = componentCombination(row.pipeline);
  const selector = 'researchSelector' in row ? row.researchSelector : null;
  const bodyAttribution = 'applicationBodyAttribution' in row && Array.isArray(row.applicationBodyAttribution)
    ? row.applicationBodyAttribution as BodyAttributionDisclosure[] : [];
  return <details className={styles.combination}>
    <summary><span>Component combination</span><small>{components.map((component) => component.name).join(' + ')}</small></summary>
    <dl>{components.map((component) => <div key={component.slot}>
      <dt>{component.name} <span>({component.slot})</span></dt>
      <dd><code>{component.component}</code><pre>{JSON.stringify(component.params ?? {}, null, 2)}</pre></dd>
    </div>)}</dl>
    <p>Configuration digest <code>{row.configurationDigest}</code></p>
    <p>Source lock <code>{row.sourceLock}</code></p>
    {'provenance' in row && <p><a href={repository + row.provenance.report}>Original evidence ↗</a>
      {row.provenance.plan && <> · <a href={repository + row.provenance.plan}>Frozen plan ↗</a></>}
      <br />Evidence SHA-256 <code>{row.provenance.reportSha256}</code></p>}
    {'buildDigest' in row && <p>Installed build <code>{row.buildDigest}</code></p>}
    {'experimentBindings' in row && <details><summary>Original per-replicate Experiment bindings</summary>
      <pre>{JSON.stringify(row.experimentBindings, null, 2)}</pre></details>}
    {'recordedCase' in row && <details><summary>Recorded target + original BuildRef (not a current-build replay)</summary>
      <pre>{JSON.stringify(row.recordedCase, null, 2)}</pre></details>}
    {'experiment' in row && <details><summary>Recorded Experiment (original lifecycle driver required)</summary>
      <pre>{JSON.stringify(row.experiment, null, 2)}</pre></details>}
    {'replay' in row && <details><summary>Frozen loader contract</summary><pre>{JSON.stringify(row.replay, null, 2)}</pre>
      <p>Current driver source alone does not recreate the original installed build or lifecycle.</p></details>}
    {selector && <div><p>Historical research selector — not a Pipeline component:</p><pre>{JSON.stringify(selector, null, 2)}</pre></div>}
    {bodyAttribution.length > 0 && <details><summary>Application-body attribution · separate from TCP streams</summary>
      <p>{bodyAttribution[0].scope}. {bodyAttribution[0].window}. {bodyAttribution[0].byteUnit}.
        Partial known amounts are not complete base/per-token headline costs. Each replicate conserves independently; column medians are not added.</p>
      <table><caption>Measured application bodies only · known base + known output-dependent + residual = total</caption>
        <thead><tr><th>Replicate / status</th><th>Known base MB</th><th>Known output-dependent MB</th><th>Residual MB</th><th>Total MB</th><th>Actual outputs</th></tr></thead>
        <tbody>{bodyAttribution.map((sample) => <tr key={sample.replicate}>
          <th>{sample.replicate} · {sample.status}</th>
          {[sample.knownBaseBytes, sample.knownOutputDependentBytes, sample.residualBytes, sample.totalBytes].map((bytes, index) =>
            <td key={index} title={`${bytes} bytes`}>{number(bytes / 1e6, 6)}</td>)}
          <td>{sample.generatedOutputTokens}</td></tr>)}</tbody></table>
    </details>}
  </details>;
}

export function ResearchSotaTable({ cohort, memoryTier, onMemoryTierChange }: {
  cohort: Cohort; memoryTier: string | null; onMemoryTierChange: (tier: string | null) => void;
}) {
  const id = useId();
  const measurement = 'measurement' in cohort ? cohort.measurement : null;
  const metrics = metricDefinitions.map((metric) => {
    const label = metric.id === 'clientPeak' ? measurement?.clientRssLabel
      : metric.id === 'clientCpu' ? measurement?.clientCpuLabel
        : metric.id === 'preparationCpu' ? measurement?.preparationCpuLabel : null;
    return label ? { ...metric, shortLabel: label, label } : metric;
  });
  const topologies = [...new Set(cohort.rows.map((row) => row.placement.topology))];
  const [selected, setTopology] = useState('Prepared');
  const [sort, setSort] = useState<MetricSort | null>(null);
  const topology = topologies.includes(selected) ? selected : topologies[0];
  const tier = data.clientMemoryTiers.find((value) => value.id === memoryTier);
  const table = sotaTable<Row>(cohort, topology, metrics, tier?.maxBytes ?? null);
  const rows = sort ? sortMetricRows(table.rows, sort.id, sort.direction) : table.rows;
  const control = topology === 'Client-only';
  // Repetition is not confirmation; only an explicitly classified cohort earns SOTA.
  const evidenceTier = 'evidenceTier' in cohort ? cohort.evidenceTier : 'unclassified';
  const badge = control ? 'Control best' : evidenceTier === 'confirmation' ? 'Scoped SOTA'
    : evidenceTier === 'screen' ? 'Screen best' : 'Observed best';
  const scenario = measurement?.requestScope === 'cold + repeat scenario';
  const attributed = cohort.rows.some((row) => row.metrics.baseNetwork !== null || row.metrics.tokenNetwork !== null);
  const partialAttribution = cohort.rows.some((row) => 'applicationBodyAttribution' in row &&
    row.applicationBodyAttribution.some((sample) => sample.status === 'Partial'));

  return <section className={styles.sota} aria-labelledby={`${id}-title`}>
    <div className={styles.heading}>
      <div><h2 id={`${id}-title`}>Measured leaders by model</h2>
        <p>Each row is one measured combination. Highlighted cells lead within this model, matched cohort, topology and selected maximum RAM budget; ties share the highlight.
          {' '}Only explicitly classified confirmations receive scoped SOTA badges; screens and unclassified observations are not confirmation.
          {' '}No badge implies a cross-cohort or universal ranking. Click metric headers to sort; unknown values stay last.</p></div>
      <label htmlFor={`${id}-topology`}>Execution topology<select id={`${id}-topology`} value={topology}
        onChange={(event) => setTopology(event.target.value)}>{topologies.map((value) => <option key={value} value={value}>
          {value === 'Client-only' ? 'Client-only control' : value}</option>)}</select></label>
    </div>
    <div className={styles.budgets} role="group" aria-labelledby={`${id}-budgets`}>
      <div className={styles.budgetHeading}><h3 id={`${id}-budgets`}>Maximum client RAM</h3>
        <button type="button" aria-pressed={!tier} onClick={() => onMemoryTierChange(null)}>Show all evidence</button></div>
      <div className={styles.tiers}>{data.clientMemoryTiers.map((value) => {
        const count = memoryBudgetRows(cohort.rows.filter((row) => row.placement.topology === topology), value.maxBytes).rows.length;
        return <button type="button" key={value.id} aria-pressed={memoryTier === value.id}
          onClick={() => onMemoryTierChange(value.id)} data-memory-tier={value.id}>
          <span>{value.label}<strong>Max {value.limit}</strong></span>
          <small>{value.description}</small><small>{count} measured {count === 1 ? 'combination' : 'combinations'} in this cohort</small>
        </button>;
      })}</div>
      <p className={styles.note}>Choose a ceiling, even on a larger machine. Higher tiers include lower-RAM choices; High does not require a GPU.
        Eligibility uses the largest recorded client-process peak across all replicates. Unknown RAM is excluded from capped views.
        {' '}<a href={withBasePath('/sdk/run/deployment/#maximum-client-ram')}>Use the same maximum in the SDK →</a></p>
      <p className={styles.note} role="status">Selected cohort · {tier ? `${tier.label} · max ${tier.limit}: ${table.rows.length} eligible; ${table.overBudgetCount} over budget; ${table.unknownCount} with unknown RAM.`
        : `All ${table.rows.length} combinations, including ${table.unknownCount} with unknown RAM. No memory ceiling selected.`}
        {' '}Observed RSS qualifies only this workload and lifecycle; SDK admission separately prices allocations and live headroom. GPU memory and container-charged file cache need separate capacity checks.</p>
    </div>
    <p className={styles.conditions}><strong>{cohort.model}</strong> · {cohort.identity.inputs} input tokens · {cohort.identity.outputs} generated
      {' '}of {cohort.identity.cap} allowed · {scenario ? 'Cold + repeat scenario totals'
        : `${cohort.identity.warm ? 'Warm' : 'Cold'} response`}
      {measurement && <> · {measurement.requestScope} · median of {measurement.sampleCount} fresh cells</>}
      {' '}· <a href={repository + cohort.report}>Matched evidence ↗</a></p>
    <div className={styles.tableWrap} tabIndex={0} role="region" aria-label="SOTA metrics and component combinations">
      <table>
        <caption>Eight objectives + measured traffic total · {table.rows.length} {topology.toLowerCase()} combinations · {tier ? `max ${tier.limit}` : 'all RAM budgets'} · decimal MB</caption>
        <thead><tr><th scope="col">Measured combination</th>{metrics.map((metric) => {
          const direction = sort?.id === metric.id ? sort.direction : undefined;
          const nextDirection = direction ? (direction === 'ascending' ? 'descending' : 'ascending')
            : (metric.direction === 'min' ? 'ascending' : 'descending');
          return <th key={metric.id} scope="col" title={metric.label} aria-sort={direction}>
            <button type="button" className={styles.sortButton} aria-label={`${metric.shortLabel}: sort ${nextDirection}`}
              onClick={() => setSort({ id: metric.id, direction: nextDirection })}>
              {metric.shortLabel}<span className={styles.sortIcon} aria-hidden="true">{direction === 'ascending' ? '↑' : direction === 'descending' ? '↓' : '↕'}</span>
            </button>
            <small><a aria-label={`${metric.shortLabel} definition`}
              href={withBasePath(`/research/problems/#${data.priorities.find((priority) => priority.metrics.includes(metric.id))?.anchor ??
                (metric.id === 'baseNetwork' ? 'base-all-link-traffic' : metric.id === 'tokenNetwork' ? 'per-token-all-link-traffic' : 'traffic-accounting-base-plus-generation')}`)}>
              {metric.unit} · {metric.direction === 'min' ? '↓ lower' : '↑ higher'}</a></small>
            {metric.id === 'totalNetwork' && <small>Measured TCP-stream total</small>}
            {metric.id === 'baseNetwork' || metric.id === 'tokenNetwork' ? <small>Covered application bodies</small> : null}
          </th>;
        })}</tr></thead>
        <tbody>{rows.length === 0 && <tr><td colSpan={metrics.length + 1} className={styles.empty}>
          No measured combination qualifies for this maximum under these conditions. Choose a larger maximum or show all evidence.
        </td></tr>}{rows.map((row) => <tr key={row.id} data-configuration={row.id}>
          <th scope="row"><strong>{row.label}</strong>
            <span className={styles.placement}>{row.placement.kernel}
              {row.metrics.bodyOffload !== null && ` · ${number(row.metrics.bodyOffload)}% body-linear offload`}
              {row.publicPrefixTokens > 0 && ` · ${row.publicPrefixTokens} public prefix tokens`}</span>
            {row.clientMemory.maxPeakBytes !== null && <span className={styles.placement} data-memory-peak-bytes={row.clientMemory.maxPeakBytes}>
              Largest observed client peak: {number(row.clientMemory.maxPeakBytes / (1 << 20))} MiB · {row.clientMemory.sampleCount} {row.clientMemory.sampleCount === 1 ? 'sample' : 'samples'}</span>}
            {row.metrics.artifactBytes !== null && row.metrics.artifactBytes > 0 && <span className={styles.placement}>
              {number(row.metrics.artifactBytes)} MB pre-positioned artifacts</span>}
            <Combination row={row} />
            <a className={styles.replay} href={repository + row.file}>{'recordedCase' in row || 'experiment' in row
              ? 'Frozen historical loader ↗' : 'Exact Python configuration ↗'}</a>
            {'scopedTimings' in row && row.scopedTimings && <details><summary>Separate lifecycle timing (not ordinary axes)</summary>
              <dl>{Object.entries(row.scopedTimings).map(([key, value]) => <div key={key}><dt>{key}</dt>
                <dd>{typeof value === 'number' ? number(value, 4) : 'Unmeasured'}</dd></div>)}</dl></details>}
          </th>
          {metrics.map((metric) => {
            const value = row.metrics[metric.id as MetricKey];
            const leading = table.leaders[metric.id].includes(row.id);
            return <td key={metric.id} data-metric={metric.id} data-accounting-plane={metric.accountingPlane ?? undefined} data-leading={leading || undefined}
              className={leading ? styles.leading : undefined}>
              {value === null ? <span className={styles.unknown}>{metric.id.startsWith('preparation') && topology !== 'Prepared'
                ? 'No role' : metric.id === 'baseNetwork' || metric.id === 'tokenNetwork' ? 'Unattributed' : 'Unmeasured'}</span>
                : <><span className={styles.value} title={String(value)}>{number(value, table.precision[metric.id])}</span>
                  {leading && <span className={styles.badge}>{badge}{table.leaders[metric.id].length > 1 ? ' · tied' : ''}</span>}</>}
            </td>;
          })}
        </tr>)}</tbody>
      </table>
    </div>
    <p className={styles.note}>{attributed ? 'Base and per-token traffic count separately attributed covered application bodies.'
      : partialAttribution ? 'Complete base and per-token costs remain unknown. Component details show the measured portions and unattributed remainder.'
        : 'Base and per-token traffic are unattributed in these reports.'} The separate total counts measured all-link TCP streams;
       checkpoint/tokenizer distribution and physical wire overhead are outside it; cold client-bundle delivery remains charged.
       Whole-response averages do not establish incremental token cost.</p>
      <p className={styles.note}>{measurement
      ? `${measurement.cpu}. ${measurement.rss}. ${measurement.timing}. ${measurement.traffic}.`
      : 'Client CPU covers the cold lifecycle; Preparation RAM is a sampled peak; TTFT starts online; decode uses N−1 later outputs.'}
       Body-linear offload and public-prefix reuse are shown beside each combination. Metric columns retain their original summaries; RAM-budget eligibility uses the maximum peak shown beside each combination.</p>
  </section>;
}

/** Standalone view reuses the chart's checked data and table contract. */
export function ResearchSota() {
  const id = useId();
  const models = [...new Set(data.cohorts.map((cohort) => cohort.model))];
  const [model, setModel] = useState(models[0]);
  const [cohortId, setCohort] = useState(() => defaultResearchCohort(
    data.cohorts.filter((cohort) => cohort.model === models[0]),
    data.clientMemoryTiers.find((tier) => tier.id === 'low')!.maxBytes,
  )!.id);
  const [memoryTier, setMemoryTier] = useState<string | null>('low');
  const available = data.cohorts.filter((cohort) => cohort.model === model);
  const cohort = available.find((item) => item.id === cohortId) ?? available[0];
  return <div className={styles.explorer}>
    <div className={styles.controls}>
      <label htmlFor={`${id}-model`}>Model<select id={`${id}-model`} value={model} onChange={(event) => {
        setModel(event.target.value);
        setCohort(defaultResearchCohort(data.cohorts.filter((cohort) => cohort.model === event.target.value),
          data.clientMemoryTiers.find((tier) => tier.id === memoryTier)?.maxBytes ?? null)!.id);
      }}>
        {models.map((value) => <option key={value}>{value}</option>)}</select></label>
      <label htmlFor={`${id}-cohort`}>Matched conditions<select id={`${id}-cohort`} value={cohort.id}
        onChange={(event) => setCohort(event.target.value)}>{available.map((value) => <option key={value.id} value={value.id}>{value.title}</option>)}</select></label>
    </div>
    <p className={styles.note}>{cohort.scope}</p>
    <ResearchSotaTable key={cohort.id} cohort={cohort} memoryTier={memoryTier} onMemoryTierChange={setMemoryTier} />
  </div>;
}
