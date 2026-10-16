/** Read-only publication views over original, independently audited cohorts. */
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { isDeepStrictEqual } from 'node:util';
import { fileURLToPath } from 'node:url';
import { adaptedCohorts } from '../docs/scripts/research-scorecard-adapters.mjs';
import { metrics, executionPlacement, providerPeak } from '../docs/scripts/research-scorecard.mjs';

const ROOT = fileURLToPath(new URL('../', import.meta.url));
const REGISTRY = 'docs/data/research/benchmark-cohorts.json';
const COMPARISON_CONTRACT = 'paper/comparison-contract.v1.json';
const OUTPUT = 'paper/artifacts/historical-results-v1.json';
const MARKDOWN = 'paper/HISTORICAL-RESULTS.md';
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const IDS = [
  'qwen25-private-boundary-tiers-150-8',
  'qwen25-private-boundary-decode-confirmation-150-8',
  'qwen25-private-boundary-decode-long-output-150-32',
  'qwen25-private-boundary-decode-long-output-150-64',
];
const TIER_RECIPES = {
  maintained: 'Maintained prepared', tiling: 'MLP tiling', digit8: 'Digit8 Preparation',
  'tiling-digit8': 'Tiling + digit8', 'tiling-digit8-placement': 'Tiling + digit8 + placement',
  'digit8-placement': 'Digit8 + placement', 'serial-digit8': 'Serial digit8 issuance',
  'coupled-boundary': 'Coupled boundary',
};
const DECODE_RECIPES = {
  maintained: 'Maintained prepared', 'coupled-serial-u32': 'Coupled serial/u32',
  'coupled-parallel-uniform_residues': 'Coupled parallel/residue',
};
const READER_SOURCES = [
  'scripts/audit_publication_evidence.mjs',
  'docs/scripts/research-scorecard.mjs',
  'docs/scripts/research-scorecard-adapters.mjs',
  'docs/scripts/research-private-boundary-tiers.mjs',
  'docs/scripts/research-private-boundary.mjs',
  'docs/scripts/research-memory.mjs',
  'python/pllm/search/client_memory_tiers.json',
];
const METRICS = {
  'network.all_link_stream_bytes': { unit: 'bytes', direction: 'min' },
  'network.covered_body_bytes': { unit: 'bytes', direction: 'min' },
  'request.0.full': { unit: 'seconds', direction: 'min' },
  'response.ttft': { unit: 'seconds', direction: 'min' },
  'response.decode_tps': { unit: 'tokens/second', direction: 'max' },
  'client.cold_cpu': { unit: 'CPU seconds', direction: 'min' },
  'preparation.cold_cpu': { unit: 'CPU seconds', direction: 'min' },
  'preparation.peak_rss': { unit: 'bytes', direction: 'min' },
  'process.peak_rss': { unit: 'bytes', direction: 'min' },
  'target.latency': { unit: 'seconds', direction: 'min' },
};
const median = values => {
  const sorted = [...values].sort((a, b) => a - b), mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
};
const requireValue = (condition, message) => { if (!condition) throw new Error(message); };
const same = (actual, expected, message) => requireValue(isDeepStrictEqual(actual, expected), message);
const sameNumber = (actual, expected, message) => requireValue(Number.isFinite(actual) && Number.isFinite(expected) &&
  Math.abs(actual - expected) <= Number.EPSILON * Math.max(1, Math.abs(actual), Math.abs(expected)) * 4, message);
const summarize = samples => {
  const complete = samples.every(value => value !== null);
  return { samples, median: complete ? median(samples) : null,
    minimum: complete ? Math.min(...samples) : null, maximum: complete ? Math.max(...samples) : null };
};
const ordered = (rows, ids, key) => {
  same(rows.map(row => row[key]).sort(), [...ids].sort(), `Publication ${key} membership changed`);
  return ids.map(id => rows.find(row => row[key] === id));
};
const number = (value, divisor = 1) => value === null ? 'unknown' : (value / divisor).toFixed(2);

// These are original SDK contracts, not units inferred from a publication label.
const RAW_METRICS = {
  'network.all_link_stream_bytes': ['bytes', 'all-links', 'relay_start_through_scenario', 'tcp_stream_payload'],
  'network.covered_body_bytes': ['bytes', 'all-links', 'setup_through_scenario', null],
  'request.0.full': ['seconds', 'client', 'request_setup_through_completion', 'ordinary_response_lifecycle'],
  'response.ttft': ['seconds', 'client', 'online_to_first_token', 'target'],
  'response.decode_tps': ['throughput', 'client', 'after_first_output', 'target'],
  'client.cold_cpu': ['cpu', 'client', 'benchmark_start_through_first_response', 'process_cpu'],
  'preparation.cold_cpu': ['cpu', 'preparation', 'role_birth_through_first_response', 'process_cpu'],
  'preparation.peak_rss': ['memory', 'preparation', 'request_windows', 'sampled_process_rss'],
  'process.peak_rss': ['memory', 'client', 'process_birth_through_target', 'process_lifetime_high_water'],
  'target.latency': ['seconds', 'client', 'execution', 'target'],
};

function checkedMetric(result, id, boundary, legacyBoundary) {
  const records = result.metrics.filter(record => record.id === id);
  requireValue(records.length === 1, `Missing publication metric: ${id}`);
  const { value, ...metadata } = records[0], report = result.execution.details.report, run = report.runs[0];
  const [kind, role, window, scope] = RAW_METRICS[id];
  const unknown = id === 'network.covered_body_bytes' && legacyBoundary;
  const phase = id.startsWith('response.') ? 'online' : 'full', statistic = kind === 'bytes' ? 'sum' : 'median';
  const [component, unit] = {
    seconds: ['pllm/latency', 'seconds'], cpu: ['pllm/cpu-time/v1', 'seconds'],
    memory: ['pllm/memory', 'bytes'], bytes: ['pllm/communication', 'bytes'], throughput: ['pllm/throughput', 'per_second'],
  }[kind];
  const coverage = legacyBoundary ? 'mixed_or_unknown' : boundary
    ? 'prepared_body_and_private_boundary_lifecycle_v4' : 'prepared_protocol_lifecycle_controls_and_correction_acks_v3';
  same(metadata, { id, component, parameters: {
    seconds: { statistic, phase }, cpu: {}, memory: { kind: 'peak_rss' },
    bytes: { direction: 'total', phase }, throughput: { basis: 'tokens' },
  }[kind], unit, unit_detail: kind === 'throughput' ? 'tokens_per_second' : null,
  statistic, phase, role, origin: unknown ? 'not_available' : 'measured',
  unavailable_reason: unknown ? 'measurement unavailable' : null, sample_count: unknown ? 0 : 1,
  window, measurement_scope: scope ?? `covered_application_bodies:${coverage}`,
  denominator: kind === 'throughput' ? { unit: 'generated_tokens', value: run.tokens.output_tokens - 1 } : null,
  owners: [role] }, `Publication raw metric contract changed: ${id}`);
  requireValue(unknown ? value === null : typeof value === 'number' && Number.isFinite(value) && value >= 0 &&
    (unit !== 'bytes' || Number.isSafeInteger(value)), `Unknown publication observation: ${id}`);
  if (unknown) return value;
  // Finalized worker high-water is itself a raw observation; all other values
  // must agree with their original timing, CPU, process or network owners.
  const observations = {
    'client.cold_cpu': () => report.process_cpu_accounting.cold_first_response_cpu_seconds_by_role.client,
    'preparation.cold_cpu': () => report.process_cpu_accounting.cold_first_response_cpu_seconds_by_role.preparation,
    'preparation.peak_rss': () => run.processes.preparation.rss_peak_bytes,
    'request.0.full': () => run.durations.full_seconds,
    'response.ttft': () => run.durations.ttft_seconds,
    'response.decode_tps': () => (run.tokens.output_tokens - 1) / run.durations.generation_seconds,
    'network.all_link_stream_bytes': () => report.native_network_accounting.samples.runs[0].after.wan_emulation.total_stream_bytes,
    'network.covered_body_bytes': () => report.topology_accounting.startup.all_link_serialized_body_bytes +
      report.topology_accounting.runs[0].all_link_serialized_body_bytes,
    'target.latency': () => result.execution.details.timings.wall_seconds[0],
  };
  if (observations[id]) sameNumber(value, observations[id](), `Publication metric differs from raw owner: ${id}`);
  return value;
}

function checkedResources(executions, boundary) {
  const roles = [...(boundary ? ['boundary_a', 'boundary_b'] : []), 'client', 'inference', 'preparation'];
  const providers = roles.filter(role => role !== 'client'), cpuSamples = [], peakSamples = [], aggregate = [];
  for (const result of executions) {
    const report = result.execution.details.report, record = report.process_cpu_accounting;
    same([...report.configuration.roles].sort(), roles, 'Publication topology CPU owners changed');
    requireValue(record?.schema === 'pllm.topology_process_cpu.v1' && record.scope ===
      'dashboard/client CPU from benchmark startup and role CPU from process birth; through the first cold measured response',
    'Publication cold CPU window changed');
    const cpu = record.cold_first_response_cpu_seconds_by_role;
    same(Object.keys(cpu ?? {}).sort(), roles, 'Publication cold CPU owner missing or unexpected');
    requireValue(Object.values(cpu).every(value => typeof value === 'number' && Number.isFinite(value) && value >= 0),
      'Publication cold CPU observation unavailable');
    const total = roles.reduce((sum, role) => sum + cpu[role], 0);
    sameNumber(record.aggregate_cold_first_response_cpu_seconds, total, 'Publication all-role CPU sum differs');
    cpuSamples.push(cpu); aggregate.push(total);
    peakSamples.push(Object.fromEntries(providers.map(role => {
      const value = report.runs[0].processes?.[role]?.rss_peak_bytes ?? null;
      requireValue(value === null || Number.isSafeInteger(value) && value >= 0, 'Publication provider peak is not integer bytes');
      return [role, value];
    })));
  }
  return {
    coldCpuSeconds: { unit: 'CPU seconds', owners: roles, cleanupIncluded: false,
      clientWindow: 'benchmark_start_through_first_response', providerWindow: 'role_birth_through_first_response',
      samplesByRole: Object.fromEntries(roles.map(role => [role, cpuSamples.map(sample => sample[role])])),
      aggregate: summarize(aggregate) },
    providerPeakBytes: { unit: 'bytes', owners: providers, window: 'request_windows', measurementScope: 'sampled_process_rss',
      statistic: 'largest provider sampled peak per replicate; not a simultaneous sum or lifetime high-water',
      samplesByRole: Object.fromEntries(providers.map(role => [role, peakSamples.map(sample => sample[role])])),
      ...summarize(peakSamples.map(sample => Object.values(sample).includes(null) ? null : Math.max(...Object.values(sample)))) },
  };
}

function checkedSensitivity(contract) {
  requireValue(contract.schema === 'pllm.publication_comparison_contract.v1' &&
    contract.kind === 'publication_metadata_not_sdk_input', 'Unknown publication comparison contract');
  const memory = contract.memory, tiers = ['low', 'medium', 'high'];
  requireValue(memory.nested === true && memory.comparison === 'inclusive_unrounded_bytes' &&
    memory.qualification_statistic === 'maximum_complete_client_peak_across_all_accepted_replicates_in_claimed_scope' &&
    memory.ordinary_native_qualification_metric === 'process.peak_rss', 'Publication memory qualification contract changed');
  same(Object.keys(memory.tiers_bytes).sort(), [...tiers].sort(), 'Publication memory tiers changed');
  same(Object.keys(memory.sensitivity_bytes).sort(), [...tiers].sort(), 'Publication sensitivity tiers changed');
  requireValue(Array.isArray(memory.sensitivity_multipliers) && memory.sensitivity_multipliers.includes(1) &&
    memory.sensitivity_multipliers.every((value, index, values) => Number.isFinite(value) && value > 0 &&
      (index === 0 || value > values[index - 1])), 'Invalid publication sensitivity multipliers');
  for (const [index, tier] of tiers.entries()) {
    const ceiling = memory.tiers_bytes[tier], grid = memory.sensitivity_bytes[tier];
    requireValue(Number.isSafeInteger(ceiling) && ceiling > (index === 0 ? 0 : memory.tiers_bytes[tiers[index - 1]]) &&
      Array.isArray(grid) && grid.every(value => Number.isSafeInteger(value) && value > 0), 'Invalid publication sensitivity bytes');
    same(grid, memory.sensitivity_multipliers.map(value => ceiling * value), 'Publication sensitivity grid differs from contract multipliers');
  }
  return { tiersBytes: memory.tiers_bytes, normativeCeilingsBytes: memory.sensitivity_bytes,
    diagnosticCeilingsBytes: [128, 384, 512, 1024].map(mib => mib * 2 ** 20),
    scope: 'reclassification of observed maxima only; no new execution or admission measurements' };
}

export function buildPublicationEvidence(readFile = file => fs.readFileSync(path.join(ROOT, file))) {
  const inputs = new Map();
  const read = file => {
    requireValue(!path.isAbsolute(file) && !file.split('/').includes('..'), 'Non-repository evidence path');
    const raw = readFile(file);
    const digest = sha256(raw);
    requireValue(!inputs.has(file) || inputs.get(file) === digest, 'Evidence changed during audit');
    inputs.set(file, digest);
    return raw;
  };
  const registry = JSON.parse(read(REGISTRY));
  const memorySensitivity = checkedSensitivity(JSON.parse(read(COMPARISON_CONTRACT)));
  READER_SOURCES.forEach(read);
  const cohorts = IDS.map(id => {
    const specs = registry.adapted_cohorts.filter(spec => spec.id === id);
    requireValue(specs.length === 1, 'Publication cohort must resolve exactly once');
    const spec = specs[0];
    const checked = adaptedCohorts(spec, read, { metrics, executionPlacement, providerPeak });
    requireValue(checked.length === 1 && checked[0].id === id, 'Publication cohort expanded or changed');
    const cohort = checked[0], raw = JSON.parse(read(spec.report));
    const summary = JSON.parse(read(spec.summary));
    requireValue(summary.maximum_observed_swap_growth_bytes === 0, 'Publication cohort observed swap growth');
    const recipes = id === IDS[0] ? TIER_RECIPES : DECODE_RECIPES;
    const rows = ordered(cohort.rows, Object.keys(recipes), 'id').map(row => {
      const executions = raw.results.filter(result => result.execution.case.name === row.id)
        .sort((a, b) => a.execution.replicate - b.execution.replicate);
      requireValue(executions.length === row.sampleCount, 'Publication replicate count changed');
      same(executions.map(result => result.execution.replicate), [0, 1, 2], 'Publication paired replicate IDs changed');
      const boundary = row.pipeline.components.boundary?.component === 'pllm/private-lookup-full-head/v1';
      const resources = checkedResources(executions, boundary);
      const values = Object.fromEntries(Object.entries(METRICS).map(([metric, contract]) => {
        const samples = executions.map(result => checkedMetric(result, metric, boundary, boundary && id === IDS[0]));
        return [metric, { ...contract, ...summarize(samples) }];
      }));
      const maximum = values['process.peak_rss'].maximum;
      requireValue(maximum === row.clientMemory.maxPeakBytes &&
        row.clientMemory.scope === 'process_birth_through_target', 'Publication client owner changed');
      return { recipe: row.id, topology: row.placement.topology, buildDigest: row.buildDigest,
        configurationDigest: row.configurationDigest, sampleCount: row.sampleCount,
        replicateIds: executions.map(result => result.execution.replicate),
        maximumCompleteClientBytes: maximum, observedLow: maximum <= memorySensitivity.tiersBytes.low,
        metrics: values, aggregateColdCpuSeconds: resources.coldCpuSeconds.aggregate.median, ...resources,
        measurementContracts: row.measurementContracts };
    });
    const candidate = id === IDS[0] ? 'coupled-boundary' : 'coupled-parallel-uniform_residues';
    const comparisons = rows.filter(row => row.recipe !== candidate).map(control => {
      const selected = rows.find(row => row.recipe === candidate);
      return { control: control.recipe, candidate,
        relationship: control.topology === selected.topology ? 'same-topology ablation' : 'explicit cross-topology costs',
        metrics: Object.fromEntries(Object.entries(METRICS).map(([metric, { direction }]) => {
          const a = control.metrics[metric], b = selected.metrics[metric];
          const fractions = a.samples.map((value, index) => value > 0 && b.samples[index] !== null
            ? (direction === 'min' ? 1 - b.samples[index] / value : b.samples[index] / value - 1) : null);
          return [metric, { improvementOfMedians: a.median > 0 && b.median !== null
            ? (direction === 'min' ? 1 - b.median / a.median : b.median / a.median - 1) : null,
          pairedImprovementFractions: fractions }];
        })) };
    });
    return { id, evidenceTier: spec.evidenceTier, report: spec.report, reportSha256: spec.sha256,
      privateCohortCommitment: raw.cohort, identity: cohort.identity,
      checkedValues: summary.checked_values, maximumObservedSwapGrowthBytes: 0,
      measurement: cohort.measurement, scope: spec.scope, rows, comparisons,
      rssNearCeilingCoverage: Object.fromEntries(Object.entries(memorySensitivity.normativeCeilingsBytes).map(([tier, grid]) =>
        [tier, rows.some(row => row.maximumCompleteClientBytes >= grid[0] && row.maximumCompleteClientBytes <= grid.at(-1))
          ? 'observed maxima within sensitivity grid' : 'absent'])),
      observedRssSensitivity: [...new Set([...Object.values(memorySensitivity.normativeCeilingsBytes).flat(),
        ...memorySensitivity.diagnosticCeilingsBytes])].sort((a, b) => a - b).map(ceilingBytes => ({
        ceilingBytes, normativeTiers: Object.entries(memorySensitivity.normativeCeilingsBytes)
          .filter(([, grid]) => grid.includes(ceilingBytes)).map(([tier]) => tier),
        eligibleRecipes: rows.filter(row => row.maximumCompleteClientBytes <= ceilingBytes)
          .map(row => row.recipe),
      })) };
  });
  return { schema: 'pllm.publication_historical_results.v1', freshMeasurements: false,
    supportsNewSotaClaim: false, permitsCohortPooling: false,
    sourcePresent: true, physicalWireMeasured: false, internetLatencyMeasured: false,
    wholeClientScope: 'SDK, supervisor, dashboard, native WAN relay and audit through target finalization',
    comparisonContract: { file: COMPARISON_CONTRACT, sha256: inputs.get(COMPARISON_CONTRACT) }, memorySensitivity,
    inputs: Object.fromEntries([...inputs].sort(([a], [b]) => a.localeCompare(b))), cohorts };
}

export function renderPublicationEvidence(evidence) {
  requireValue(evidence.schema === 'pllm.publication_historical_results.v1' &&
    evidence.freshMeasurements === false && evidence.supportsNewSotaClaim === false &&
    evidence.permitsCohortPooling === false, 'Historical publication scope changed');
  const text = [
    '# Verified historical publication tables', '',
    'Generated by `node scripts/audit_publication_evidence.mjs --write`;',
    '`--write` regenerates the evidence JSON and this Markdown only; both modes validate the manuscript table strings.',
    '`--check` independently rereads the original reports, plans, source pins, and strict evidence adapters.',
    'These are retained historical cohorts, not a new publication campaign. Cohorts are never pooled.', '',
    'Each row has three fresh clients. Traffic is decimal MB of measured all-link TCP streams;',
    'RSS is the **largest complete-client lifetime peak**, in binary MiB. Other entries are medians.',
    'Request latency excludes provider startup; complete-target latency charges startup and cleanup.',
    'Aggregate cold CPU sums every role through the first response; CPU after that response, including cleanup, is excluded.',
    'Provider RSS retains raw integer-byte samples per role and per-replicate maxima, plus separate median and maximum statistics.',
    'Provider peaks are request-window samples, not lifetime high-water marks or simultaneous aggregate RSS.',
    'Native shared 100/40 Mbps pacing is local emulation, with source/tokenizer delivery excluded.',
    'The coupled method has two additional workers and a different non-collusion contract.',
    'Missing provider peaks remain unknown; these records cannot establish equal aggregate resources.', '',
    `Normative Low sensitivity grid: ${evidence.memorySensitivity.normativeCeilingsBytes.low.map(bytes => bytes / 2 ** 20).join('/')} MiB,`,
    `from \`${evidence.comparisonContract.file}\` (SHA-256 \`${evidence.comparisonContract.sha256}\`).`,
    'The 128/384 MiB rows are additional diagnostics. Medium/High normative grids are retained in JSON;',
    'their near-ceiling coverage is absent in these cohorts. Eligibility at those thresholds is only inherited from the same low-memory executions.', '',
  ];
  for (const cohort of evidence.cohorts) {
    text.push(`## ${cohort.id}`, '',
      `Original evidence tier: **${cohort.evidenceTier}**. Checked logit/KV values: ${cohort.checkedValues.toLocaleString('en-US')}.`,
      `Source: \`${cohort.report}\`; SHA-256 \`${cohort.reportSha256}\`.`, '',
      '| Recipe | TCP MB | Request s | Online TTFT s | Decode tok/s | Max client MiB | Complete target s |',
      '| --- | ---: | ---: | ---: | ---: | ---: | ---: |');
    for (const row of cohort.rows) {
      const med = id => row.metrics[id].median;
      text.push(`| ${row.recipe} | ${number(med('network.all_link_stream_bytes'), 1e6)} | ${number(med('request.0.full'))} | ${number(med('response.ttft'))} | ${number(med('response.decode_tps'))} | ${number(row.maximumCompleteClientBytes, 2 ** 20)} | ${number(med('target.latency'))} |`);
    }
    text.push('', '| Recipe | Client CPU s | Preparation CPU s | Preparation peak MiB | Aggregate cold CPU s |',
      '| --- | ---: | ---: | ---: | ---: |');
    for (const row of cohort.rows) text.push(`| ${row.recipe} | ${number(row.metrics['client.cold_cpu'].median)} | ${number(row.metrics['preparation.cold_cpu'].median)} | ${number(row.metrics['preparation.peak_rss'].median, 2 ** 20)} | ${number(row.aggregateColdCpuSeconds)} |`);
    text.push('', '| RSS ceiling (MiB) | Eligible recipes |', '| ---: | --- |');
    for (const sensitivity of cohort.observedRssSensitivity.filter(row => row.ceilingBytes <= 384 * 2 ** 20)) {
      text.push(`| ${sensitivity.ceilingBytes / 2 ** 20} | ${sensitivity.eligibleRecipes.join(', ') || 'none'} |`);
    }
    text.push('', 'Threshold sensitivity only reclassifies measured peaks; it does not rerun execution or admission.',
      'All accepted clients observed zero swap growth. Raw replicate observations, measurement',
      'contracts, build/configuration digests, and within-cohort paired changes remain in',
      '`artifacts/historical-results-v1.json`. Three replicates do not establish workload breadth.', '');
  }
  text.push('## Interpretation boundary', '',
    'The 24-client cohort compares seven prepared incumbents with the original serial/u32 coupled method.',
    'The later 150+8 confirmation compares only three selected finalists; 150+32 and 150+64 are separate screens.',
    'Later timings do not replace earlier timings, and no configuration borrows another cohort\'s samples.',
    'Observed Low eligibility does not establish conservative SDK admission, a causal RSS improvement,',
    'representative quality, cryptographic review, independent operators, or contemporary external SOTA.', '');
  return text.join('\n');
}

export function manuscriptTables(evidence) {
  const cohorts = ordered(evidence.cohorts, IDS, 'id');
  const cells = row => [number(row.metrics['network.all_link_stream_bytes'].median, 1e6),
    number(row.metrics['request.0.full'].median), number(row.metrics['response.decode_tps'].median),
    number(row.maximumCompleteClientBytes, 2 ** 20)].join(' | ');
  const first = ['| Recipe | All-link TCP MB | Request s | Decode tok/s | Maximum client MiB |',
    '| :--- | ---: | ---: | ---: | ---: |',
    ...ordered(cohorts[0].rows, Object.keys(TIER_RECIPES), 'recipe')
      .map(row => `| ${TIER_RECIPES[row.recipe]} | ${cells(row)} |`)];
  const second = ['| Inputs + outputs | Recipe | TCP MB | Request s | Decode tok/s | Max client MiB |',
    '| :--- | :--- | ---: | ---: | ---: | ---: |'];
  for (const cohort of cohorts.slice(1)) {
    for (const row of ordered(cohort.rows, Object.keys(DECODE_RECIPES), 'recipe').filter(row => row.recipe !== 'coupled-serial-u32')) {
      second.push(`| ${cohort.identity.inputs}+${cohort.identity.outputs} | ${DECODE_RECIPES[row.recipe]} | ${cells(row)} |`);
    }
  }
  return { 'paper/manuscript.md': first.join('\n'), 'paper/low-memory.md': second.join('\n') };
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2);
  requireValue(args.length === 1 && ['--check', '--write'].includes(args[0]),
    'Usage: node scripts/audit_publication_evidence.mjs --check|--write (--write regenerates evidence outputs; both modes validate manuscript tables)');
  const evidence = buildPublicationEvidence();
  const outputs = { [OUTPUT]: JSON.stringify(evidence, null, 2) + '\n',
    [MARKDOWN]: renderPublicationEvidence(evidence) };
  for (const [relative, table] of Object.entries(manuscriptTables(evidence))) {
    requireValue(fs.readFileSync(path.join(ROOT, relative), 'utf8').includes(table),
      `Canonical manuscript table differs from original measurements: ${relative}`);
  }
  for (const [relative, content] of Object.entries(outputs)) {
    const destination = path.join(ROOT, relative);
    if (args[0] === '--write') {
      fs.mkdirSync(path.dirname(destination), { recursive: true });
      const temporary = `${destination}.tmp-${process.pid}`;
      fs.writeFileSync(temporary, content, { flag: 'wx' });
      fs.renameSync(temporary, destination);
    } else requireValue(fs.readFileSync(destination, 'utf8') === content,
      `Publication evidence is stale: ${relative}`);
  }
  console.log(`Verified ${evidence.cohorts.length} separate historical cohorts; no new measurements or SOTA claim.`);
}
