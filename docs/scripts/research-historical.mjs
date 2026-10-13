/** Schema-specific historical adapters. No invented canonical observations. */
import { createHash } from 'node:crypto';
import { isDeepStrictEqual } from 'node:util';
import { clientMemoryEvidence } from './research-memory.mjs';

const digest = (bytes) => createHash('sha256').update(bytes).digest('hex');
const require = (condition, message) => { if (!condition) throw new Error(message); };
const finite = (value) => { require(Number.isFinite(value) && value >= 0, 'Invalid historical metric'); return value; };
const mb = (value) => value === null ? null : value / 1e6;
const median = (values) => {
  if (values.some((value) => value === null)) return null;
  const sorted = values.map(finite).sort((a, b) => a - b), n = sorted.length;
  require(n > 0, 'Missing historical observations');
  return (sorted[Math.floor((n - 1) / 2)] + sorted[Math.floor(n / 2)]) / 2;
};

export function conservedWan(wan) {
  require(wan?.schema === 'pllm.native_wan.v1' && wan.enforced === true && wan.conditions_digest,
    'Require enforced historical WAN');
  let total = 0, client = 0;
  for (const [edge, bytes] of Object.entries(wan.directed_stream_bytes)) {
    require(/^(client|preparation|inference)->(client|preparation|inference)$/.test(edge) &&
      Number.isSafeInteger(bytes) && bytes >= 0, 'Invalid historical directed stream');
    total += bytes;
    if (edge.split('->').includes('client')) client += bytes;
  }
  require(total === wan.total_stream_bytes && client === wan.client_stream_bytes, 'Historical WAN conservation failed');
  return { totalNetwork: total / 1e6, clientNetwork: client / 1e6 };
}

function observedMetric(record, id, unit, window, scope) {
  const values = record.metrics.filter((metric) => metric.id === id);
  require(values.length === 1, `Missing or duplicate historical metric: ${id}`);
  const value = values[0];
  require(value.unit === unit && value.window === window && value.measurement_scope === scope &&
    value.origin === 'measured', `Historical metric scope changed: ${id}`);
  if (value.value === null) {
    require(Boolean(value.unavailable_reason), 'Unknown metric needs reason');
    return null;
  }
  return finite(value.value);
}

function resultObservation(record, document) {
  const execution = record.execution, report = execution.details.report;
  require(record.schema_version === 'pllm.benchmark_result.v2' && record.status === 'completed' &&
    record.warmups === 0 && record.repetitions === 1 && execution.oracle?.outcome === 'passed' &&
    execution.isolation?.fresh_process === true && execution.cohort === document.cohort &&
    report.checks?.passed === true && report.runs?.length === 2 && !report.warmup_runs?.length,
  'Require complete fresh-client cold/repeat evidence');
  require(report.runs.every((run, index) => run.status === 'completed' && run.tokens.authoritative === true &&
    run.tokens.output_tokens > 0 && run.warm === Boolean(index) && run.generation?.output_text_digest),
  'Incomplete historical response identity');
  const parameters = execution.case.target.parameters;
  const pipeline = parameters.model.experiment.pipeline;
  const comparison = execution.comparison;
  require(['source', 'numeric', 'state', 'privacy', 'workload', 'measurement', 'output'].every(
    (key) => /^[a-f0-9]{64}$/.test(comparison?.[`${key}_digest`])), 'Incomplete historical comparison identity');
  require(execution.build.digest === execution.case.build.expected_digest &&
    record.model.source_lock_digest === comparison.source_digest &&
    report.configuration.source_lock_digest === comparison.source_digest,
  'Historical source/build lock differs');
  const wan = report.native_network_accounting.samples.runs.at(-1).after.wan_emulation;
  require(report.wan_readiness?.emulation?.native_stream_rate_snapshots_checked === true &&
    wan.conditions_digest === report.configuration.wan_emulation_digest, 'Unchecked historical WAN');
  const network = conservedWan(wan);
  const streams = observedMetric(record, 'network.all_link_stream_bytes', 'bytes', 'relay_start_through_scenario', 'tcp_stream_payload');
  require(streams === wan.total_stream_bytes, 'Historical stream metric differs from counters');
  const values = {
    clientCpu: observedMetric(record, 'client.cold_cpu', 'seconds', 'benchmark_start_through_first_response', 'process_cpu'),
    clientPeak: mb(observedMetric(record, 'client.peak_rss', 'bytes', 'process_birth_through_scenario', 'process_lifetime_high_water')),
    preparationCpu: observedMetric(record, 'preparation.cold_cpu', 'seconds', 'role_birth_through_first_response', 'process_cpu'),
    preparationPeak: mb(observedMetric(record, 'preparation.peak_rss', 'bytes', 'request_windows', 'sampled_process_rss')),
    ttft: null, decodeTps: null,
    ...network,
  };
  const clientPeakBytes = observedMetric(record, 'client.peak_rss', 'bytes', 'process_birth_through_scenario', 'process_lifetime_high_water');
  const identity = { ...comparison, build: execution.build.digest, environment: record.environment.digest,
    cohort: document.cohort, source: pipeline.model, quantization: pipeline.components.quantization,
    workload: report.runs.map((run) => ({ inputs: run.tokens.input_tokens, outputs: run.tokens.output_tokens,
      cap: run.max_output_tokens, warm: run.warm, sampling: run.sampling, output: run.generation.output_text_digest })),
    wan: wan.conditions_digest, lifecycle: 'fresh-client-cold-exact-repeat', window: 'benchmark-result-v2-response' };
  return { identity, pipeline, values, clientPeakBytes, configurationDigest: record.configuration_digest,
    selector: parameters.client_mlp ?? null, sourceLock: comparison.source_digest,
    target: execution.case.target, recordedCase: execution.case, report, replicate: execution.replicate,
    scopedTimings: { coldRepeatOnlineTtftSummary: observedMetric(record, 'response.ttft', 'seconds', 'online_to_first_token', 'target'),
      coldRepeatDecodeTpsSummary: observedMetric(record, 'response.decode_tps', 'per_second', 'after_first_output', 'target') } };
}

export function historicalRows(cohort, document, read, metricDefinitions, placement) {
  require(cohort.adapter === 'pllm.benchmark_run.v1' && cohort.adapter === document.schema && document.complete === true && Array.isArray(document.results),
    'Incomplete or unsupported historical schema');
  const raw = read(cohort.report);
  require(digest(raw) === cohort.sha256 && isDeepStrictEqual(JSON.parse(raw), document), 'Historical source identity changed');
  const factory = read(cohort.factory.file);
  require(digest(factory) === cohort.factory.sha256 && String(factory).includes(cohort.sha256), 'Historical factory identity changed');
  const observations = document.results.filter((record) => cohort.cases.includes(record.execution.case.name));
  const byCase = new Map(), identities = new Set();
  for (const record of observations) {
    const name = record.execution.case.name;
    const value = resultObservation(record, document);
    identities.add(JSON.stringify(value.identity));
    const records = byCase.get(name) ?? [];
    require(!records.some((other) => other.replicate === value.replicate), 'Duplicate historical replicate');
    records.push(value); byCase.set(name, records);
  }
  require(identities.size === 1, 'Historical cohort source/numeric/workload/build/window/output identities differ');
  require(byCase.size === cohort.cases.length && byCase.size >= 2, 'Missing historical cases');
  const rows = cohort.cases.map((name) => {
    const records = byCase.get(name), first = records[0];
    require(records.length === cohort.replicates && records.every((record) =>
      isDeepStrictEqual(record.target, first.target) && record.configurationDigest === first.configurationDigest),
    'Historical factory parameters or replicate count changed');
    const metrics = Object.fromEntries(metricDefinitions.map(({ id }) => [id, null]));
    for (const key of Object.keys(first.values)) metrics[key] = median(records.map((record) => record.values[key]));
    const graph = placement(first.pipeline, ['client', 'preparation', 'inference'], first.report?.client_body_placement);
    return { id: name, label: cohort.labels?.[name] ?? name, file: cohort.factory.file,
      fileSha256: cohort.factory.sha256, configurationDigest: first.configurationDigest, pipeline: first.pipeline,
      sourceLock: first.sourceLock, paper: null, papers: [], reference: null, publicPrefixTokens: 0, placement: graph,
      metrics, researchSelector: first.selector ? { selection: first.selector, pipeline_component: false } : null,
      clientMemory: clientMemoryEvidence(records.map((record) => record.clientPeakBytes), 'process_birth_through_scenario'),
      recordedCase: first.recordedCase, samplingContract: document.suite.sampling,
      scopedTimings: Object.fromEntries(Object.keys(first.scopedTimings).map((key) => [key, median(records.map((record) => record.scopedTimings[key]))])),
      factory: { source: cohort.report.replace(/^.*\//, '').replace(/\.json$/, ''), case: name, callable: 'case' },
      observations: records.map(({ replicate, values, scopedTimings }) => ({ replicate, metrics: values, ...(scopedTimings ? { scopedTimings } : {}) })) };
  });
  const identity = JSON.parse([...identities][0]);
  const totals = Object.fromEntries(['inputs', 'outputs', 'cap'].map((key) => [key,
    identity.workload.reduce((sum, request) => {
      require(Number.isSafeInteger(request[key]) && request[key] > 0, 'Invalid historical scenario token count');
      return sum + request[key];
    }, 0)]));
  return { ...cohort, model: rows[0].pipeline.model.source, rows, reportSha256: cohort.sha256,
    identity: { ...identity, ...totals, requestCount: identity.workload.length,
      tokenCountScope: 'cold + repeat scenario totals', warm: false, clientIsolation: true },
    measurement: { requestScope: 'cold + repeat scenario', phase: 'cold + repeat', sampleCount: cohort.replicates,
      clientRssLabel: 'Client RAM · scenario-lifetime (cold + repeat)', clientCpuLabel: 'Client CPU · cold first response',
      preparationCpuLabel: 'Prep CPU · birth to cold first response',
      cpu: 'Client benchmark-start through first cold response; Preparation birth through first cold response',
      rss: 'Client process-lifetime peak across cold + repeat; Preparation sampled request windows',
      timing: 'Cold/repeat summary timings stay separate scoped observations; ordinary per-request timing axes unknown',
      traffic: 'All-link TCP streams include setup + cold + repeat scenario' },
    unknownReasons: { baseNetwork: 'No measured fixed/output all-link decomposition', tokenNetwork: 'No measured fixed/output all-link decomposition',
       ttft: 'Cold/repeat summary, not selected single-request online TTFT', decodeTps: 'Cold/repeat summary, not selected single-request decode' } };
}
