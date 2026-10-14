/** One audited cold cohort; complete worker peaks, with distinct role contracts. */
import { createHash } from 'node:crypto';
import { isDeepStrictEqual } from 'node:util';
import { clientMemoryEvidence } from './research-memory.mjs';

const hash = (bytes) => createHash('sha256').update(bytes).digest('hex');
const digest = (value) => /^[a-f0-9]{64}$/.test(value ?? '');
const check = (condition, message) => { if (!condition) throw new Error(message); };
const same = (left, right, message) => check(isDeepStrictEqual(left, right), message);
const median = (values) => {
  check(values.length > 0 && values.every((value) => Number.isFinite(value) && value >= 0), 'Invalid tier observation');
  const sorted = [...values].sort((a, b) => a - b), mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
};
const axes = {
  clientPeak: ['process.peak_rss', 1e6], clientCpu: ['client.cold_cpu', 1],
  preparationCpu: ['preparation.cold_cpu', 1], preparationPeak: ['preparation.peak_rss', 1e6],
  ttft: ['response.ttft', 1], decodeTps: ['response.decode_tps', 1],
  totalNetwork: ['network.all_link_stream_bytes', 1e6], full: ['request.0.full', 1],
};

export function privateBoundaryTierCohorts(spec, document, plan, read, helpers) {
  const summaryRaw = read(spec.summary), summary = JSON.parse(summaryRaw);
  check(hash(summaryRaw) === spec.summary_sha256 &&
    plan?.schema === 'pllm.private_boundary_tiers_plan.v1' && plan.tiny === false &&
    summary.schema === 'pllm.private_boundary_tiers.v1' && summary.complete === true &&
    summary.all_logits_active_kv_positions_selections_rng_and_outputs_match === true &&
    document.schema === 'pllm.benchmark_run.v1' && document.complete === true && digest(document.cohort),
  'Missing complete audited tier evidence');
  same(document.suite, plan.suite, 'Tier suite differs from frozen plan');
  const suite = document.suite, count = suite.sampling.processes;
  check(suite.schema === 'pllm.benchmark_suite.v1' && count >= 3 && count === summary.replicates &&
    suite.sampling.order === 'balanced' && suite.sampling.warmups === 0 && suite.sampling.repetitions === 1 &&
    suite.cases.length === 8 && document.results.length === count * suite.cases.length &&
    summary.clients === document.results.length && plan.preflight.length === suite.cases.length &&
    plan.preflight.every((cell) => cell.admission.admitted === true), 'Incomplete repeated tier cohort');
  for (const file of [spec.replay_file, spec.frozen_factory.file]) {
    const entry = Object.entries(plan.frozen_sources).find(([name]) => name === file || name.endsWith(`/${file}`));
    check(entry && hash(read(file)) === entry[1], 'Tier measurement source changed');
  }
  same(Object.keys(summary.groups).sort(), suite.cases.map((cell) => cell.name).sort(), 'Tier recipe set changed');
  const groups = new Map(), seen = new Set();
  let reference = null, states = null, contracts = null, checkedValues = 0, maximumSwap = 0;
  for (const result of document.results) {
    const e = result.execution, name = e.case.name;
    const selected = suite.cases.find((cell) => cell.name === name);
    same(e.case, selected, 'Tier case differs from frozen suite');
    check(result.schema_version === 'pllm.benchmark_result.v2' && result.status === 'completed' &&
      e.cohort === document.cohort && e.failure_code === null && e.admission.admitted === true &&
      e.oracle.outcome === 'passed' && e.isolation.fresh_process === true &&
      e.fixture.metadata.owned_client_cache === 'fresh' &&
      e.build.digest === selected.build.expected_digest && digest(e.build.digest) &&
      result.warmups === 0 && Number.isInteger(e.replicate) && e.replicate >= 0 && e.replicate < count,
    'Unchecked tier build, isolation or admission');
    same(e.sampling, suite.sampling, 'Tier sampling changed');
    const cell = `${name}/${e.replicate}`;
    check(!seen.has(cell), 'Duplicate tier replicate'); seen.add(cell);
    const parameters = selected.target.parameters, experiment = parameters.model.experiment;
    const report = e.details.report, run = report.runs[0], audit = e.details.response_state_audit;
    check(selected.target.kind === 'response' && parameters.state_audit === true &&
      parameters.scenario.kind === 'cold' && !parameters.concurrent_requests &&
      report.runs.length === 1 && report.warmup_runs.length === 0 &&
      report.checks.passed === true && Object.values(report.checks).every((value) => value === true) &&
      report.prepared_material_accounting.conserved === true && report.memory_guard.tripped === false &&
      report.configuration.provider_backend === 'native' && run.cold === true && run.warm === false &&
      run.tokens.input_tokens === 150 && run.tokens.output_tokens === 8 && run.max_output_tokens === 8 &&
      experiment.budget.requests === 1 && experiment.budget.max_input_tokens === 150 && experiment.budget.max_new_tokens === 8 &&
      audit.schema === 'pllm.response_state_audit.v1' &&
      audit.bindings.length === 1 && audit.bindings[0].failed === false &&
      audit.bindings[0].cryptographic_randomness_frozen === false && audit.bindings[0].selections.length === 8,
    'Tier response or complete-state audit failed');
    helpers.checkRun(run);
    same(audit.client_memory_owners, ['sdk', 'dashboard', 'native_wan_relay', 'audit', 'role_supervision'],
      'Tier client owner list changed');
    check(report.configuration.source_lock_digest === plan.source_lock_digest &&
      e.comparison.source_digest === report.configuration.source_lock_digest &&
      e.fixture.digest === e.comparison.source_digest && e.comparison.workload_digest === result.workload_digest &&
      Object.keys(e.comparison).length === 7 && Object.values(e.comparison).every(digest) &&
      digest(result.environment.digest) && digest(result.configuration_digest) && digest(e.suite_digest) &&
      digest(audit.bindings[0].numeric_body_fingerprint), 'Tier source or numeric commitment changed');
    const identity = { source: report.configuration.source_lock_digest, prompt: report.configuration.prompt_digest,
      body: audit.bindings[0].numeric_body_fingerprint, tokens: run.tokens,
      output: run.generation.output_text_digest, wan: report.configuration.wan_emulation_digest,
      sampling: report.configuration.sampling, build: e.build };
    same(identity, summary.identity, 'Tier summary identity changed');
    // Deliberately compare exact state and selected metric contracts across the
    // different role graphs. Ordinary privacy/body-coverage identities survive.
    const comparison = { identity, environment: result.environment.digest, suite: e.suite_digest,
      numeric: e.comparison.numeric_digest, state: e.comparison.state_digest,
      workload: e.comparison.workload_digest, output: e.comparison.output_digest };
    if (reference) same(comparison, reference, 'Tier source, environment or comparison changed');
    reference = comparison;
    if (states) same(audit.bindings, states, 'Tier logits, KV, positions, selections or RNG changed');
    states = audit.bindings;
    const records = new Map(result.metrics.map((metric) => [metric.id, metric]));
    check(records.size === result.metrics.length, 'Duplicate tier metric');
    const measurements = Object.fromEntries([...records].map(([id, metric]) => [id, metric.value]));
    const metricContracts = Object.values(axes).map(([id]) => {
      const { value, sample_count, unavailable_reason, ...contract } = records.get(id);
      check(Number.isFinite(value) && value >= 0 && sample_count === 1 && unavailable_reason === null,
        'Missing tier priority metric');
      return contract;
    });
    if (contracts) same(metricContracts, contracts, 'Tier metric owner, unit or window changed');
    contracts = metricContracts;
    const peak = measurements['process.peak_rss'], scenarioPeak = report.client_process_memory.after.lifetime_peak_rss_bytes;
    check(Number.isSafeInteger(peak) && peak >= scenarioPeak && scenarioPeak > 0 &&
      scenarioPeak === measurements['client.peak_rss'] &&
      records.get('process.peak_rss').window === 'process_birth_through_target' &&
      records.get('process.peak_rss').measurement_scope === 'process_lifetime_high_water' &&
      report.client_process_memory.scope.includes('client and in-process dashboard'), 'Whole-client tier owner missing');
    const wan = helpers.checkedWan(report.native_network_accounting.samples.runs[0].after.wan_emulation);
    check(report.wan_readiness.emulation.native_stream_rate_snapshots_checked === true &&
      wan.conditions_digest === identity.wan && wan.total_stream_bytes === measurements['network.all_link_stream_bytes'],
    'Tier stream measurement or rate scope changed');
    const swap = report.memory_guard.maximum_host_swap_growth_bytes;
    check(Number.isSafeInteger(swap) && swap >= 0, 'Unknown tier swap evidence');
    maximumSwap = Math.max(maximumSwap, swap);
    for (const selection of states[0].selections) {
      check(Number.isSafeInteger(selection.checked_values) && selection.checked_values > 0, 'Empty tier state check');
      checkedValues += selection.checked_values;
    }
    if (!groups.has(name)) groups.set(name, []);
    groups.get(name).push({ result, report, experiment, measurements, peak, wan });
  }
  check(checkedValues === summary.checked_values && maximumSwap === summary.maximum_observed_swap_growth_bytes,
    'Tier state or swap totals changed');
  const rows = [...groups].map(([name, samples]) => {
    const first = samples[0], recorded = summary.groups[name];
    check(samples.length === count, 'Missing tier recipe replicate');
    same(recorded.experiment, first.experiment, 'Tier summarized recipe changed');
    for (const sample of samples) {
      same(sample.experiment, first.experiment, 'Tier recipe changed across replicates');
      check(sample.result.configuration_digest === first.result.configuration_digest &&
        sample.report.experiment.pipeline_digest === first.report.experiment.pipeline_digest,
      'Tier configuration binding changed');
    }
    const maxima = Math.max(...samples.map((sample) => sample.peak));
    check(recorded.maximum_client_peak_bytes === maxima, 'Tier maximum hid a replicate');
    same(recorded.observed_tiers, Object.fromEntries([['low', 256 * 2 ** 20], ['medium', 2 * 2 ** 30], ['high', 16 * 2 ** 30]]
      .map(([tier, maximum]) => [tier, maxima <= maximum])), 'Tier budget qualification changed');
    same(Object.keys(recorded.medians).sort(), [...new Set([...Object.values(axes).map(([id]) => id),
      'client.peak_rss', 'network.covered_body_bytes', 'target.latency'])].sort(), 'Tier summary metric set changed');
    for (const [id, cached] of Object.entries(recorded.medians)) {
      const values = samples.map((sample) => sample.measurements[id]);
      same(cached, values.some((value) => value === null) ? null : median(values), 'Tier summary median changed');
    }
    const pipeline = first.experiment.pipeline;
    const placement = helpers.executionPlacement(pipeline, first.report.configuration.roles, first.report.client_body_placement);
    const optional = (select, divisor = 1) => {
      const observations = samples.map(select);
      return observations.some((value) => value == null) ? null : median(observations) / divisor;
    };
    const values = { ...Object.fromEntries(helpers.metrics.map(({ id }) => [id, null])),
      ...Object.fromEntries(Object.entries(axes).map(([axis, [id, divisor]]) => [axis, recorded.medians[id] / divisor])),
      requestTps: median(samples.map(({ report }) => 8 / report.runs[0].durations.full_seconds)),
      clientNetwork: median(samples.map(({ wan }) => wan.client_stream_bytes)) / 1e6,
      networkPerToken: recorded.medians['network.all_link_stream_bytes'] / 1e6 / 8,
      aggregateCpu: optional(({ report }) => report.process_cpu_accounting?.aggregate_cold_first_response_cpu_seconds),
      providerPeak: optional(({ report }) => helpers.providerPeak(report.configuration.roles, report.runs[0].processes)),
      inferencePeak: optional(({ report }) => report.runs[0].processes?.inference?.rss_peak_bytes, 1e6),
      totalPeak: optional(({ report }) => report.native_process_memory?.complete === true
        ? report.native_process_memory.sampled_total_peak_rss_bytes : null, 1e6),
      bodyOffload: placement.clientMacPercent === null ? null : 100 - placement.clientMacPercent };
    return { id: name, label: name, file: spec.frozen_factory.file, fileSha256: spec.frozen_factory.sha256,
      configurationDigest: first.result.configuration_digest, sourceLock: summary.identity.source,
      pipeline, experiment: first.experiment, reference: name === 'maintained' ? 'prepared' : null,
      paper: null, publicPrefixTokens: 0, placement, metrics: values, sampleCount: count,
      clientMemory: clientMemoryEvidence(samples.map((sample) => sample.peak), 'process_birth_through_target'),
      provenance: { report: spec.report, reportSha256: spec.sha256, plan: spec.plan, planSha256: spec.plan_sha256 },
      buildDigest: first.result.execution.build.digest, frozenSources: plan.frozen_sources,
      comparisonCommitments: first.result.execution.comparison,
      recordedCase: first.result.execution.case, samplingContract: suite.sampling,
      measurementContracts: first.result.metrics.map(({ value, ...contract }) => contract),
      replay: { originalDriver: spec.replay_file, originalDriverSha256: spec.replay_sha256,
        source: spec.report.split('/').at(-1).replace(/\.json$/, ''), case: name,
        callable: 'pllm.metrics.BenchmarkCase.from_spec', argument: 'recordedCase',
        requiresOriginalBuildAndLifecycle: true },
      scopedTimings: { completeTargetSeconds: recorded.medians['target.latency'] } };
  });
  const first = document.results[0].execution.details.report, pipeline = rows[0].pipeline;
  return [{ ...spec, baseline: 'maintained', model: pipeline.model.source,
    identity: { model: pipeline.model.source, source: pipeline.model, body: summary.identity.body,
      prompt: summary.identity.prompt, quantization: pipeline.components.quantization,
      inputs: 150, outputs: 8, cap: 8, warm: false, output: summary.identity.output,
      nonlinear: null, state: null, verification: null, sampling: first.runs[0].sampling,
      backend: 'native', wan: summary.identity.wan, links: null, wanBackend: 'native-shared-tcp-pacer', clientIsolation: true },
    measurement: { requestScope: 'single audited cold request', phase: 'cold', sampleCount: count,
      clientRssLabel: 'Client RAM · complete worker lifetime', clientCpuLabel: 'Client CPU · cold first response',
      preparationCpuLabel: 'Prep CPU · birth to cold first response',
      cpu: 'SDK, dashboard and relay share the client process; cold first-response window',
      rss: 'Largest complete worker lifetime peak across replicates qualifies each tier; idle suite launcher excluded',
      timing: 'Online TTFT and N−1 decode; exact logits/KV audit charged in every arm; provider startup excluded from request latency',
      traffic: 'Source-present all-link TCP streams from relay startup through response; physical wire and source/tokenizer distribution unknown' }, rows }];
}
