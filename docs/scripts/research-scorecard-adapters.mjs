/** Strict readers for original SDK evidence; never rewrite it as legacy comparisons. */
import { createHash } from 'node:crypto';
import { clientMemoryEvidence } from './research-memory.mjs';
import { isDeepStrictEqual } from 'node:util';
import { privateBoundaryCohorts } from './research-private-boundary.mjs';
import { privateBoundaryTierCohorts, privateBoundaryDecodeConfirmationCohorts, privateBoundaryByteDotCohorts } from './research-private-boundary-tiers.mjs';

const hash = (value) => createHash('sha256').update(value).digest('hex');
const digest = (value) => /^[a-f0-9]{64}$/.test(value ?? '');
const sameNumber = (a, b) => Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) <= Number.EPSILON * Math.max(1, Math.abs(a), Math.abs(b)) * 4;
const stable = (value) => JSON.stringify(value, (_, item) => item && typeof item === 'object' && !Array.isArray(item)
  ? Object.fromEntries(Object.entries(item).sort(([a], [b]) => a.localeCompare(b))) : item);
const requireEvidence = (condition, message) => { if (!condition) throw new Error(message); };
export function median(values) {
  requireEvidence(values.length > 0 && values.every((value) => Number.isFinite(value) && value >= 0), 'Invalid repeated metric');
  const sorted = [...values].sort((a, b) => a - b), middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}
const aggregate = (samples, keys) => Object.fromEntries(keys.map((key) => [key,
  samples.some((sample) => sample[key] == null) ? null : median(samples.map((sample) => sample[key]))]));
const empty = (helpers) => Object.fromEntries(helpers.metrics.map(({ id }) => [id, null]));

function readBound(spec, read) {
  const raw = read(spec.report);
  requireEvidence(hash(raw) === spec.sha256, `Scorecard evidence changed: ${spec.report}`);
  const planRaw = spec.plan ? read(spec.plan) : null;
  if (planRaw) requireEvidence(hash(planRaw) === spec.plan_sha256, 'Scorecard plan changed');
  requireEvidence(hash(read(spec.replay_file)) === spec.replay_sha256, 'Scorecard replay source changed');
  if (spec.frozen_factory) requireEvidence(hash(read(spec.frozen_factory.file)) === spec.frozen_factory.sha256, 'Frozen replay factory changed');
  return { document: JSON.parse(raw), plan: planRaw ? JSON.parse(planRaw) : null };
}

function checkedWan(wan) {
  requireEvidence(wan?.schema === 'pllm.native_wan.v1' && wan.enforced === true && digest(wan.conditions_digest), 'Unchecked native WAN scope');
  const edges = Object.entries(wan.directed_stream_bytes);
  requireEvidence(edges.length > 0 && edges.every(([edge, value]) => /^[a-z_]+->[a-z_]+$/.test(edge) && Number.isSafeInteger(value) && value >= 0), 'Invalid directed stream counter');
  requireEvidence(edges.reduce((sum, [, value]) => sum + value, 0) === wan.total_stream_bytes &&
    edges.filter(([edge]) => edge.split('->').includes('client')).reduce((sum, [, value]) => sum + value, 0) === wan.client_stream_bytes,
  'Native stream conservation failed');
  return wan;
}

function checkedLedger(ledger) {
  requireEvidence(['pllm.topology_body_accounting.v1', 'pllm.topology_body_accounting.v2', 'pllm.topology_body_accounting.v3'].includes(ledger?.schema) &&
    ledger.tracked_body_counter_set_present === true, 'Unknown covered body scope');
  if (!ledger.schema.endsWith('.v1')) requireEvidence(ledger.control_body_counter_set_present === true, 'Partial lifecycle control counters');
  if (ledger.schema.endsWith('.v3')) requireEvidence(ledger.correction_ack_counter_set_present === true &&
    ledger.body_counter_coverage === 'prepared_protocol_lifecycle_controls_and_correction_acks_v3', 'Partial correction ACK coverage');
  requireEvidence(ledger.body_bytes_by_edge.every((edge) => Number.isSafeInteger(edge.serialized_body_bytes) && edge.serialized_body_bytes >= 0) &&
    ledger.body_bytes_by_edge.reduce((sum, edge) => sum + edge.serialized_body_bytes, 0) === ledger.all_link_serialized_body_bytes,
  'Covered body conservation failed');
  return ledger;
}

function checkRun(run) {
  requireEvidence(run.status === 'completed' && run.tokens?.authoritative === true &&
    Number.isSafeInteger(run.tokens.input_tokens) && run.tokens.input_tokens > 0 &&
    Number.isSafeInteger(run.tokens.output_tokens) && run.tokens.output_tokens > 0 &&
    run.tokens.output_tokens <= run.max_output_tokens && digest(run.model_fingerprint) &&
    digest(run.generation?.output_text_digest) && ['completed', 'incomplete'].includes(run.generation.response_status) &&
    (run.generation.response_status !== 'incomplete' || run.tokens.output_tokens === run.max_output_tokens),
  'Incomplete authoritative response');
  const audit = run.privacy;
  requireEvidence(audit?.plaintext_prompt_bytes_sent === 0 && audit.plaintext_token_ids_sent === 0 &&
    audit.preparation_requests_during_online === 0 &&
    audit.prepared_stage_rows_reserved === audit.prepared_stage_rows_claimed + audit.prepared_stage_rows_burned,
  'Response privacy or material lifecycle failed');
}

function planCase(plan, name) {
  if (plan.suite) return plan.suite.cases.find((item) => item.name === name);
  const study = plan.study;
  requireEvidence(study?.schema === 'pllm.frontier_study.v1' && plan.coverage?.status === 'ready', 'Unknown confirmation plan');
  const base = study.suite.cases.find((item) => name === item.name || name.startsWith(`${item.name}/`));
  requireEvidence(base, 'Case missing from frozen study');
  const result = structuredClone(base), suffix = name.slice(base.name.length + 1);
  if (name !== base.name) {
    requireEvidence(plan.coverage.nominees.includes(name), 'Case not nominated by confirmation plan');
    const variants = suffix.split('+');
    if (variants.length > 1) requireEvidence(study.interactions.some((item) => isDeepStrictEqual(item, variants)), 'Unplanned interaction');
    for (const variant of variants) {
      const selected = study.variants.find((item) => item.name === variant);
      requireEvidence(selected, 'Unknown study variant');
      for (const [key, value] of Object.entries(selected.parameters)) {
        if (key.startsWith('pipeline__')) {
          const parts = key.split('__').slice(1), components = result.target.parameters.model.experiment.pipeline.components;
          let owner = components;
          for (let index = 0; index < parts.length - 1; index++) owner = index === 0 ? owner[parts[index]].params : owner[parts[index]];
          owner[parts.at(-1)] = value;
        } else result.target.parameters[key] = value;
      }
    }
  }
  // PreparedInventory omits these two public defaults in its serialized identity.
  const inventory = result.target.parameters.model.experiment.pipeline.components.inventory?.params;
  if (inventory?.stage_window === 1) delete inventory.stage_window;
  if (inventory?.stage_window_bytes === (16 << 20)) delete inventory.stage_window_bytes;
  result.name = name;
  return result;
}

function runIdentity(report, pipeline, run, index) {
  const ledger = report.topology_accounting.runs[index];
  return { model: run.model_id, source: pipeline.model, body: run.model_fingerprint,
    prompt: report.configuration.prompt_digest, quantization: pipeline.components.quantization,
    nonlinear: pipeline.components.nonlinear ?? null, state: pipeline.components.state ?? null,
    verification: pipeline.components.verification ?? null, inputs: run.tokens.input_tokens,
    outputs: run.tokens.output_tokens, cap: run.max_output_tokens, sampling: run.sampling,
    warm: run.warm, cold: run.cold, requestIndex: index, output: run.generation.output_text_digest,
    backend: report.configuration.provider_backend, wan: report.configuration.wan_emulation_digest,
    links: report.configuration.link_conditions_digest ?? null, wanBackend: 'native-shared-tcp-pacer', clientIsolation: true,
    bodyCoverage: ledger.body_counter_coverage ?? ledger.schema, bodyWindow: ledger.measurement_window ?? ledger.scope };
}

/** Bind cached rates to the selected authoritative request, not a scenario summary. */
export function checkedCommunicationWindow(report, index) {
  const communication = report.communication_per_token;
  requireEvidence(communication?.schema === 'pllm.communication_per_generated_token.v1' &&
    communication.byte_unit === 'decimal MB (1000000 bytes)' &&
    communication.denominator === 'authoritative generated output tokens, not input tokens or output cap' &&
    communication.scope === 'covered application bodies; no full wire or checkpoint download' &&
    communication.decode_scope === 'online bodies after first output through completion, divided by N-1' &&
    Array.isArray(communication.runs) && communication.runs.length === report.runs.length &&
    report.topology_accounting.runs.length === report.runs.length &&
    Number.isInteger(index) && index >= 0 && index < report.runs.length,
  'Communication sample, denominator or window contract changed');
  const run = report.runs[index], comm = communication.runs[index];
  const ledger = checkedLedger(report.topology_accounting.runs[index]);
  const output = run.tokens.output_tokens, online = ledger.online_all_link_serialized_body_bytes ?? null;
  requireEvidence(run.status === 'completed' && run.tokens.authoritative === true && byteCount(output) && output > 0 &&
    (online === null || byteCount(online) && online <= ledger.all_link_serialized_body_bytes &&
      online === ledger.body_bytes_by_edge.filter((edge) => edge.phase === 'online')
        .reduce((sum, edge) => sum + edge.serialized_body_bytes, 0)), 'Invalid authoritative online body counter');
  const network = run.network ?? {};
  const decodeKnown = byteCount(network.decode_output_tokens) && network.decode_output_tokens === output - 1 &&
    network.decode_scope === 'after-first-output-through-completion' && online !== null &&
    byteCount(network.decode_online_body_bytes) && network.decode_online_body_bytes <= online;
  const decodeTokens = decodeKnown ? network.decode_output_tokens : null;
  const decodeBytes = decodeKnown ? network.decode_online_body_bytes : null;
  const rateMatches = (cached, bytes, tokens) => bytes === null || !tokens
    ? cached === null : sameNumber(cached, bytes / 1e6 / tokens);
  requireEvidence(comm.generated_output_tokens === output && comm.covered_body_bytes === ledger.all_link_serialized_body_bytes &&
    comm.online_body_bytes === online && comm.decode_output_tokens === decodeTokens && comm.decode_online_body_bytes === decodeBytes &&
    (comm.body_counter_coverage === undefined || comm.body_counter_coverage === (ledger.body_counter_coverage ?? ledger.schema)) &&
    rateMatches(comm.covered_mb_per_output_token, ledger.all_link_serialized_body_bytes, output) &&
    rateMatches(comm.online_mb_per_output_token, online, output) &&
    rateMatches(comm.decode_online_mb_per_output_token, decodeBytes, decodeTokens),
  'Response body or output denominator changed');
  return comm;
}

function ordinaryMetrics(report, index, helpers) {
  const run = report.runs[index], comm = checkedCommunicationWindow(report, index);
  const ledger = checkedLedger(report.topology_accounting.runs[index]);
  if (ledger.schema.endsWith('.v3')) requireEvidence(Number.isSafeInteger(run.privacy.correction_ack_bytes) &&
    run.privacy.correction_ack_bytes >= 0 && run.privacy.correction_ack_accounted_push_bytes === run.privacy.correction_push_bytes &&
    ledger.body_bytes_by_edge.filter((edge) => edge.source === 'inference' && edge.destination === 'preparation' && edge.phase === 'offline')
      .reduce((sum, edge) => sum + edge.serialized_body_bytes, 0) === run.privacy.correction_ack_bytes,
  'Correction ACK bodies do not match accounted pushes');
  const after = checkedWan(report.native_network_accounting.samples.runs[index].after.wan_emulation);
  requireEvidence(after.conditions_digest === report.configuration.wan_emulation_digest &&
    report.wan_readiness.emulation.native_stream_rate_snapshots_checked === true, 'WAN conditions changed');
  const before = index === 0 ? null : checkedWan(report.native_network_accounting.samples.runs[index].before.wan_emulation);
  const cpu = index === 0 ? report.process_cpu_accounting : null;
  const placement = helpers.executionPlacement(report._pipeline, report.configuration.roles, report.client_body_placement);
  const values = { ...empty(helpers), covered: comm.covered_mb_per_output_token, online: comm.online_mb_per_output_token,
    decodeBytes: comm.decode_online_mb_per_output_token, full: run.durations.full_seconds, ttft: run.durations.ttft_seconds,
    requestTps: run.tokens.output_tokens / run.durations.full_seconds,
    decodeTps: run.tokens.output_tokens > 1 ? (run.tokens.output_tokens - 1) / run.durations.generation_seconds : null,
    aggregateCpu: cpu?.aggregate_cold_first_response_cpu_seconds ?? null,
    clientCpu: cpu?.cold_first_response_cpu_seconds_by_role.client ?? null,
    preparationCpu: cpu?.cold_first_response_cpu_seconds_by_role.preparation ?? null,
    bodyOffload: placement.clientMacPercent === null ? null : 100 - placement.clientMacPercent,
    clientPeak: report.client_process_memory?.after?.lifetime_peak_rss_bytes == null ? null : report.client_process_memory.after.lifetime_peak_rss_bytes / 1e6,
    providerPeak: helpers.providerPeak(report.configuration.roles, run.processes),
    preparationPeak: run.processes?.preparation?.rss_peak_bytes == null ? null : run.processes.preparation.rss_peak_bytes / 1e6,
    inferencePeak: run.processes?.inference?.rss_peak_bytes == null ? null : run.processes.inference.rss_peak_bytes / 1e6,
    totalPeak: report.native_process_memory?.complete === true ? report.native_process_memory.sampled_total_peak_rss_bytes / 1e6 : null,
    clientNetwork: (after.client_stream_bytes - (before?.client_stream_bytes ?? 0)) / 1e6,
    totalNetwork: (after.total_stream_bytes - (before?.total_stream_bytes ?? 0)) / 1e6,
    networkPerToken: (after.total_stream_bytes - (before?.total_stream_bytes ?? 0)) / 1e6 / run.tokens.output_tokens };
  return applyTrafficAttribution(values, { report, requestIndex: index, pipeline: report._pipeline }, helpers.trafficAttributionConsumer ?? null);
}

function rowBase(spec, name, pipeline, configurationDigest, sourceLock, placement) {
  return { id: name, label: name, file: spec.frozen_factory?.file ?? spec.replay_file,
    fileSha256: spec.frozen_factory?.sha256 ?? spec.replay_sha256, configurationDigest,
    replay: { originalDriver: spec.replay_file, originalDriverSha256: spec.replay_sha256,
      source: spec.report.replace(/^.*\//, '').replace(/\.json$/, ''), case: name,
      callable: spec.adapter === 'epoch-prefix-v1' ? 'recorded_contract' : 'case',
      requiresOriginalBuildAndLifecycle: true },
    pipeline, sourceLock, paper: null, reference: null, publicPrefixTokens: 0, placement,
    provenance: { report: spec.report, reportSha256: spec.sha256, plan: spec.plan ?? null, planSha256: spec.plan_sha256 ?? null } };
}

export function benchmarkSuiteCohorts(spec, document, plan, helpers) {
  requireEvidence(document.schema === 'pllm.benchmark_run.v1' && document.complete === true && digest(document.cohort) &&
    document.suite?.schema === 'pllm.benchmark_suite.v1' && document.suite.sampling.warmups === 0 &&
    document.suite.sampling.repetitions === 1 && document.suite.sampling.processes >= 3, 'Require complete repeated SDK cohort');
  const suite = document.suite, expected = suite.cases.length * suite.sampling.processes;
  requireEvidence(document.results.length === expected, 'Missing suite cells');
  const groups = new Map(), seen = new Set();
  for (const result of document.results) {
    const e = result.execution, name = e.case.name, selected = suite.cases.find((item) => item.name === name);
    requireEvidence(isDeepStrictEqual(selected, e.case) && isDeepStrictEqual(planCase(plan, name), e.case), `Measured case differs from exact suite/plan: ${spec.id}/${name}`);
    requireEvidence(result.schema_version === 'pllm.benchmark_result.v2' && result.status === 'completed' &&
      e.cohort === document.cohort && e.failure_code === null && e.admission.admitted === true &&
      e.oracle.outcome === 'passed' && e.isolation.fresh_process === true &&
      e.fixture.metadata.owned_client_cache === 'fresh' &&
      e.build.digest === selected.build.expected_digest && digest(e.build.digest) &&
      isDeepStrictEqual(e.sampling, suite.sampling) && result.warmups === 0 &&
      Number.isInteger(e.replicate) && e.replicate >= 0 && e.replicate < suite.sampling.processes,
    'Unchecked SDK build, isolation, admission or sample');
    for (const key of ['suite_digest', 'case_digest']) requireEvidence(digest(e[key]), 'Missing suite/configuration digest');
    requireEvidence(digest(result.configuration_digest) && digest(result.environment.digest) &&
      Object.keys(e.comparison).length === 7 && Object.values(e.comparison).every(digest) &&
      e.comparison.workload_digest === result.workload_digest && e.comparison.source_digest === result.model.source_lock_digest,
    'Missing comparison commitments');
    const cell = `${name}/${e.replicate}`;
    requireEvidence(!seen.has(cell), 'Duplicate replicate'); seen.add(cell);
    const parameters = e.case.target.parameters, pipeline = parameters.model.experiment.pipeline, report = e.details.report;
    requireEvidence(e.case.target.kind === 'response' && ['repeat', 'cold'].includes(parameters.scenario.kind) &&
      !parameters.concurrent_requests && report?.checks.passed === true && Object.values(report.checks).every((value) => value === true) && report.warmup_runs.length === 0 &&
      report.runs.length === (parameters.scenario.count ?? 1) && report.memory_guard.tripped === false &&
      report.prepared_material_accounting.conserved === true &&
      report.configuration.source_lock_digest === e.comparison.source_digest && e.fixture.digest === e.comparison.source_digest &&
      pipeline.model.source === result.model.id && /^[a-f0-9]{40}$/.test(pipeline.model.revision) &&
      report.runs.every((run) => run.model_id === pipeline.model.source) && digest(report.configuration.prompt_digest),
    'Unchecked response scope or lifecycle');
    const selector = e.details.client_mlp_research ?? null;
    if (parameters.client_mlp) requireEvidence(selector?.pipeline_component === false &&
      isDeepStrictEqual(selector.selection, parameters.client_mlp) && !pipeline.components.client_mlp,
    'Historical MLP selector must not become Pipeline component');
    const key = stable({ comparison: e.comparison, environment: result.environment.digest, build: e.build.digest,
      scope: result.scope, fixture: e.fixture.kind, suite: e.suite_digest });
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push({ result, report, pipeline, selector });
  }
  return [...groups.values()].flatMap((observations, groupIndex) => {
    const first = observations[0], count = first.report.runs.length;
    const names = [...new Set(observations.map(({ result }) => result.execution.case.name))];
    requireEvidence(names.length >= 2 && observations.length === names.length * suite.sampling.processes, 'Disjoint or incomplete comparison group');
    const contract = (result) => result.metrics.map(({ value, sample_count, unavailable_reason, ...rest }) => rest);
    requireEvidence(observations.every(({ result }) => isDeepStrictEqual(contract(result), contract(first.result))), 'Metric ownership, units or timing windows differ');
    return Array.from({ length: count }, (_, index) => {
      const identities = new Set(observations.map(({ report, pipeline }) => stable(runIdentity(report, pipeline, report.runs[index], index))));
      requireEvidence(identities.size === 1, 'Source, numeric, workload, output or lifecycle mismatch');
      const rows = names.map((name) => {
        const samples = observations.filter(({ result }) => result.execution.case.name === name), firstSample = samples[0];
        const configs = new Set(samples.map(({ result }) => result.configuration_digest));
        requireEvidence(configs.size === 1 && samples.every(({ pipeline }) => isDeepStrictEqual(pipeline, firstSample.pipeline)), 'Configuration changed across replicates');
        requireEvidence(samples.every(({ report }) => digest(report.experiment.configuration_digest) &&
          digest(report.experiment.pipeline_digest) && report.experiment.pipeline_digest === firstSample.report.experiment.pipeline_digest),
        'Experiment/Pipeline binding changed across replicates');
        const values = samples.map(({ report, pipeline }) => {
          report.runs.forEach(checkRun);
          requireEvidence(report.runs.every((run, position) => run.warm === (position > 0) && run.cold === (position === 0)), 'Cold/repeat order changed');
          return ordinaryMetrics({ ...report, _pipeline: pipeline }, index, helpers);
        });
        const placement = helpers.executionPlacement(firstSample.pipeline, firstSample.report.configuration.roles, firstSample.report.client_body_placement);
        return { ...rowBase(spec, name, firstSample.pipeline, [...configs][0], firstSample.report.configuration.source_lock_digest, placement),
          researchSelector: firstSample.selector, sampleCount: samples.length, metrics: aggregate(values, Object.keys(empty(helpers))),
          clientMemory: clientMemoryEvidence(samples.map(({ report }) => report.client_process_memory?.after?.lifetime_peak_rss_bytes),
            'process_birth_through_scenario'),
          measurementContracts: firstSample.result.metrics.map(({ value, ...contract }) => contract),
          experimentBindings: samples.map(({ result, report }) => ({ replicate: result.execution.replicate, ...report.experiment })),
          frozenSources: plan.frozen_sources ?? {},
          buildDigest: firstSample.result.execution.build.digest,
          comparisonCommitments: firstSample.result.execution.comparison,
          recordedCase: firstSample.result.execution.case, samplingContract: suite.sampling,
          applicationBodyAttribution: samples.flatMap(({ result, report }) => {
            const attribution = applicationBodyAttribution({ report, requestIndex: index });
            return attribution ? [{ replicate: result.execution.replicate, ...attribution.disclosure }] : [];
          }) };
      });
      const phase = index === 0 ? 'cold' : `repeat-${index}`;
      return { ...spec, id: `${spec.id}-${groupIndex}-${phase}`, title: `${spec.title} · ${first.report.runs[index].tokens.input_tokens} input · ${phase}`,
        baseline: names[0], model: first.report.runs[index].model_id, identity: runIdentity(first.report, first.pipeline, first.report.runs[index], index),
        measurement: { requestScope: 'single request', sampleCount: suite.sampling.processes, phase,
          cpu: index === 0 ? 'Cold first-response CPU; provider process birth through first response' : 'Repeat CPU unmeasured',
          clientRssLabel: count > 1 ? 'Client RAM · scenario-lifetime (cold + repeat)' : 'Client RAM · single cold process-lifetime',
          clientCpuLabel: index === 0 ? 'Client CPU · cold first response' : 'Client CPU · repeat unmeasured',
          preparationCpuLabel: index === 0 ? 'Prep CPU · birth to cold first response' : 'Prep CPU · repeat unmeasured',
          rss: `Client/aggregate RSS cover entire ${count > 1 ? 'cold+repeat' : 'single cold'} scenario; provider RSS sampled in selected request`,
          timing: 'Online TTFT; N−1 decode; request latency excludes provider startup',
          traffic: index === 0 ? 'Cold all-link TCP streams through first response, including startup' : 'Repeat-only all-link TCP-stream counter delta' },
        scope: `${spec.scope} ${index === 0 ? 'Cold request only; TCP streams include startup.' : 'Repeat request only; TCP streams are consecutive snapshot deltas; cold CPU remains unknown.'} Client lifetime and aggregate RSS cover the entire ${count}-request scenario.`, rows };
    });
  });
}

const trafficSchema = 'pllm.benchmark_traffic_decomposition.v1';
const trafficSource = 'sdk_audit_phase_and_prepared_row_payload_counters_v1';
const ackCoverage = 'prepared_protocol_lifecycle_controls_and_correction_acks_v3';
const byteCount = (value) => Number.isSafeInteger(value) && value >= 0;
const trafficAmounts = ['total_all_link_body_bytes', 'known_base_all_link_body_bytes',
  'known_output_dependent_all_link_body_bytes', 'unattributed_all_link_body_bytes'];

function checkedPartition(window, total) {
  requireEvidence(window?.schema === trafficSchema && trafficAmounts.every((key) => byteCount(window[key])) &&
    window.total_all_link_body_bytes === total && window.conservation_checked === true &&
    window.known_base_all_link_body_bytes + window.known_output_dependent_all_link_body_bytes + window.unattributed_all_link_body_bytes === total &&
    typeof window.attribution_valid === 'boolean' && typeof window.headline_eligible === 'boolean', 'Application-body partition conservation failed');
  const complete = window.attribution_valid && window.headline_eligible && window.unattributed_all_link_body_bytes === 0;
  const known = window.known_base_all_link_body_bytes + window.known_output_dependent_all_link_body_bytes;
  requireEvidence(window.status === (complete ? 'Attributed' : window.attribution_valid && known > 0 ? 'Partial' : 'Unattributed') &&
    (window.attribution_valid || known === 0) &&
    window.base_all_link_body_bytes === (complete ? window.known_base_all_link_body_bytes : null) &&
    window.output_dependent_all_link_body_bytes === (complete ? window.known_output_dependent_all_link_body_bytes : null) &&
    (window.unused_output_material_body_bytes === null || byteCount(window.unused_output_material_body_bytes) &&
      window.unused_output_material_body_bytes <= window.known_output_dependent_all_link_body_bytes), 'Incomplete attribution cannot supply headline bodies');
  return complete;
}

function checkedRates(window, output, allowance, input) {
  requireEvidence(window.generated_output_tokens === output && window.output_allowance === allowance && window.input_tokens === input,
    'Application-body output denominator or workload differs');
  const base = window.base_all_link_body_bytes, dependent = window.output_dependent_all_link_body_bytes;
  requireEvidence(base === null ? window.base_all_link_mb_per_request === null : sameNumber(window.base_all_link_mb_per_request, base / 1e6),
    'Application-body base unit differs');
  requireEvidence(dependent === null ? window.incremental_all_link_mb_per_generated_output === null
    : sameNumber(window.incremental_all_link_mb_per_generated_output, dependent / output / 1e6), 'Application-body output rate differs');
}

function checkedBodyEdges(ledger, audit, boundary = false) {
  const expectedEdges = [
    ['client', 'preparation', 'offline', ['session_authorization_upload_bytes', 'preparation_upload_bytes']],
    ['preparation', 'client', 'offline', ['session_authorization_download_bytes', 'preparation_download_bytes']],
    ['preparation', 'inference', 'offline', ['correction_push_bytes']],
    ['inference', 'preparation', 'offline', ['correction_ack_bytes']],
    ['inference', 'client', 'cold', ['bundle_network_bytes']],
    ['client', 'inference', 'online', ['inference_upload_bytes']],
    ['inference', 'client', 'online', ['inference_download_bytes']],
    ...['inference', 'preparation'].flatMap((role) => [
      ['client', role, 'control', [`${role}_control_upload_bytes`]],
      [role, 'client', 'control', [`${role}_control_download_bytes`]],
    ]),
    ...(boundary ? ['boundary_a', 'boundary_b'].flatMap((role) => ['setup', 'online', 'teardown'].flatMap((phase) => [
      ['client', role, phase, [`role_link.${role}.${phase}_upload_bytes`]],
      [role, 'client', phase, [`role_link.${role}.${phase}_download_bytes`]],
    ])) : []),
  ];
  requireEvidence(ledger.body_bytes_by_edge.length === expectedEdges.length && expectedEdges.every(([source, destination, phase, fields]) => {
    const edges = ledger.body_bytes_by_edge.filter((edge) => edge.source === source && edge.destination === destination && edge.phase === phase);
    return edges.length === 1 && fields.every((key) => byteCount(audit[key])) &&
      fields.reduce((sum, key) => sum + audit[key], 0) === edges[0].serialized_body_bytes;
  }), 'Metered application-body edges no longer match raw audit');
}

/** Complete startup/request body owners, charged against their own TCP windows. */
function checkedResponseBodies(report, measurements, boundary) {
  const accounting = report.topology_accounting, samples = report.native_network_accounting.samples;
  requireEvidence(accounting.warmups.length === 0 && accounting.runs.length === 1 && samples.runs.length === 1,
    'Decode accounting windows changed');
  const startup = checkedWan(samples.startup.wan_emulation).directed_stream_bytes;
  const before = checkedWan(samples.runs[0].before.wan_emulation).directed_stream_bytes;
  const after = checkedWan(samples.runs[0].after.wan_emulation).directed_stream_bytes;
  for (const [left, right] of [[startup, before], [before, after]]) {
    requireEvidence(Object.entries(left).every(([edge, value]) => value <= (right[edge] ?? 0)),
      'Decode TCP counters regressed');
  }
  const windows = [accounting.startup, accounting.runs[0]];
  for (const [index, ledger] of windows.entries()) {
    const audit = index === 0 ? ledger.traffic_attribution.audit_counters : report.runs[0].privacy;
    requireEvidence(ledger.schema === `pllm.topology_body_accounting.v${boundary ? 4 : 3}` &&
      ledger.body_counter_coverage === (boundary ? 'prepared_body_and_private_boundary_lifecycle_v4' : ackCoverage) &&
      ledger.tracked_body_counter_set_present === true && ledger.control_body_counter_set_present === true &&
      ledger.correction_ack_counter_set_present === true && ledger.total_wire_bytes === null &&
      byteCount(audit.correction_ack_accounted_push_bytes) && audit.correction_ack_accounted_push_bytes === audit.correction_push_bytes &&
      (!boundary || ledger.boundary_body_counter_set_present === true && audit.private_boundary_body_coverage_failures === 0),
    'Decode raw body coverage incomplete');
    checkedBodyEdges(ledger, audit, boundary);
    for (const [field, select] of [
      ['all_link_serialized_body_bytes', () => true],
      ['client_serialized_body_bytes', (edge) => [edge.source, edge.destination].includes('client')],
      ['online_all_link_serialized_body_bytes', (edge) => edge.phase === 'online'],
      ['online_client_serialized_body_bytes', (edge) => edge.phase === 'online' && [edge.source, edge.destination].includes('client')],
      ['lifecycle_control_serialized_body_bytes', (edge) => edge.phase === 'control'],
    ]) {
      requireEvidence(byteCount(ledger[field]) && ledger[field] === ledger.body_bytes_by_edge.filter(select)
        .reduce((sum, edge) => sum + edge.serialized_body_bytes, 0), 'Decode cached body total differs');
    }
    const directed = {};
    for (const edge of ledger.body_bytes_by_edge) {
      const link = `${edge.source}->${edge.destination}`;
      directed[link] = (directed[link] ?? 0) + edge.serialized_body_bytes;
    }
    requireEvidence(Object.entries(directed).every(([link, value]) => value <= (index === 0
      ? startup[link] ?? 0 : (after[link] ?? 0) - (before[link] ?? 0))), 'Decode bodies exceed matching TCP window');
  }
  requireEvidence(windows.reduce((sum, ledger) => sum + ledger.all_link_serialized_body_bytes, 0) === measurements['network.covered_body_bytes'],
    'Decode body metric differs from raw owners');
}

function checkedPreparedAttribution(run, ledger, window) {
  const attribution = window.attribution;
  requireEvidence(isDeepStrictEqual(attribution, ledger.traffic_attribution ?? null), 'Cached application-body attribution differs from ledger');
  const snapshot = attribution?.audit_counters;
  if (snapshot !== undefined) requireEvidence(snapshot !== null && typeof snapshot === 'object' && !Array.isArray(snapshot) &&
    Object.keys(snapshot).length <= 64 && Object.values(snapshot).every((value) => value === null || byteCount(value)),
  'Invalid bounded application-body audit snapshot');
  if (!window.attribution_valid) return false;
  requireEvidence(attribution?.schema === trafficSchema && attribution.source === trafficSource &&
    attribution.attribution_valid === true && window.headline_eligible === attribution.headline_eligible,
  'Missing metered application-body attribution source');
  checkedPartition(attribution, ledger.all_link_serialized_body_bytes);
  const components = attribution.components_body_bytes;
  requireEvidence(components && Object.entries(components).every(([key, value]) =>
    ['setup_and_delivery', 'prefill_online', 'prefill_material', 'decode_online', 'decode_material'].includes(key) && byteCount(value)) &&
    ['setup_and_delivery', 'prefill_online', 'prefill_material'].reduce((sum, key) => sum + (components[key] ?? 0), 0) === window.known_base_all_link_body_bytes &&
    ['decode_online', 'decode_material'].reduce((sum, key) => sum + (components[key] ?? 0), 0) === window.known_output_dependent_all_link_body_bytes,
  'Application-body component classification differs');
  // Legacy startup disclosures remain hash-bound records, not fresh attribution proof.
  if (!run && snapshot === undefined) return ledger.all_link_serialized_body_bytes === 0;
  const audit = run ? run.privacy : snapshot;
  if (run && snapshot !== undefined) requireEvidence(Object.entries(snapshot).every(([key, value]) =>
    value === (byteCount(audit[key]) ? audit[key] : null)), 'Application-body audit snapshot differs from raw response');
  const rowConserved = ['reserved', 'claimed', 'burned'].every((name) => byteCount(audit[`prepared_stage_rows_${name}`])) &&
    audit.prepared_stage_rows_reserved === audit.prepared_stage_rows_claimed + audit.prepared_stage_rows_burned;
  requireEvidence(attribution.headline_eligible === rowConserved, 'Application-body headline eligibility lacks row conservation');
  checkedBodyEdges(ledger, audit);
  requireEvidence(attribution.generated_output_tokens === (run ? run.tokens.output_tokens : audit.generation_output_tokens ?? null) &&
    attribution.output_allowance === (run ? run.max_output_tokens : audit.generation_requested_output_tokens ?? null),
  'Raw application-body output identity differs');
  const unsupported = ['decode_epochs_started', 'inflight_producer_registrations', 'inflight_joins', 'public_prefix_artifact_local_bytes']
    .some((key) => key in audit && audit[key] !== 0);
  if ('setup_and_delivery' in components) requireEvidence(components.setup_and_delivery ===
    audit.bundle_network_bytes + audit.session_authorization_upload_bytes + audit.session_authorization_download_bytes, 'Setup/delivery body attribution differs');
  if ('prefill_online' in components || 'decode_online' in components) {
    const zeroOnlineStartup = !run && ledger.online_all_link_serialized_body_bytes === 0 &&
      audit.inference_upload_bytes === 0 && audit.inference_download_bytes === 0 &&
      components.prefill_online === 0 && components.decode_online === 0;
    requireEvidence(zeroOnlineStartup || !unsupported && run &&
      audit.generation_attribution_supported_responses === 1 && audit.generation_completed_responses === 1 &&
      audit.generation_output_tokens === run.tokens.output_tokens && audit.generation_decode_output_tokens === run.tokens.output_tokens - 1 &&
      byteCount(audit.generation_executed_decode_steps) && audit.generation_executed_decode_steps >= run.tokens.output_tokens - 1 &&
      audit.generation_executed_decode_steps <= run.tokens.output_tokens && audit.generation_requested_output_tokens === run.max_output_tokens &&
      components.prefill_online === audit.generation_prefill_online_body_bytes && components.decode_online === audit.generation_decode_online_body_bytes &&
      components.prefill_online + components.decode_online === ledger.online_all_link_serialized_body_bytes,
    'Online application-body phase attribution differs');
  }
  if ('prefill_material' in components || 'decode_material' in components) requireEvidence(!unsupported &&
    components.prefill_material === audit.preparation_prefill_material_body_bytes && components.decode_material === audit.preparation_decode_material_body_bytes &&
    byteCount(audit.preparation_attributed_body_bytes) && audit.preparation_attributed_body_bytes ===
      audit.preparation_upload_bytes + audit.preparation_download_bytes + audit.correction_push_bytes &&
    (!['reserved', 'claimed', 'burned'].some((name) => `prepared_stage_rows_${name}` in audit) || rowConserved) &&
    components.prefill_material + components.decode_material <= audit.preparation_attributed_body_bytes &&
    ['prefill', 'decode', 'padding'].every((name) => byteCount(audit[`preparation_${name}_rows_issued`])) &&
    ['prefill', 'decode', 'padding'].reduce((sum, name) => sum + audit[`preparation_${name}_rows_issued`], 0) === audit.prepared_stage_rows_issued,
  'Prepared material attribution differs from measured row payloads');
  return true;
}

/** Select canonical request windows, never scenario summaries or TCP counters. */
export function applicationBodyAttribution({ report, requestIndex }) {
  const direct = report?.traffic_decomposition, nested = report?.communication_per_token?.traffic_decomposition;
  if (direct == null && nested == null) return null;
  requireEvidence(direct == null || nested == null || isDeepStrictEqual(direct, nested), 'Application-body report copies differ');
  const document = direct ?? nested;
  requireEvidence(report.schema_version === 'pllm.loopback_benchmark.v1' && document.schema === trafficSchema && document.source === trafficSource &&
    document.scope === 'measured directed all-link serialized application bodies; not TCP stream or physical wire' &&
    document.byte_unit === 'decimal MB (1000000 bytes)' && document.tcp_stream_bytes === null && document.physical_wire_bytes === null &&
    document.windows_aligned === true && document.body_counter_coverage === ackCoverage &&
    document.lifecycle_scope === 'startup charged once to first warmup or first measured request; repeats exclude startup' &&
    report.warmup_runs.length === 0 && report.topology_accounting.warmups.length === 0 && document.warmups.length === 0 && document.runs.length === report.runs.length &&
    report.runs.every((run) => run.status === 'completed' && run.tokens.authoritative === true &&
      byteCount(run.tokens.input_tokens) && byteCount(run.tokens.output_tokens) && run.tokens.output_tokens > 0 &&
      byteCount(run.max_output_tokens) && run.tokens.output_tokens <= run.max_output_tokens) &&
    Number.isInteger(requestIndex) && requestIndex >= 0 && requestIndex < report.runs.length,
  'Unknown application-body schema, plane, coverage or selected window');
  const accounting = report.topology_accounting, startupLedger = checkedLedger(accounting.startup);
  const startup = document.startup;
  checkedPartition(startup, startupLedger.all_link_serialized_body_bytes);
  const startupProven = checkedPreparedAttribution(null, startupLedger, startup);
  for (const [index, window] of document.runs.entries()) {
    const run = report.runs[index], ledger = checkedLedger(accounting.runs[index]);
    requireEvidence(ledger.body_counter_coverage === document.body_counter_coverage &&
      startupLedger.body_counter_coverage === document.body_counter_coverage && window.cold === run.cold, 'Application-body window coverage or lifecycle differs');
    checkedPartition(window, ledger.all_link_serialized_body_bytes);
    checkedRates(window, run.tokens.output_tokens, run.max_output_tokens, run.tokens.input_tokens);
    checkedPreparedAttribution(run, ledger, window);
    requireEvidence(window.known_output_dependent_all_link_mb_per_generated_output === null ||
      window.attribution_valid && sameNumber(window.known_output_dependent_all_link_mb_per_generated_output,
        window.known_output_dependent_all_link_body_bytes / run.tokens.output_tokens / 1e6), 'Known output-dependent denominator differs');
  }
  requireEvidence(isDeepStrictEqual(document.repeat_requests, document.runs.slice(1)), 'Repeat windows silently include startup');
  const first = document.runs[0], cold = document.cold_first_request;
  checkedPartition(cold, startup.total_all_link_body_bytes + first.total_all_link_body_bytes);
  const accepted = startup.attribution_valid && first.attribution_valid;
  requireEvidence(cold.attribution_valid === accepted && cold.headline_eligible === (startup.headline_eligible && first.headline_eligible) &&
    cold.known_base_all_link_body_bytes === (accepted ? startup.known_base_all_link_body_bytes + first.known_base_all_link_body_bytes : 0) &&
    cold.known_output_dependent_all_link_body_bytes === (accepted ? startup.known_output_dependent_all_link_body_bytes + first.known_output_dependent_all_link_body_bytes : 0),
  'Cold-first attribution did not charge startup exactly once');
  checkedRates(cold, report.runs[0].tokens.output_tokens, report.runs[0].max_output_tokens, report.runs[0].tokens.input_tokens);
  const lifecycle = document.covered_lifecycle;
  checkedPartition(lifecycle, startup.total_all_link_body_bytes + document.runs.reduce((sum, window) => sum + window.total_all_link_body_bytes, 0));
  const lifecycleWindows = [startup, ...document.runs], lifecycleAccepted = lifecycleWindows.every((window) => window.attribution_valid);
  requireEvidence(lifecycle.attribution_valid === lifecycleAccepted &&
    lifecycle.headline_eligible === lifecycleWindows.every((window) => window.headline_eligible) &&
    lifecycle.known_base_all_link_body_bytes === (lifecycleAccepted ? lifecycleWindows.reduce((sum, window) => sum + window.known_base_all_link_body_bytes, 0) : 0) &&
    lifecycle.known_output_dependent_all_link_body_bytes === (lifecycleAccepted ? lifecycleWindows.reduce((sum, window) => sum + window.known_output_dependent_all_link_body_bytes, 0) : 0),
  'Lifecycle application-body classification does not reconcile startup once');
  const window = requestIndex === 0 ? cold : document.runs[requestIndex], run = report.runs[requestIndex];
  const complete = window.status === 'Attributed' && (requestIndex > 0 || startupProven);
  return { schema: trafficSchema, scope: 'covered_application_bodies', validated: true, complete,
    baseMbPerRequest: complete ? window.base_all_link_mb_per_request : null,
    outputDependentMbPerToken: complete ? window.incremental_all_link_mb_per_generated_output : null,
    disclosure: { accountingPlane: 'covered_application_bodies', scope: document.scope, byteUnit: document.byte_unit,
      window: requestIndex === 0 ? 'startup + cold first request (startup charged once)' : `request ${requestIndex} only (startup excluded)`,
      bodyCounterCoverage: document.body_counter_coverage,
      status: !complete && window.status === 'Attributed' ? 'Partial' : window.status,
      startupRawAuditValidated: startupProven && startup.attribution?.audit_counters !== undefined,
      generatedOutputTokens: run.tokens.output_tokens, outputAllowance: run.max_output_tokens,
      knownBaseBytes: window.known_base_all_link_body_bytes, knownOutputDependentBytes: window.known_output_dependent_all_link_body_bytes,
      residualBytes: window.unattributed_all_link_body_bytes, totalBytes: window.total_all_link_body_bytes,
      rateScope: document.rate_scope, unusedOutputMaterialBytes: window.unused_output_material_body_bytes } };
}

/** Optional legacy hook may agree with canonical evidence, never substitute for it. */
export function applyTrafficAttribution(values, evidence, consumer = null) {
  const split = applicationBodyAttribution(evidence);
  if (consumer) {
    const proposed = consumer(evidence);
    requireEvidence(proposed === null || split?.complete && proposed.scope === split.scope && proposed.validated === true && proposed.complete === true &&
      sameNumber(proposed.baseMbPerRequest, split.baseMbPerRequest) && sameNumber(proposed.outputDependentMbPerToken, split.outputDependentMbPerToken),
    'Optional attribution hook differs from canonical application bodies');
  }
  if (!split?.complete) return { ...values, baseNetwork: null, tokenNetwork: null };
  return { ...values, baseNetwork: split.baseMbPerRequest, tokenNetwork: split.outputDependentMbPerToken };
}

export function epochPrefixCohorts(spec, document, helpers) {
  requireEvidence(document.schema === 'pllm.epoch_prefix_sdk_benchmark.v1' && document.complete === true &&
    document.results.length === 36 && Object.keys(document.experiments).length === 12 &&
    digest(document.workload_commitment) && digest(document.source_lock_digest) && digest(document.build.expected_digest) &&
    Object.keys(document.frozen_sources).length === 5 && Object.values(document.frozen_sources).every(digest) &&
    document.memory_guard.failed === false && document.memory_guard.maximum_swap_growth_bytes === 0 &&
    document.checks.exact_state_and_output_match === true && document.checks.checked_response_states === 60 &&
    document.checks.workload_lifecycles_checked === true && document.checks.pending_joins_checked === true,
  'Require completed 36-cell epoch/prefix confirmation');
  const seen = new Set(), groups = new Map();
  for (const cell of document.results) {
    const name = `${cell.workload}/${cell.variant}`, experiment = document.experiments[name];
    requireEvidence(experiment && Number.isInteger(cell.replicate) && cell.replicate >= 0 && cell.replicate < 3 &&
      !seen.has(`${name}/${cell.replicate}`), 'Missing or duplicate epoch configuration/replicate');
    seen.add(`${name}/${cell.replicate}`);
    const client = cell.client, audit = client.audit, grouped = ['shared', 'unrelated'].includes(cell.workload);
    requireEvidence(client.responses.length === (grouped ? 2 : 1) &&
      client.responses.every((response) => response.status === (cell.workload === 'early' ? 'completed' : 'incomplete') && response.output_tokens > 0 &&
        Object.values(response.state).filter((value) => typeof value === 'string').every(digest)) &&
      client.cleanup.known_private_containers_empty === true &&
      Object.values(client.cleanup.container_counts).every((value) => value === 0) &&
      client.cleanup.owner_counters_after_close.every((owner) => ['reserved_bytes', 'requests', 'nodes', 'pending_nodes', 'active_producers'].every((key) => owner[key] === 0)) &&
      audit.plaintext_prompt_bytes_sent === 0 && audit.plaintext_token_ids_sent === 0 &&
      audit.preparation_requests_during_online === 0 &&
      audit.prepared_stage_rows_reserved === audit.prepared_stage_rows_claimed + audit.prepared_stage_rows_burned,
    'Epoch response privacy, exact state or cleanup lifecycle failed');
    const sharing = ['inflight', 'combined'].includes(cell.variant), epochs = ['epochs', 'combined'].includes(cell.variant);
    requireEvidence(audit.inflight_joins === (cell.workload === 'shared' && sharing ? 1 : 0) &&
      audit.decode_epoch_pauses === (cell.workload !== 'early' && epochs ? client.responses.length : 0) &&
      client.responses.every((response) => cell.workload === 'early'
        ? response.output_tokens < experiment.budget.max_new_tokens : response.output_tokens === experiment.budget.max_new_tokens),
    'Declared pending join, epoch transition or EOS/cap lifecycle differs');
    checkedWan(cell.wan); checkedLedger(client.ledger);
    if (!groups.has(cell.workload)) groups.set(cell.workload, []);
    groups.get(cell.workload).push(cell);
  }
  return [...groups].map(([workload, cells]) => {
    const names = [...new Set(cells.map((cell) => cell.variant))], grouped = cells[0].client.responses.length > 1;
    requireEvidence(names.includes('control') && cells.length === names.length * 3, 'Incomplete epoch workload');
    const controls = cells.filter((cell) => cell.variant === 'control');
    const scopeIdentity = (cell) => ({ wanDigest: cell.wan.conditions_digest, wanConditions: cell.wan.conditions,
      roleParties: cell.wan.role_parties, ledgerSchema: cell.client.ledger.schema,
      bodyCoverage: cell.client.ledger.body_counter_coverage, bodyWindow: cell.client.ledger.measurement_window,
      includesBundleAndPreparation: cell.client.ledger.window_includes_bundle_and_preparation,
      controlsPresent: cell.client.ledger.control_body_counter_set_present, controlScope: cell.client.ledger.control_scope,
      resourceWindows: document.scope });
    requireEvidence(cells.every((cell) => cell.client.ledger.schema === 'pllm.topology_body_accounting.v2' &&
      cell.client.ledger.body_counter_coverage === 'prepared_protocol_and_instrumented_lifecycle_controls_v2' &&
      cell.client.ledger.measurement_window === 'client_construction_through_close' &&
      cell.client.ledger.window_includes_bundle_and_preparation === true &&
      isDeepStrictEqual(scopeIdentity(cell), scopeIdentity(controls[0]))),
    'Epoch WAN, body coverage or measurement-window identity differs');
    const commitment = (cell) => cell.client.responses.map(({ index, input_tokens, output_tokens, state }) => ({ index, input_tokens, output_tokens, state }));
    requireEvidence(cells.every((cell) => isDeepStrictEqual(commitment(cell), commitment(controls[0]))), 'Exact state/output or workload commitments differ');
    const numeric = names.map((name) => {
      const experiment = document.experiments[`${workload}/${name}`], p = experiment.pipeline;
      return stable({ model: p.model, quantization: p.components.quantization, nonlinear: p.components.nonlinear ?? null,
        state: p.components.state ?? null, verification: p.components.verification ?? null, budget: experiment.budget,
        kernels: p.components.kernels });
    });
    requireEvidence(new Set(numeric).size === 1, 'Epoch source, numeric or workload bound differs');
    const rows = names.map((variant) => {
      const samples = cells.filter((cell) => cell.variant === variant), experiment = document.experiments[`${workload}/${variant}`];
      const values = samples.map((cell) => {
        const client = cell.client, outputs = client.responses.reduce((sum, response) => sum + response.output_tokens, 0);
        return { ...empty(helpers), full: grouped ? null : client.responses[0].request_seconds,
          // Setup/queue-inclusive text events are not ordinary online token axes.
          ttft: null, decodeTps: null,
          requestTps: grouped ? null : outputs / client.responses[0].request_seconds,
          covered: client.ledger.all_link_serialized_body_bytes / 1e6 / outputs,
          online: client.ledger.online_all_link_serialized_body_bytes / 1e6 / outputs,
          aggregateCpu: cell.aggregate_cleanup_cpu_seconds, clientCpu: client.client_lifetime_cpu_seconds,
          preparationCpu: cell.provider_birth_cpu_seconds.preparation,
          clientPeak: client.client_peak_rss_bytes / 1e6,
          preparationPeak: cell.provider_sampled_peak_rss_bytes.preparation / 1e6,
          inferencePeak: cell.provider_sampled_peak_rss_bytes.inference / 1e6,
          providerPeak: Math.max(...Object.values(cell.provider_sampled_peak_rss_bytes)) / 1e6,
          totalNetwork: cell.wan.total_stream_bytes / 1e6, clientNetwork: cell.wan.client_stream_bytes / 1e6,
          networkPerToken: cell.wan.total_stream_bytes / 1e6 / outputs };
      });
      const row = rowBase(spec, variant, experiment.pipeline, hash(stable(experiment)), document.source_lock_digest,
        helpers.executionPlacement(experiment.pipeline, ['client', 'inference', 'preparation']));
      row.replay.case = `${workload}/${variant}`;
      return { ...row,
        sampleCount: 3, researchSelector: null, measurementContracts: [], metrics: aggregate(values, Object.keys(empty(helpers))),
        clientMemory: clientMemoryEvidence(samples.map((cell) => cell.client.client_peak_rss_bytes), 'complete_cell_process_lifetime'),
        experiment, frozenSources: document.frozen_sources, buildDigest: document.build.expected_digest,
        scopedTimings: { firstTextSeconds: median(samples.map((cell) => median(cell.client.responses.map((response) => response.first_text_seconds)))),
          textDeltaDecodeTps: median(samples.map((cell) => median(cell.client.responses.map((response) => response.decode_tps)))),
          groupSeconds: median(samples.map((cell) => cell.client.group_seconds)),
          completeCellSeconds: median(samples.map((cell) => cell.complete_case_seconds)) } };
    });
    const first = controls[0], responses = first.client.responses, pipeline = document.experiments[`${workload}/control`].pipeline;
    return { ...spec, id: `${spec.id}-${workload}`, title: `${spec.title} · ${workload} · ${grouped ? 'two-request group' : 'single request'}`,
      baseline: 'control', model: pipeline.model.source,
      identity: { model: pipeline.model.source, source: pipeline.model, body: null, prompt: document.workload_commitment,
        quantization: pipeline.components.quantization, inputs: responses.reduce((sum, response) => sum + response.input_tokens, 0),
        outputs: responses.reduce((sum, response) => sum + response.output_tokens, 0),
        cap: document.experiments[`${workload}/control`].budget.max_new_tokens * responses.length,
        warm: false, output: responses.map((response) => response.state.output).join(','), backend: 'native',
        nonlinear: null, state: null, verification: null, sampling: null, wan: first.wan.conditions_digest, links: null,
        wanBackend: 'native-shared-tcp-pacer', clientIsolation: true, measurementIdentity: scopeIdentity(first) },
      measurement: { requestScope: grouped ? 'two-request group (token counts summed)' : 'single request', phase: 'cold', sampleCount: 3,
        cpu: 'Client lifetime CPU; aggregate includes launcher/relays and all reaped children through cleanup; Preparation birth through client exit',
        clientRssLabel: 'Client RAM · complete-cell lifetime', clientCpuLabel: 'Client CPU · through cleanup',
        preparationCpuLabel: 'Prep CPU · birth to client exit',
        rss: 'Client lifetime peak; providers sampled every 50 ms after startup; aggregate RSS unknown',
        timing: grouped ? 'First-text delay and N−1 decode: median across both responses, then three cells; group latency/throughput not ranked'
          : 'First-text delay includes setup/waiting; N−1 text-delta decode includes epoch pauses',
        traffic: 'Complete-cell TCP streams; covered application bodies v2 include instrumented lifecycle controls' },
      scope: `${spec.scope} ${document.scope.join('; ')}. ${grouped ? 'Grouped and single-request metrics do not rank together; per-response timing first takes within-cell median.' : 'Single-request workload view.'}`, rows };
  });
}

export function adaptedCohorts(spec, read, helpers) {
  if (['private-boundary-row-codec-v1', 'private-boundary-row-codec-coverage-v1'].includes(spec.adapter)) {
    throw new Error('Row-codec evidence is coverage-only pending independent read-only audit; no summary-to-score fallback');
  }
  const { document, plan } = readBound(spec, read);
  if (spec.adapter === 'benchmark-suite-v1') return benchmarkSuiteCohorts(spec, document, plan, helpers);
  if (spec.adapter === 'epoch-prefix-v1') return epochPrefixCohorts(spec, document, helpers);
  if (spec.adapter === 'private-boundary-v1') return privateBoundaryCohorts(spec, document, read, helpers);
  if (spec.adapter === 'private-boundary-decode-confirmation-v1') {
    return privateBoundaryDecodeConfirmationCohorts(spec, document, plan, read, { ...helpers, checkRun, checkedWan, checkedResponseBodies });
  }
  if (spec.adapter === 'private-boundary-byte-dot-v1') {
    return privateBoundaryByteDotCohorts(spec, document, plan, read, { ...helpers, checkRun, checkedWan, checkedResponseBodies });
  }
  if (['private-boundary-tiers-v1', 'private-boundary-decode-v1'].includes(spec.adapter)) {
    return privateBoundaryTierCohorts(spec, document, plan, read, { ...helpers, checkRun, checkedWan, checkedResponseBodies });
  }
  throw new Error('Unknown scorecard evidence adapter');
}
