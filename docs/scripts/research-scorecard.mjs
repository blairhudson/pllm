/** Evidence-backed scorecards. Never combine independently salted cohorts. */
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../../', import.meta.url));
const hash = (value) => createHash('sha256').update(value).digest('hex');

export const metrics = [
  { id: 'covered', label: 'Covered bodies / output token', unit: 'MB/token', direction: 'min' },
  { id: 'online', label: 'Online bodies / output token', unit: 'MB/token', direction: 'min' },
  { id: 'decodeBytes', label: 'Decode bodies / later output', unit: 'MB/token', direction: 'min' },
  { id: 'full', label: 'Full request latency', unit: 's', direction: 'min' },
  { id: 'ttft', label: 'Time to first token', unit: 's', direction: 'min' },
  { id: 'requestTps', label: 'Full request throughput', unit: 'tokens/s', direction: 'max' },
  { id: 'decodeTps', label: 'Decode throughput (N−1)', unit: 'tokens/s', direction: 'max' },
  { id: 'aggregateCpu', label: 'Cold aggregate process CPU', unit: 'CPU s', direction: 'min' },
  { id: 'clientCpu', label: 'Cold client process CPU', unit: 'CPU s', direction: 'min' },
  { id: 'preparationPeak', label: 'Preparation lifetime peak RSS', unit: 'MB', direction: 'min' },
  { id: 'inferencePeak', label: 'Inference lifetime peak RSS', unit: 'MB', direction: 'min' },
  { id: 'artifactBytes', label: 'Pre-positioned public artifacts', unit: 'MB', direction: 'min' },
  { id: 'clientPeak', label: 'Isolated client peak RSS', unit: 'MB', direction: 'min' },
  { id: 'wire', label: 'Full physical wire / output', unit: 'MB/token', direction: 'min' },
  { id: 'quality', label: 'Representative generation quality', unit: 'score', direction: 'max' },
  { id: 'energy', label: 'Whole-response energy', unit: 'J', direction: 'min' },
];

export function cohortRows(cohort, document, read = (file) => fs.readFileSync(path.join(root, file))) {
  if (!Array.isArray(document.candidates) || document.candidates.length < 2) {
    throw new Error('Require a canonical multi-Experiment cohort with its shared salt');
  }
  for (const [file, expected] of Object.entries(cohort.configuration_sources ?? {})) {
    if (hash(read(file)) !== expected) throw new Error(`Measured configuration source changed: ${file}`);
  }
  const publisher = cohort.publisher_report ? JSON.parse(read(cohort.publisher_report)) : null;
  const identities = new Set();
  const sourceLocks = new Map();
  const names = new Set();
  const rows = document.candidates.map((candidate) => {
    if (names.has(candidate.name)) throw new Error('Duplicate candidate');
    names.add(candidate.name);
    const spec = cohort.candidates[candidate.name];
    if (!spec || !spec.file.endsWith('.py')) throw new Error('Every score needs its Python configuration');
    const fileSha256 = hash(read(spec.file));
    if (fileSha256 !== spec.sha256) throw new Error(`Measured Python configuration changed: ${spec.file}`);
    const report = candidate.report;
    if (!report.checks?.passed || report.runs?.length !== 1 || report.warmup_runs?.length) {
      throw new Error('Require checked single-response, zero-warmup evidence');
    }
    const run = report.runs[0], config = report.configuration;
    if (run.status !== 'completed' || !run.tokens.authoritative || !run.generation?.output_text_digest ||
        !run.model_id || !run.model_fingerprint || !config.prompt_digest || !config.source_lock_digest || !run.tokens.output_tokens) {
      throw new Error('Incomplete authoritative identity');
    }
    const identity = {
      model: run.model_id, source: candidate.pipeline.model, body: run.model_fingerprint, prompt: config.prompt_digest,
      quantization: candidate.pipeline.components.quantization, inputs: run.tokens.input_tokens,
      outputs: run.tokens.output_tokens, cap: run.max_output_tokens, sampling: run.sampling,
      warm: run.warm, output: run.generation.output_text_digest, backend: config.provider_backend,
      wan: config.wan_emulation_digest ?? null, links: config.link_conditions_digest ?? null,
    };
    identities.add(JSON.stringify(identity));
    const roles = [...config.roles].sort().join(',');
    if (sourceLocks.has(roles) && sourceLocks.get(roles) !== config.source_lock_digest) {
      throw new Error('Source lock changed within a role graph');
    }
    sourceLocks.set(roles, config.source_lock_digest);
    let artifactBytes = 0, publicPrefixTokens = 0;
    for (const [slot, key, implementation] of [
      ['tokenizer', 'tokenizer', 'pllm/indexed-tokenizer/v1'],
      ['public_prefix', 'capsule', 'pllm/public-prefix-capsule/v1'],
    ]) {
      const selected = candidate.pipeline.components[slot];
      if (!selected) continue;
      const artifact = publisher?.[key];
      if (publisher?.schema !== 'pllm.public_artifact_build.v1' ||
          publisher.body_fingerprint !== run.model_fingerprint ||
          publisher.input_tokens !== run.tokens.input_tokens || publisher.output_cap !== run.max_output_tokens ||
          selected.component !== implementation || selected.params.digest !== artifact?.digest ||
          !Number.isSafeInteger(artifact?.bytes) || artifact.bytes <= 0) {
        throw new Error('Public artifact does not match publisher evidence and measured Pipeline');
      }
      artifactBytes += artifact.bytes;
      if (slot === 'public_prefix') {
        publicPrefixTokens = publisher.public_prefix_tokens;
        if (selected.params.size_bytes !== artifact.bytes || !Number.isSafeInteger(publicPrefixTokens) ||
            publicPrefixTokens <= 0 || publicPrefixTokens > run.tokens.input_tokens) {
          throw new Error('Invalid public-prefix artifact bounds');
        }
      }
    }
    if (spec.artifact_bytes !== artifactBytes || (spec.public_prefix_tokens ?? 0) !== publicPrefixTokens) {
      throw new Error('Public artifact cost or prefix scope differs from measured evidence');
    }
    const comm = report.communication_per_token.summary, cpu = report.process_cpu_accounting;
    const metricValues = {
      covered: comm.setup_inclusive_mb_per_output_token, online: comm.online_mb_per_output_token,
      decodeBytes: comm.decode_online_mb_per_output_token, full: run.durations.full_seconds,
      ttft: run.durations.ttft_seconds,
      requestTps: run.durations.full_seconds > 0 ? run.tokens.output_tokens / run.durations.full_seconds : null,
      decodeTps: run.tokens.output_tokens > 1 && run.durations.generation_seconds > 0
        ? (run.tokens.output_tokens - 1) / run.durations.generation_seconds : null,
      aggregateCpu: cpu?.aggregate_cold_first_response_cpu_seconds ?? null,
      clientCpu: cpu?.cold_first_response_cpu_seconds_by_role?.client ?? null,
      preparationPeak: run.processes?.preparation?.rss_peak_bytes == null ? null : run.processes.preparation.rss_peak_bytes / 1e6,
      inferencePeak: run.processes?.inference?.rss_peak_bytes == null ? null : run.processes.inference.rss_peak_bytes / 1e6,
      artifactBytes: artifactBytes / 1e6,
      // Candidates share the client process. A lifetime high-water mark cannot rank them.
      clientPeak: null, wire: null, quality: null, energy: null,
    };
    for (const value of Object.values(metricValues)) {
      if (value !== null && (!Number.isFinite(value) || value < 0)) throw new Error('Invalid metric');
    }
    return { id: candidate.name, label: spec.label, file: spec.file, fileSha256,
      configurationDigest: candidate.configuration_digest, pipeline: candidate.pipeline,
      sourceLock: config.source_lock_digest, paper: spec.paper ?? null,
      publicPrefixTokens, metrics: metricValues };
  });
  if (identities.size !== 1) throw new Error('Cohort identity, numeric contract, or outputs differ');
  if (!names.has(cohort.baseline) || names.size !== Object.keys(cohort.candidates).length) {
    throw new Error('Baseline or configured score missing');
  }
  return { ...cohort, candidates: undefined, model: document.candidates[0].report.runs[0].model_id,
    identity: JSON.parse([...identities][0]), rows };
}

export function buildScorecard() {
  const manifest = JSON.parse(fs.readFileSync(path.join(root, 'docs/data/research/benchmark-cohorts.json')));
  return { schema: 'pllm.research_scorecard.v1', metrics,
    cohorts: manifest.cohorts.map((cohort) => {
      const raw = fs.readFileSync(path.join(root, cohort.report));
      return { ...cohortRows(cohort, JSON.parse(raw)), reportSha256: hash(raw) };
    }) };
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const target = path.join(root, 'docs/data/research/scorecard.json');
  fs.writeFileSync(target, JSON.stringify(buildScorecard(), null, 2) + '\n');
}
