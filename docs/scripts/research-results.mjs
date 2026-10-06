/** Complete research ledgers, derived from the same checked sources as the chart. */
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { buildScorecard, reportMetrics } from './research-scorecard.mjs';

const root = fileURLToPath(new URL('../../', import.meta.url));
const repository = 'https://github.com/blairhudson/pllm/blob/main/';
const hash = (bytes) => createHash('sha256').update(bytes).digest('hex');
const number = (value) => value == null ? 'Unmeasured' : Number(value).toLocaleString('en', { maximumFractionDigits: 3 });
const cell = (value) => String(value).replaceAll('|', '\\|').replaceAll('\n', ' ').replaceAll('<', '&lt;');
const table = (head, rows) => [head, head.map(() => '---'), ...rows].map((row) => `| ${row.map(cell).join(' | ')} |`).join('\n');
const link = (title, file) => `[${title}](${repository}${file})`;

export function searchResults(spec, read = (file) => fs.readFileSync(path.join(root, file))) {
  const raw = read(spec.file);
  if (hash(raw) !== spec.sha256) throw new Error('Search evidence changed');
  const document = JSON.parse(raw);
  const search = document.search;
  if (search.evaluations.length + search.rejections.length !== search.attempted ||
      document.candidates.length !== search.evaluations.length) throw new Error('Search attempts do not reconcile');
  const trials = document.candidates.map((candidate) => {
    if (hash(read(candidate.configuration_file)) !== candidate.configuration_sha256) throw new Error('Search Python configuration changed');
    const evaluation = search.evaluations.find((item) => item.trial_id === candidate.name);
    if (evaluation?.result.configuration_digest !== candidate.configuration_digest || !candidate.report.checks.passed) {
      throw new Error('Search measurement identity mismatch');
    }
    return { id: candidate.name, configurationDigest: candidate.configuration_digest,
      file: candidate.configuration_file, fileSha256: candidate.configuration_sha256,
      metrics: evaluation.result.metrics, observations: reportMetrics(candidate.report), pipeline: candidate.pipeline };
  });
  return { ...spec, attempted: search.attempted, stopReason: search.stop_reason, trials,
    rejections: search.rejections.map((item) => ({ reason: item.reason, parameters: item.parameters })) };
}

export function qualityResults(spec, read = (file) => fs.readFileSync(path.join(root, file))) {
  const raw = read(spec.report), calibrationRaw = read(spec.calibration_report);
  if (hash(raw) !== spec.sha256 || hash(calibrationRaw) !== spec.calibration_sha256) {
    throw new Error('Quality or calibration evidence changed');
  }
  for (const [file, expected] of Object.entries(spec.configuration_sources)) {
    if (hash(read(file)) !== expected) throw new Error('Quality configuration source changed');
  }
  const document = JSON.parse(raw), calibration = JSON.parse(calibrationRaw);
  const prompts = JSON.parse(read(spec.evaluation_prompts));
  const calibrationPrompts = JSON.parse(read(spec.calibration_prompts));
  if (document.schema_version !== 'pllm.reference_quality_benchmark.v1' ||
      document.scope !== 'local-compiled-clear-kernel-prefill-reference' ||
      document.cohort.phase !== 'prefill' || document.cohort.prompt_count !== prompts.length ||
      document.cohort.input_token_counts.length !== prompts.length ||
      calibration.schema !== 'pllm.public_equalization_build.v1' ||
      calibration.source_lock_digest !== document.model.source_lock_digest ||
      calibration.calibration_file_sha256 !== hash(read(spec.calibration_prompts)) ||
      calibration.calibration_token_counts.length !== calibrationPrompts.length ||
      calibrationPrompts.some((prompt) => prompts.includes(prompt))) {
    throw new Error('Quality/calibration source, held-out cohort or scope differs');
  }
  const names = new Set();
  const rows = document.candidates.map((candidate) => {
    const config = spec.candidates[candidate.name];
    if (!config || names.has(candidate.name) || hash(read(config.file)) !== config.sha256 ||
        candidate.configuration_digest !== config.configuration_digest || candidate.sample_count !== prompts.length) {
      throw new Error('Quality Python configuration or sample identity differs');
    }
    names.add(candidate.name);
    if (![candidate.top1_agreement, candidate.top_k_recall].every((value) => Number.isFinite(value) && value >= 0 && value <= 1) ||
        !Number.isFinite(candidate.max_abs_logit_error) || candidate.max_abs_logit_error < 0) {
      throw new Error('Invalid reference quality measurement');
    }
    if (candidate.public_equalization_profile_digest && (
        candidate.public_equalization_profile_digest !== calibration.profile_digest ||
        candidate.public_calibration_digest !== calibration.calibration_digest ||
        candidate.public_profile_bytes !== calibration.profile_bytes)) {
      throw new Error('Quality profile differs from offline calibration');
    }
    return { ...candidate, ...config };
  });
  if (names.size !== Object.keys(spec.candidates).length ||
      !rows.some((row) => row.public_equalization_profile_digest === calibration.profile_digest)) {
    throw new Error('Quality cohort or calibrated candidate missing');
  }
  return { ...spec, model: document.model, cohort: document.cohort, rows, calibration };
}

export function buildResults() {
  const card = buildScorecard();
  const manifest = JSON.parse(fs.readFileSync(path.join(root, 'docs/data/research/benchmark-cohorts.json')));
  const searches = (manifest.search_reports ?? []).map((spec) => searchResults(spec));
  const quality = (manifest.quality_cohorts ?? []).map((spec) => qualityResults(spec));
  const numeric = quality.map((study) => `## ${study.title}

${study.scope}

${link('Reference-quality report', study.report)} · ${link('Offline calibration report', study.calibration_report)}

${table(['Configuration / Python', 'Samples', 'Top-1 agreement', `Top-${study.cohort.metric.params.top_k} recall`, 'Worst absolute logit error'], study.rows.map((row) => [
  link(row.label, row.file), row.sample_count, number(row.top1_agreement), number(row.top_k_recall), number(row.max_abs_logit_error)]))}

Public profile: **${number(study.calibration.profile_bytes / 1e6)} MB**, ${study.calibration.stage_count} stages,
${study.calibration.calibration_token_counts.length} public calibration sequences.
Offline build: **${number(study.calibration.process_cpu_seconds)} CPU s**,
${number(study.calibration.elapsed_seconds)} elapsed s and ${number(study.calibration.process_peak_rss_bytes / 1e6)} MB process-lifetime peak RSS.
Profile digest: \`${study.calibration.profile_digest}\`.
Publisher work and provider profile distribution are outside response counters; client-delivered scales remain in measured bundles.
These quality observations do not populate the performance chart's unmeasured whole-generation quality metric.
`);
  const implementations = JSON.parse(fs.readFileSync(path.join(root, 'docs/data/research/implementations.json'))).papers;
  const componentReferences = Object.entries(implementations)
    .filter(([, paper]) => paper.status === 'Component reference' && paper.evidence)
    .map(([id, paper]) => ({ id, ...paper }));
  const referenceTable = table(['Paper', 'Scope and observations', 'Replay / evidence'], componentReferences.map((paper) => [
    `[${paper.id}](/research/papers/${paper.id}/)`, `${paper.scope} ${paper.result}`,
    `${link('Python', paper.configuration)} · ${link('Report', paper.evidence)}`]));
  const sections = card.cohorts.map((cohort) => {
    const axes = card.defaultMetrics.map((id) => card.metrics.find((metric) => metric.id === id));
    return `## ${cohort.title}

${cohort.scope}

${link('Canonical report', cohort.report)} · SHA-256: \`${cohort.reportSha256}\`

${table(['Experiment / Python', ...axes.map((metric) => `${metric.shortLabel} (${metric.unit})`)], cohort.rows.map((row) => [
  link(row.label, row.file), ...axes.map((metric) => number(row.metrics[metric.id]))]))}

${table(['Experiment', 'Topology / kernel', 'Public prefix tokens', 'Covered bodies MB/token', 'Decode tokens/s', 'Public artifacts MB'],
  cohort.rows.map((row) => [row.label, `${row.placement.topology} / ${row.placement.kernel}`, row.publicPrefixTokens,
    number(row.metrics.covered), number(row.metrics.decodeTps), number(row.metrics.artifactBytes)]))}

### Complete metric ledger

${table(['Experiment', ...card.metrics.map((metric) => `${metric.shortLabel} (${metric.unit})`)], cohort.rows.map((row) => [
  row.label, ...card.metrics.map((metric) => number(row.metrics[metric.id]))]))}`;
  });
  const exploratory = searches.map((search) => {
    const metric = (trial, id) => trial.metrics.find((item) => item.id === id)?.value;
    return `## ${search.title}

${search.attempted} attempts · ${search.trials.length} measurements · ${search.rejections.length} rejections. Stop: ${search.stopReason}.
Uncapped native exploratory observations; these separate private salts cannot be ranked together or against the confirmed 100/40 cohort.

${link('Complete search evidence, parameters and admission records', search.file)} · SHA-256: \`${search.sha256}\`

${table(['Trial / Python', 'Request tokens/s', 'Request seconds', 'Covered bodies MB', 'Online bodies MB'], search.trials.map((trial) => [
  link(trial.id, trial.file), number(metric(trial, 'request_tps')), number(metric(trial, 'full_seconds')),
  number(metric(trial, 'covered_bytes') == null ? null : metric(trial, 'covered_bytes') / 1e6),
   number(metric(trial, 'online_bytes') == null ? null : metric(trial, 'online_bytes') / 1e6)]))}

### Complete observation metrics

${table(['Trial', ...card.metrics.map((metric) => `${metric.shortLabel} (${metric.unit})`)], search.trials.map((trial) => [
  trial.id, ...card.metrics.map((metric) => number(trial.observations[metric.id]))]))}

${search.rejections.length ? `### Rejected trials\n\n${table(['Attempt', 'Reason', 'Selected parameters'], search.rejections.map((item, index) => [
  index + 1, item.reason, '`' + JSON.stringify(item.parameters) + '`']))}` : 'No rejected trials.'}`;
  });
  const text = `---
title: "Research data"
description: "Every confirmed experiment, every exploratory observation and every rejected SDK-search trial."
---

{/* Generated from checked canonical reports by docs/scripts/research-results.mjs. */}

The [interactive frontier](/research/papers/#research-frontier) uses these confirmed cohorts.
Every score links to its rerunnable Python configuration. Unknown values remain **Unmeasured**.
Read the [reproduction guide](/research/scorecard/) for workload identities, commands and measurement boundaries.

Native 100/40 means shared per-party **100 Mbps download / 40 Mbps upload TCP-stream caps**.
Traffic counts every directed role link once, including HTTP/WebSocket framing. Kernel headers,
retransmissions, telemetry, checkpoint downloads and pre-positioned artifact distribution are excluded.
CPU/network cover startup through the cold response; request tokens/s excludes provider startup.
Client RSS needs a fresh process; total RSS is a same-window sampled sum, not a sum of lifetime peaks.
CPU includes the native relay, but not GPU compute. These single-response samples are not confidence intervals.

**Body offload** is plan-derived: the percentage of declared per-row body-linear MACs
assigned to providers, counting each model stage once. Client-only is 0%; provider-owned
body stages are 100%. Attention, nonlinear work and token boundaries are outside this
fraction. Additional provider CPU or duplicated worker arithmetic cannot raise it.
It describes delegation; the CPU columns measure its actual resource cost.

${sections.join('\n\n')}

# Paper numeric and offline evidence

Changed numeric methods have their own reference-quality evidence. Equal outputs on one cost prompt do not establish numeric equivalence.

${numeric.join('\n\n')}

# Native component references

These bounded references have no executable decoder cohort. Their isolated timings and
payload counts retain their own measurement scope; they do not populate frontier axes.

${referenceTable}

# Exploratory search ledger

All ${searches.reduce((sum, search) => sum + search.attempted, 0)} attempts and ${searches.reduce((sum, search) => sum + search.trials.length, 0)} measurements are retained below.
They explain finalist selection; they are not additional matched confirmation samples.

${exploratory.join('\n\n')}
`;
  return { text, data: { schema: 'pllm.research_results.v1', cohorts: card.cohorts, metrics: card.metrics, searches, quality, componentReferences } };
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const results = buildResults();
  fs.writeFileSync(path.join(root, 'docs/content/docs/research/results.mdx'), results.text);
  fs.writeFileSync(path.join(root, 'docs/data/research/results.json'), JSON.stringify(results.data, null, 2) + '\n');
}
