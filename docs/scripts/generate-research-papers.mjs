import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const dataRoot = fileURLToPath(new URL('../data/research/', import.meta.url));
const outputRoot = fileURLToPath(new URL('../content/docs/research/papers/', import.meta.url));
const repository = 'https://github.com/blairhudson/pllm/blob/main';
const allowedStatuses = new Set([
  'PLLM reimplementation',
  'PLLM component adaptation',
  'reference primitive only',
  'planned',
  'review',
  'tracked',
]);

function statusFor(paper) {
  if (['R01', 'R02'].includes(paper.id)) return 'reference primitive only';
  if (paper.id === 'R03') return 'PLLM component adaptation';
  if (paper.id === 'R23') return 'PLLM reimplementation';
  if (paper.id === 'R24') return 'planned';
  return 'tracked';
}

function outputSlug(paper) {
  return `${paper.id.toLowerCase()}-${paper.slug.replaceAll('_', '-')}`;
}

function readNoteSection(paper, heading) {
  if (!paper.card) return null;
  const notePath = path.join(dataRoot, paper.card);
  if (!fs.existsSync(notePath)) return null;
  const note = fs.readFileSync(notePath, 'utf8');
  const marker = `## ${heading}\n`;
  const start = note.indexOf(marker);
  if (start === -1) return null;
  const section = note.slice(start + marker.length);
  const end = section.search(/^## /m);
  return section.slice(0, end === -1 ? undefined : end).trim() || null;
}

function perspectiveFor(paper) {
  const notePerspective = readNoteSection(paper, 'PLLM perspective');
  const context = notePerspective ?? `${paper.proposed_implementation}\n\n${paper.prior_evidence}`;
  const boundary = `The source's boundary remains binding: ${paper.security_boundary}`;

  if (paper.id === 'R23') {
    return `${context}\n\nPLLM implements only a structural \`DecoderPlan\` adaptation. It does not reproduce MPCache's three-party protected runtime. Fidelity, protected execution, assurance, and matched benchmark status remain unchecked.\n\n${boundary}`;
  }
  if (paper.id === 'R24') {
    return `${context}\n\nThis work is planned. PLLM does not yet implement Maverick's delegation, LPN masking, or batch verification, and claims no reproduction or matched benchmark.\n\n${boundary}`;
  }
  if (paper.id === 'R03') {
    return `${context}\n\nPLLM uses the source-locked projection and CRT constructions in a clean-room compact scalar Q7 program. This is a scoped component adaptation, not a paper reproduction, reviewed protected runtime, tensor method, or complete-model result.\n\n${boundary}`;
  }
  if (['R01', 'R02'].includes(paper.id)) {
    return `${context}\n\nPLLM has related clean-room reference primitives only. They are not a paper reproduction, reviewed protected runtime, or complete-model result.\n\n${boundary}`;
  }
  return `${context}\n\nThis paper remains tracked. Named modules below are implementation targets, not finished PLLM components; no reproduction, integration, assurance, or matched benchmark is claimed.\n\n${boundary}`;
}

function documentationLinks(paper) {
  const links = [
    '- [Research backlog](/research/backlog/)',
    '- [Reimplementation backlog](/research/backlog/)',
    '- [Clean-room workflow](/research/clean-room/)',
  ];
  if (['R01', 'R02', 'R03'].includes(paper.id)) {
    links.push(`- [Arithmetic garbling documentation](/sdk/pipeline/protocols/garbling/arithmetic/)`);
    links.push(`- [Reference garbling source](${repository}/crates/pllm-garble/src/lib.rs)`);
  }
  if (paper.id === 'R23') {
    links.push('- [PLLM method status](/research/methods/)');
    links.push('- [Decoder plan usage](/sdk/plans/)');
    links.push(`- [MPCache structural adaptation source](${repository}/crates/pllm-models/src/cache.rs)`);
  }
  if (paper.id === 'R24') links.push('- [Masked linear protocol](/sdk/pipeline/protocols/masked-linear/)');
  return links.join('\n');
}

function renderPaper(paper, phase) {
  const status = statusFor(paper);
  const dependencies = paper.depends_on.length ? paper.depends_on.join(', ') : 'None';
  return `---
title: ${JSON.stringify(`${paper.id}: ${paper.title}`)}
description: ${JSON.stringify(`${paper.source_summary} PLLM status: ${status}.`)}
---

**Status:** \`${status}\`  
**Year:** ${paper.year}  
**Authors:** ${paper.authors.join(', ')}  
**External source:** [${paper.title}](${paper.primary_url})

## Source contribution

${paper.source_summary}

## PLLM perspective

${perspectiveFor(paper)}

## PLLM modules and documentation

Target module identifiers: ${paper.modules.map((module) => `\`${module}\``).join(', ')}. These identifiers describe scoped work unless this page explicitly links an implementation source. Dependencies: ${dependencies}.

${documentationLinks(paper)}

## Backlog

This source is assigned to reimplementation priority ${phase.priority}, **${phase.name}**. Its source-registry order is ${paper.priority}. Follow current gates and unfinished work in the [research backlog](/research/backlog/).
`;
}

function renderIndex(registry, plan, papers) {
  const rows = papers.map((paper) => {
    const slug = outputSlug(paper);
    return `| ${paper.priority} | [${paper.id}](/research/papers/${slug}/) | ${paper.title} | \`${statusFor(paper)}\` |`;
  });
  const paperById = new Map(papers.map((paper) => [paper.id, paper]));
  const phaseRows = plan.phases.map((phase) => {
    const links = phase.papers.map((id) => {
      const paper = paperById.get(id);
      return `[${id}](/research/papers/${outputSlug(paper)}/)`;
    });
    return `| ${phase.priority} | ${phase.name} | ${links.join(', ')} | \`${phase.status}\` |`;
  });
  const phaseDetails = plan.phases.map((phase) => `### ${phase.priority}. ${phase.name}

**Status:** \`${phase.status}\`  
**Papers:** ${phase.papers.map((id) => {
    const paper = paperById.get(id);
    return `[${id}](/research/papers/${outputSlug(paper)}/)`;
  }).join(', ')}

${phase.purpose}

Module homes:

${phase.module_homes.map((module) => `- \`${module}\``).join('\n')}`).join('\n\n');
  return `---
title: "Research papers"
description: "Prioritized paper reimplementation plan, source status, contribution summaries, and PLLM implementation boundaries."
---

This catalog is generated from the canonical research registry as of ${registry.as_of}. A tracked source, target module, or similar data flow is not evidence of implementation, reproduction, security, or benchmark parity.

R03 has a scoped clean-room component adaptation in the compact scalar Q7 program. R23 has a structural \`DecoderPlan\` adaptation. Neither is a paper reproduction or protected-runtime result. R01-R02 remain reference primitives only, and R24 is planned.

## Reimplementation plan

The execution order is dependency-driven rather than paper-number order. Work starts only after these prerequisites:

${plan.prerequisites.map((prerequisite, index) => `${index + 1}. ${prerequisite}`).join('\n')}

Plan status as of ${plan.as_of}:

| Priority | Capability track | Papers | Track status |
| ---: | --- | --- | --- |
${phaseRows.join('\n')}

${phaseDetails}

Paper names remain provenance. Implementations live in capability-oriented modules so compatible alternatives can be composed and benchmarked through the same plan slots.

## Source registry

| Source order | ID | Paper | Implementation status |
| ---: | --- | --- | --- |
${rows.join('\n')}

See the [research backlog](/research/backlog/) for cross-paper gates and current unfinished work.
`;
}

function validate(registry, plan) {
  if (registry.schema_version !== 'pllm.reproduction_portfolio.v2') {
    throw new Error(`Unsupported paper registry schema: ${registry.schema_version}`);
  }
  if (!Array.isArray(registry.papers) || registry.papers.length !== 24) {
    throw new Error('Paper registry must contain exactly 24 papers');
  }
  const ids = registry.papers.map((paper) => paper.id);
  const expected = Array.from({ length: 24 }, (_, index) => `R${String(index + 1).padStart(2, '0')}`);
  if (ids.join(',') !== expected.join(',')) throw new Error(`Expected ordered IDs ${expected.join(', ')}`);
  if (new Set(registry.papers.map(outputSlug)).size !== registry.papers.length) {
    throw new Error('Paper output slugs must be unique');
  }
  for (const paper of registry.papers) {
    if (!allowedStatuses.has(statusFor(paper))) throw new Error(`Invalid status for ${paper.id}`);
    if (!paper.primary_url.startsWith('https://')) throw new Error(`Invalid external source for ${paper.id}`);
    if (!Array.isArray(paper.modules) || !paper.modules.length) throw new Error(`Missing modules for ${paper.id}`);
  }
  if (plan.schema_version !== 'pllm.reimplementation_plan.v1') {
    throw new Error(`Unsupported reimplementation plan schema: ${plan.schema_version}`);
  }
  if (!Array.isArray(plan.prerequisites) || !plan.prerequisites.length) {
    throw new Error('Reimplementation plan must define prerequisites');
  }
  if (!Array.isArray(plan.phases) || !plan.phases.length) {
    throw new Error('Reimplementation plan must define phases');
  }
  const plannedIds = [];
  for (const [index, phase] of plan.phases.entries()) {
    if (phase.priority !== index + 1) throw new Error('Reimplementation priorities must be contiguous');
    if (!phase.name || !phase.purpose || !phase.status) throw new Error(`Incomplete phase ${phase.priority}`);
    if (!Array.isArray(phase.papers) || !phase.papers.length) throw new Error(`Missing papers for phase ${phase.priority}`);
    if (!Array.isArray(phase.module_homes) || !phase.module_homes.length) throw new Error(`Missing modules for phase ${phase.priority}`);
    plannedIds.push(...phase.papers);
  }
  if (plannedIds.length !== ids.length || new Set(plannedIds).size !== ids.length) {
    throw new Error('Each research paper must appear in exactly one reimplementation phase');
  }
  for (const id of ids) {
    if (!plannedIds.includes(id)) throw new Error(`Research paper ${id} is absent from the reimplementation plan`);
  }
}

const registry = JSON.parse(fs.readFileSync(path.join(dataRoot, 'papers.json'), 'utf8'));
const plan = JSON.parse(fs.readFileSync(path.join(dataRoot, 'reimplementation-plan.json'), 'utf8'));
validate(registry, plan);
fs.mkdirSync(outputRoot, { recursive: true });
const phaseByPaper = new Map(plan.phases.flatMap((phase) => phase.papers.map((id) => [id, phase])));

const generated = new Map([
  ['index.mdx', renderIndex(registry, plan, registry.papers)],
  ['meta.json', `${JSON.stringify({
    title: 'Research papers',
    root: true,
    pages: ['index', ...registry.papers.map(outputSlug)],
  }, null, 2)}\n`],
  ...registry.papers.map((paper) => [`${outputSlug(paper)}.mdx`, renderPaper(paper, phaseByPaper.get(paper.id))]),
]);

for (const entry of fs.readdirSync(outputRoot, { withFileTypes: true })) {
  if (entry.isFile() && /\.(?:mdx|json)$/.test(entry.name) && !generated.has(entry.name)) {
    fs.rmSync(path.join(outputRoot, entry.name));
  }
}
for (const [filename, content] of generated) {
  const target = path.join(outputRoot, filename);
  if (!fs.existsSync(target) || fs.readFileSync(target, 'utf8') !== content) fs.writeFileSync(target, content);
}

console.log(`Generated ${registry.papers.length} research paper pages and index.`);
