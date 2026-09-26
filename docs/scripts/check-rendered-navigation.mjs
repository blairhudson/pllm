import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const siteRoot = fileURLToPath(new URL('../', import.meta.url));

function rendered(route) {
  const page = fs.readFileSync(path.join(siteRoot, 'out', route, 'index.html'), 'utf8');
  const sidebarStart = page.indexOf('id="nd-sidebar"');
  assert.ok(sidebarStart >= 0, `Missing rendered sidebar: /${route}/`);
  const sidebarEnd = page.indexOf('</aside>', sidebarStart);
  assert.ok(sidebarEnd > sidebarStart, `Unclosed rendered sidebar: /${route}/`);
  return { page, sidebar: page.slice(sidebarStart, sidebarEnd) };
}

const learn = rendered('learn');
const learnSections = ['Start', 'Core concepts', 'Working with PLLM'];
let previous = -1;
for (const label of learnSections) {
  const position = learn.sidebar.indexOf(`>${label}<`);
  assert.ok(position > previous, `Learn section missing or out of order: ${label}`);
  previous = position;
}

const sdk = rendered('sdk');
const sdkSections = ['SDK', 'Experiments', 'Models', 'Inference options',
  'Component options', 'Run PLLM', 'Plans', 'Evaluate', 'Extend PLLM', 'API reference'];
previous = -1;
for (const label of sdkSections) {
  const position = sdk.sidebar.indexOf(`>${label}<`);
  assert.ok(position > previous, `SDK section missing or out of order: ${label}`);
  previous = position;
}
assert.doesNotMatch(sdk.sidebar, /href="\/sdk\/(?:build|pipeline|research|operate|contribute)(?:\/|\")/);

const { page, sidebar } = rendered('research/papers');
const papers = JSON.parse(fs.readFileSync(path.join(siteRoot, 'data/research/papers.json'), 'utf8'));
const library = JSON.parse(fs.readFileSync(path.join(siteRoot, 'data/research/paper-library.json'), 'utf8'));
const byId = new Map(papers.papers.map((paper) => [paper.id, paper]));
const expected = library.papers
  .sort((left, right) => right.year - left.year || left.title.localeCompare(right.title))
  .map((paper) => byId.get(paper.registry_id)?.slug.replaceAll('_', '-') ?? paper.id);
const actual = [...sidebar.matchAll(/href="\/research\/papers\/([a-z0-9-]+)\/"/g)]
  .map((match) => match[1]);
assert.deepEqual(actual, expected, 'Research paper sidebar must list all cited papers newest first');
assert.equal((page.match(/class="paper-timeline-entry"/g) ?? []).length, expected.length);
const titles = [...page.matchAll(/<div class="paper-timeline-title">([\s\S]*?)<\/div>/g)];
assert.equal(titles.length, expected.length, 'Each timeline entry needs one linked title');
for (const title of titles) {
  assert.equal((title[1].match(/<a\b/g) ?? []).length, 1, 'Timeline title must contain exactly one link');
  assert.match(title[1], /href="\/research\/papers\/[a-z0-9-]+\/"/);
}
for (const heading of page.matchAll(/<h3\b[^>]*>([\s\S]*?)<\/h3>/g)) {
  assert.doesNotMatch(heading[1], /href="\/research\/papers\//,
    'A linked paper title inside an auto-linked heading creates nested anchors');
}
assert.ok(page.includes('class="paper-timeline"'), 'Paper timeline did not render');
assert.ok(!page.includes('Back to the chronological bibliography'));

const whitepaper = rendered('research/whitepaper').page;
assert.equal((whitepaper.match(/<figure>/g) ?? []).length, 3);
for (const name of ['mechanics', 'research-loop', 'qwen-baseline']) {
  assert.ok(whitepaper.includes(`src="/downloads/figures/${name}.png"`),
    `Canonical whitepaper figure did not render: ${name}`);
}
assert.ok(whitepaper.includes('Economic value needs an honest denominator'));

const experimentGuide = rendered('sdk/experiments').page;
const exampleCode = [...experimentGuide.matchAll(/<code\b[^>]*>([\s\S]*?)<\/code>/g)].map((match) => match[1]);
assert.ok(exampleCode.some((code) => /<a\b(?=[^>]*data-import-link)(?=[^>]*href="\/sdk\/reference\/python\/pllm\/)[^>]*>/.test(code)),
  'Python imports must link to API entries inline inside rendered code');
assert.doesNotMatch(experimentGuide, /data-import-links|Imports: <a/, 'Do not duplicate links under code examples');

const apiReference = rendered('sdk/reference/python/pllm').page;
assert.ok(apiReference.includes('href="/sdk/experiments/"'), 'Experiment API needs a user-guide backlink');
assert.ok(apiReference.includes('href="/sdk/models/"'), 'Model API needs a user-guide backlink');

const status = rendered('sdk/reference/status').page;
const matrixStart = status.indexOf('aria-label="Model family and reusable capability matrix"');
assert.ok(matrixStart >= 0, 'Model family matrix missing from rendered status');
const matrixEnd = status.indexOf('</table>', matrixStart);
assert.ok(matrixEnd > matrixStart, 'Model family matrix did not render as a table');
const matrix = status.slice(matrixStart, matrixEnd);
assert.match(status.slice(matrixStart - 200, matrixStart), /overflow-x-auto/,
  'Wide capability matrix must scroll inside its own region on mobile');
assert.equal((matrix.match(/<th scope="row"/g) ?? []).length, 12,
  'Checked and candidate families must share one rendered matrix');
assert.equal((matrix.match(/<th scope="col"/g) ?? []).length, 34);
assert.match(matrix, /sticky left-0/, 'Family names must remain visible while scrolling');
assert.match(matrix, /Phi-4-mini-instruct/);
assert.match(matrix, /aria-label="Qwen3.5-4B text decoder: Gated-delta recurrence: missing required executable variant"/);
const moduleStart = status.indexOf('Python SDK completeness', matrixEnd);
assert.ok(moduleStart > matrixEnd, 'SDK completeness must follow the family matrix on one page');
assert.match(status.slice(moduleStart), /Paper stubs/);
assert.match(status.slice(moduleStart), /Model capability stubs/);
assert.ok(status.indexOf('</table>', moduleStart) > moduleStart,
  'SDK completeness must render as the second table');
const capability = rendered('sdk/models/capabilities/gated-delta').page;
assert.match(capability, /GatedDeltaRecurrence/);
assert.match(capability, /ModelCapabilityUnavailable/);
assert.match(capability, /href="\/sdk\/reference\/status\/"/);

console.log(`Rendered navigation, inline import links, API backlinks, and ${expected.length} Research paper links passed.`);
