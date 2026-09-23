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

console.log(`Rendered Learn and SDK sections and ${expected.length} Research paper links are ordered correctly.`);
