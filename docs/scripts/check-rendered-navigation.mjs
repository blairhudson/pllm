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

const { page, sidebar } = rendered('research/papers');
const papers = JSON.parse(fs.readFileSync(path.join(siteRoot, 'data/research/papers.json'), 'utf8'));
const byId = new Map(papers.papers.map((paper) => [paper.id, paper]));
const expected = papers.public_bibliography_ids.map((id) => byId.get(id))
  .sort((left, right) => right.year - left.year || left.title.localeCompare(right.title))
  .map((paper) => paper.slug.replaceAll('_', '-'));
const actual = [...sidebar.matchAll(/href="\/research\/papers\/([a-z0-9-]+)\/"/g)]
  .map((match) => match[1]);
assert.deepEqual(actual, expected, 'Research paper sidebar must list all cited papers newest first');
assert.equal((page.match(/class="paper-timeline-entry"/g) ?? []).length, expected.length);
assert.ok(page.includes('class="paper-timeline"'), 'Paper timeline did not render');
assert.ok(!page.includes('Back to the chronological bibliography'));

console.log(`Rendered Learn sections and ${expected.length} Research paper links are ordered correctly.`);
