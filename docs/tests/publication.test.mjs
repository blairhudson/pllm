import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import navigation from '../navigation.json' with { type: 'json' };
import { buildPublicationGraph, siteRoot, walk } from '../scripts/content.mjs';
import { renderPublicationOutputs } from '../scripts/generate.mjs';

const graph = buildPublicationGraph();
const outputs = renderPublicationOutputs(graph);
const manifest = JSON.parse(outputs.get('public/docs-manifest.json'));
const redirects = JSON.parse(outputs.get('public/redirects.json')).redirects;
const generatedPath = (relative) => path.join(siteRoot, relative);

test('registry covers every public authored page exactly once', () => {
  const discovered = [
    ...walk(path.join(siteRoot, 'content/docs')).filter((file) => file.endsWith('.mdx')),
    ...walk(path.join(siteRoot, 'content/research')).filter((file) => file.endsWith('.mdx')),
    ...['content/home.html', 'content/research.html'].map(generatedPath),
  ].map((file) => path.relative(siteRoot, file).replaceAll(path.sep, '/')).sort();
  assert.deepEqual(graph.pages.map((page) => page.sourcePath).sort(), discovered);
});

test('generated CLI command pages retain source and test provenance', () => {
  const cliPages = graph.pages.filter((page) => page.sourcePath.startsWith('content/docs/reference/cli/'));
  assert.ok(cliPages.length > 1);
  for (const page of cliPages) {
    assert.deepEqual(page.sourcePaths, [page.sourcePath, '../python/pllm/_cli/app.py']);
    assert.deepEqual(page.testPaths, ['../tests/test_cli.py', '../tests/test_developer_reference.py']);
  }
});

test('publication identities, canonical routes, and Markdown twins are unique', () => {
  for (const field of ['id', 'sourcePath', 'canonicalUrl', 'markdownUrl']) assert.equal(new Set(graph.pages.map((page) => page[field])).size, graph.pages.length, field);
  const aliases = graph.pages.flatMap((page) => page.aliases);
  const markdownAliases = graph.pages.flatMap((page) => page.markdownAliases);
  assert.equal(new Set(aliases).size, aliases.length);
  assert.equal(new Set(markdownAliases).size, markdownAliases.length);
  const canonical = new Set(graph.pages.map((page) => page.canonicalUrl));
  const markdown = new Set(graph.pages.map((page) => page.markdownUrl));
  assert.ok(aliases.every((alias) => !canonical.has(alias)));
  assert.ok(markdownAliases.every((alias) => !markdown.has(alias)));
  for (const page of graph.pages) {
    assert.match(page.id, /^pllm\.[a-z0-9.-]+$/);
    assert.ok(page.canonicalUrl.endsWith('/'));
    assert.doesNotMatch(page.canonicalUrl, /\/index(?:\/|$)/);
    assert.equal(page.markdownUrl, page.canonicalUrl === '/'
      ? '/index.md'
      : `${page.canonicalUrl.slice(0, -1)}.md`);
    assert.deepEqual(page.aliases, page.canonicalUrl === '/' ? [] : [page.canonicalUrl.slice(0, -1)]);
    assert.deepEqual(page.markdownAliases, []);
  }
});

test('manifest is complete and versioned without generated hashes', () => {
  assert.equal(manifest.schemaVersion, '2.0.0');
  assert.ok(!Object.hasOwn(graph, 'buildId'));
  assert.ok(!Object.hasOwn(manifest, 'buildId'));
  assert.equal(manifest.pages.length, graph.pages.length);
  for (const record of manifest.pages) {
    for (const field of ['id', 'title', 'summary', 'sourcePath', 'canonicalUrl', 'markdownUrl', 'kind', 'status', 'release', 'aliases', 'markdownAliases', 'parent', 'children', 'prerequisites', 'related', 'publicModules', 'publicSymbols', 'componentIds', 'sourcePaths', 'testPaths', 'navigationGroup', 'navigationGroupTitle', 'navigationRoot', 'navigationOrder']) assert.ok(Object.hasOwn(record, field), `${record.id}.${field}`);
    assert.ok(!Object.hasOwn(record, 'contentHash'));
    assert.ok(fs.existsSync(generatedPath(record.sourcePath)));
  }
  assert.ok(graph.pages.every((page) => !Object.hasOwn(page, 'contentHash')));
  const search = JSON.parse(outputs.get('public/search-index.json'));
  assert.ok(search.every((record) => !Object.hasOwn(record, 'contentHash')));
  for (const [relative, content] of outputs) {
    if (!relative.endsWith('.md') && !relative.endsWith('.txt')) continue;
    assert.doesNotMatch(content, /^(?:Build|Source hash):/m, relative);
  }
  assert.ok(manifest.pages.find((record) => record.id === 'pllm.docs.reference.python.pllm').publicModules.includes('pllm.runtime'));
  assert.deepEqual(manifest.pages.find((record) => record.id === 'pllm.docs.reference.components').componentIds, ['pllm/bfv-correlations/v1', 'pllm/binary-table/v1', 'pllm/blinded-linear/v1', 'pllm/chunked-independent-lanes/v1', 'pllm/cleartext-linear', 'pllm/client-local-kv', 'pllm/cpu', 'pllm/direct-fhe', 'pllm/guarded-linear/v1', 'pllm/he-authenticated-preprocessing', 'pllm/independent-lanes/v1', 'pllm/inference', 'pllm/kv-cache-eviction', 'pllm/linear-integrity', 'pllm/masked-linear', 'pllm/model-aware-corrections', 'pllm/r03-crt/v1', 'pllm/scalar/v1', 'pllm/secure-linear/v1', 'pllm/seeded-expansion']);
  assert.ok(!manifest.pages.some((record) => record.canonicalUrl.startsWith('/cli/reference/research/')));
  assert.ok(!manifest.pages.some((record) => record.canonicalUrl === '/research/records/method-catalog/'));
  assert.ok(!manifest.pages.find((record) => record.id === 'pllm.docs.reference.python.pllm').publicModules.includes('pllm.research'));
  assert.deepEqual(JSON.parse(outputs.get(`public/releases/${graph.release}/docs-manifest.json`)), manifest);
});

test('parent-child topology is exact and acyclic', () => {
  const byId = new Map(graph.pages.map((page) => [page.id, page]));
  for (const page of graph.pages) {
    if (page.id === 'pllm.home') assert.equal(page.parent, null);
    else {
      assert.ok(byId.has(page.parent), `${page.id}.parent`);
      assert.ok(byId.get(page.parent).children.includes(page.id), `${page.id}.child`);
    }
    const seen = new Set([page.id]);
    let current = page;
    while (current.parent) {
      assert.ok(!seen.has(current.parent), page.id);
      seen.add(current.parent);
      current = byId.get(current.parent);
    }
  }
});

test('publication and search records follow canonical navigation groups', () => {
  const search = JSON.parse(outputs.get('public/search-index.json'));
  const searchById = new Map(search.map((record) => [record.id, record]));
  const groupIds = new Set(navigation.publicationGroups.map((group) => group.id));
  for (const page of graph.pages.filter((candidate) => candidate.canonicalUrl.startsWith('/') && candidate.canonicalUrl !== '/')) {
    assert.ok(groupIds.has(page.navigationGroup), `${page.id}.navigationGroup`);
    assert.equal(searchById.get(page.id).section, page.navigationGroup);
    assert.ok(page.navigationRoot.startsWith('/'));
    assert.equal(typeof page.navigationOrder, 'number');
  }
  for (const group of groupIds) {
    assert.ok(outputs.has(`public/releases/${graph.release}/docs/groups/${group}/llms.txt`), group);
  }
});

test('redirects only normalize canonical HTML trailing slashes', () => {
  assert.deepEqual(redirects, graph.pages
    .filter((page) => page.canonicalUrl !== '/')
    .map((page) => ({
      from: page.canonicalUrl.slice(0, -1),
      to: page.canonicalUrl,
      status: 308,
      representation: 'html',
    })));
});

test('generated surfaces are fresh and managed directories contain no orphans', () => {
  for (const [relative, expected] of outputs) assert.equal(fs.readFileSync(generatedPath(relative), 'utf8'), expected, relative);
  const markdownFiles = manifest.pages.map((page) => `public${page.markdownUrl}`);
  const managed = [...markdownFiles, ...walk(generatedPath('public/releases')).map((file) => path.relative(siteRoot, file).replaceAll(path.sep, '/'))].sort();
  const expected = [...outputs.keys()].filter((file) => markdownFiles.includes(file) || file.startsWith('public/releases/')).sort();
  assert.deepEqual(managed, expected);
});

test('search, root llms, and hierarchical release indexes cover graph', () => {
  const search = JSON.parse(outputs.get('public/search-index.json'));
  assert.deepEqual(search.map((record) => record.id), graph.pages.map((page) => page.id));
  const llms = outputs.get('public/llms.txt');
  for (const page of graph.pages) {
    assert.ok(llms.includes(`${graph.canonicalOrigin}${page.canonicalUrl}`), page.id);
    assert.ok(llms.includes(`${graph.canonicalOrigin}${page.markdownUrl}`), page.id);
  }
  const docsSections = new Set(graph.pages
    .filter((page) => page.id.startsWith('pllm.docs'))
    .map((page) => page.canonicalUrl.split('/')[1])
    .filter(Boolean));
  for (const section of docsSections) assert.ok(outputs.has(`public/releases/${graph.release}/docs/${section}/llms.txt`), section);
});

test('Markdown twins are readable, complete, and free of MDX or HTML artifacts', () => {
  for (const page of graph.pages) {
    const markdown = outputs.get(`public${page.markdownUrl}`);
    assert.ok(markdown.includes(`[View canonical HTML](${graph.canonicalOrigin}${page.canonicalUrl})`), page.id);
    assert.ok(markdown.includes(`Document ID: \`${page.id}\``), page.id);
    assert.ok(markdown.length > page.description.length + 150, page.id);
    assert.doesNotMatch(markdown, /<(?:div|table|section|article)\b|className=|^import .+ from ['"]/m, page.id);
  }
});

test('sitemap, robots, headers, and redirects derive from registry', () => {
  const locations = [...outputs.get('public/sitemap.xml').matchAll(/<loc>([^<]+)<\/loc>/g)].map((match) => match[1]);
  assert.deepEqual(locations, graph.pages.map((page) => `${graph.canonicalOrigin}${page.canonicalUrl}`));
  assert.equal(new Set(locations).size, graph.pages.length);
  assert.ok(redirects.every((redirect) => redirect.status === 308));
  assert.equal(outputs.get('public/_redirects'), `${redirects.map((redirect) => `${redirect.from} ${redirect.to} 308`).join('\n')}\n`);
  const markdownHeaders = graph.pages.flatMap((page) => [
    page.markdownUrl,
    '  Content-Type: text/markdown; charset=utf-8',
    '  X-Content-Type-Options: nosniff',
    '',
  ]).join('\n');
  assert.equal(outputs.get('public/_headers'), markdownHeaders);
});
