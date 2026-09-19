import fs from 'node:fs';
import path from 'node:path';
import { buildPublicationGraph, plain, siteRoot } from './content.mjs';

function json(value) {
  return `${JSON.stringify(value, null, 2)}\n`;
}

function absolute(graph, route) {
  return `${graph.canonicalOrigin}${route}`;
}

function pageMarkdown(graph, page) {
  return [
    `# ${page.title}`,
    '',
    page.description,
    '',
    `[View canonical HTML](${absolute(graph, page.canonicalUrl)})`,
    '',
    `Document ID: \`${page.id}\`  `,
    `Release: \`${page.release}\``,
    '',
    page.bodyMarkdown.trim(),
    '',
  ].join('\n');
}

function manifestFor(graph) {
  return {
    schemaVersion: graph.schemaVersion,
    release: graph.release,
    canonicalOrigin: graph.canonicalOrigin,
    pages: graph.pages.map((page) => {
      const record = {
        id: page.id,
        title: page.title,
        summary: page.description,
        sourcePath: page.sourcePath,
        canonicalUrl: page.canonicalUrl,
        markdownUrl: page.markdownUrl,
        kind: page.kind,
        status: page.status,
        release: page.release,
        aliases: page.aliases,
        markdownAliases: page.markdownAliases,
        parent: page.parent,
        children: page.children,
        prerequisites: page.prerequisites,
        related: page.related,
        publicModules: page.publicModules,
        publicSymbols: page.publicSymbols,
        componentIds: page.componentIds,
        sourcePaths: page.sourcePaths,
        testPaths: page.testPaths,
        navigationGroup: page.navigationGroup,
        navigationGroupTitle: page.navigationGroupTitle,
        navigationRoot: page.navigationRoot,
        navigationOrder: page.navigationOrder,
      };
      if (page.citationLinks) record.citationLinks = page.citationLinks;
      if (page.evidenceLinks) record.evidenceLinks = page.evidenceLinks;
      return record;
    }),
  };
}

function llmsIndex(graph, pages = graph.pages) {
  return [
    '# PLLM',
    '',
    `Release: ${graph.release}`,
    '',
    'Private LLM inference documentation. Public weights; provider follows the protocol.',
    '',
    ...pages.map((page) => `- [${page.title}](${absolute(graph, page.canonicalUrl)}) ([Markdown](${absolute(graph, page.markdownUrl)})): ${page.description}`),
    '',
  ].join('\n');
}

export function renderPublicationOutputs(graph = buildPublicationGraph()) {
  const manifest = json(manifestFor(graph));
  const redirects = graph.pages.flatMap((page) => [
    ...page.aliases.map((from) => ({ from, to: page.canonicalUrl, status: 308, representation: 'html' })),
    ...page.markdownAliases.map((from) => ({ from, to: page.markdownUrl, status: 308, representation: 'markdown' })),
  ]);
  const search = graph.pages.map((page) => ({
    id: page.id,
    title: page.title,
    description: page.description,
    url: page.canonicalUrl,
    markdownUrl: page.markdownUrl,
    section: page.navigationGroup ?? (page.canonicalUrl.split('/')[1] || 'home'),
    release: page.release,
    text: plain(page.bodyMarkdown),
  }));
  const sitemap = [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    ...graph.pages.map((page) => `  <url><loc>${absolute(graph, page.canonicalUrl)}</loc></url>`),
    '</urlset>',
    '',
  ].join('\n');
  const versionRoot = `public/releases/${graph.release}`;
  const markdownHeaders = graph.pages.flatMap((page) => [
    page.markdownUrl,
    '  Content-Type: text/markdown; charset=utf-8',
    '  X-Content-Type-Options: nosniff',
    '',
  ]).join('\n');
  const outputs = new Map([
    ['public/_redirects', `${redirects.map((redirect) => `${redirect.from} ${redirect.to} ${redirect.status}`).join('\n')}\n`],
    ['public/_headers', markdownHeaders],
    ['public/docs-manifest.json', manifest],
    [`${versionRoot}/docs-manifest.json`, manifest],
    ['public/redirects.json', json({ schemaVersion: graph.schemaVersion, release: graph.release, redirects })],
    ['public/search-index.json', json(search)],
    ['public/llms.txt', llmsIndex(graph)],
    ['public/llms-full.txt', graph.pages.map((page) => pageMarkdown(graph, page)).join('\n---\n\n')],
    [`${versionRoot}/llms.txt`, llmsIndex(graph)],
    [`${versionRoot}/llms-full.txt`, graph.pages.map((page) => pageMarkdown(graph, page)).join('\n---\n\n')],
    [`${versionRoot}/docs/llms.txt`, llmsIndex(graph, graph.pages.filter((page) => page.id.startsWith('pllm.docs')))],
    [`${versionRoot}/research/llms.txt`, llmsIndex(graph, graph.pages.filter((page) => page.canonicalUrl === '/research' || page.canonicalUrl.startsWith('/research/')))],
    ['public/sitemap.xml', sitemap],
    ['public/robots.txt', `User-agent: *\nAllow: /\n\nSitemap: ${graph.canonicalOrigin}/sitemap.xml\n`],
  ]);
  const docsSections = [...new Set(graph.pages
    .filter((page) => page.id.startsWith('pllm.docs'))
    .map((page) => page.canonicalUrl.match(/^\/([^/]+)\//)?.[1])
    .filter(Boolean))];
  for (const section of docsSections) {
    const pages = graph.pages.filter((page) =>
      page.id.startsWith('pllm.docs') &&
      (page.canonicalUrl === `/${section}/` || page.canonicalUrl.startsWith(`/${section}/`)));
    outputs.set(`${versionRoot}/docs/${section}/llms.txt`, llmsIndex(graph, pages));
  }
  for (const group of new Set(graph.pages.map((page) => page.navigationGroup).filter(Boolean))) {
    const pages = graph.pages.filter((page) => page.navigationGroup === group);
    outputs.set(`${versionRoot}/docs/groups/${group}/llms.txt`, llmsIndex(graph, pages));
  }
  for (const page of graph.pages) {
    outputs.set(`public${page.markdownUrl}`, pageMarkdown(graph, page));
  }
  return outputs;
}

export function writePublicationOutputs(root = siteRoot) {
  const graph = buildPublicationGraph(root);
  const outputs = renderPublicationOutputs(graph);
  try {
    const previous = JSON.parse(fs.readFileSync(path.join(root, 'public/docs-manifest.json'), 'utf8'));
    for (const page of previous.pages ?? []) {
      if (typeof page.markdownUrl !== 'string' || !page.markdownUrl.endsWith('.md')) continue;
      fs.rmSync(path.join(root, 'public', page.markdownUrl), { force: true });
    }
  } catch (error) {
    if (error?.code !== 'ENOENT') throw error;
  }
  for (const relative of ['public/markdown', 'public/releases']) {
    fs.rmSync(path.join(root, relative), { recursive: true, force: true });
  }
  for (const [relative, content] of outputs) {
    const file = path.join(root, relative);
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, content);
  }
  console.log(`Generated ${graph.pages.length} publication records for release ${graph.release}.`);
}

if (process.argv[1] === new URL(import.meta.url).pathname) writePublicationOutputs();
