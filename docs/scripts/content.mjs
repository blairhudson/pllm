import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import navigation from '../navigation.json' with { type: 'json' };
import { publicationRegistry } from './publication-registry.mjs';

export const siteRoot = fileURLToPath(new URL('../', import.meta.url));

export function walk(dir) {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((entry) =>
    entry.isDirectory() ? walk(path.join(dir, entry.name)) : [path.join(dir, entry.name)]);
}

function parseMdx(raw, file) {
  const match = raw.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
  if (!match) throw new Error(`Missing frontmatter: ${file}`);
  const field = (name) => {
    const value = match[1].match(new RegExp(`^${name}: (.+)$`, 'm'))?.[1];
    if (!value) throw new Error(`Missing ${name}: ${file}`);
    return JSON.parse(value);
  };
  return { title: field('title'), description: field('description'), content: match[2] };
}

function decodeHtml(value) {
  return value
    .replace(/&(?:nbsp|#160);/gi, ' ')
    .replace(/&amp;/gi, '&')
    .replace(/&lt;/gi, '<')
    .replace(/&gt;/gi, '>')
    .replace(/&quot;/gi, '"')
    .replace(/&#39;|&apos;/gi, "'");
}

function stripTags(value) {
  return decodeHtml(value.replace(/<br\s*\/?>/gi, ' ').replace(/<[^>]+>/g, '')).replace(/\s+/g, ' ').trim();
}

function convertTable(table) {
  const caption = stripTags(table.match(/<caption\b[^>]*>([\s\S]*?)<\/caption>/i)?.[1] ?? '');
  const rows = [...table.matchAll(/<tr\b[^>]*>([\s\S]*?)<\/tr>/gi)].map((row) =>
    [...row[1].matchAll(/<t[hd]\b[^>]*>([\s\S]*?)<\/t[hd]>/gi)].map((cell) => stripTags(cell[1])));
  if (!rows.length || !rows[0].length) return stripTags(table);
  const width = Math.max(...rows.map((row) => row.length));
  const normalized = rows.map((row) => [...row, ...Array(width - row.length).fill('')]);
  return `\n${caption ? `**${caption}**\n\n` : ''}${normalized.map((row, index) =>
    `| ${row.join(' | ')} |${index === 0 ? `\n| ${row.map(() => '---').join(' | ')} |` : ''}`).join('\n')}\n`;
}

function htmlFragmentToMarkdown(source) {
  const codeBlocks = [];
  let value = source
    .replace(/<svg\b[^>]*>[\s\S]*?<\/svg>/gi, (svg) => {
      const title = stripTags(svg.match(/<title\b[^>]*>([\s\S]*?)<\/title>/i)?.[1] ?? 'Diagram');
      const description = stripTags(svg.match(/<desc\b[^>]*>([\s\S]*?)<\/desc>/i)?.[1] ?? '');
      return `\n> Diagram: ${title}${description ? `. ${description}` : ''}\n`;
    })
    .replace(/<table\b[^>]*>[\s\S]*?<\/table>/gi, convertTable)
    .replace(/<pre\b[^>]*>([\s\S]*?)<\/pre>/gi, (_match, code) => {
      const token = `\u0000CODE${codeBlocks.length}\u0000`;
      codeBlocks.push(`\`\`\`text\n${decodeHtml(code.replace(/<[^>]+>/g, '')).trim()}\n\`\`\``);
      return `\n${token}\n`;
    })
    .replace(/<img\b[^>]*alt=["']([^"']*)["'][^>]*src=["']([^"']+)["'][^>]*\/?>/gi, '![$1]($2)')
    .replace(/<img\b[^>]*src=["']([^"']+)["'][^>]*alt=["']([^"']*)["'][^>]*\/?>/gi, '![$2]($1)')
    .replace(/<button\b[^>]*data-copy[^>]*>[\s\S]*?<\/button>/gi, '')
    .replace(/<button\b[^>]*>([\s\S]*?)<\/button>/gi, (_match, label) => `\n**${stripTags(label)}**\n`)
    .replace(/<\/a>\s*<a\b/gi, '</a>\n<a')
    .replace(/<a\b[^>]*href=["']([^"']+)["'][^>]*>([\s\S]*?)<\/a>/gi, (_match, href, label) => `[${stripTags(label)}](${href})`)
    .replace(/<h([1-6])\b[^>]*>([\s\S]*?)<\/h\1>/gi, (_match, level, text) => `\n${'#'.repeat(Number(level))} ${stripTags(text)}\n`)
    .replace(/<li\b[^>]*>([\s\S]*?)<\/li>/gi, (_match, text) => `\n- ${stripTags(text)}`)
    .replace(/<(strong|b)\b[^>]*>([\s\S]*?)<\/\1>/gi, '**$2** ')
    .replace(/<(em|i)\b[^>]*>([\s\S]*?)<\/\1>/gi, '*$2*')
    .replace(/<code\b[^>]*>([\s\S]*?)<\/code>/gi, '`$1`')
    .replace(/<sup\b[^>]*>([\s\S]*?)<\/sup>/gi, '$1')
    .replace(/<br\s*\/?>/gi, '\n')
    .replace(/<p\b[^>]*>([\s\S]*?)<\/p>/gi, '\n$1\n')
    .replace(/<\/?(?:article|div|figure|header|nav|ol|section|ul)\b[^>]*>/gi, '\n')
    .replace(/<[^>]+>/g, '');
  value = decodeHtml(value)
    .replace(/[ \t]+\n/g, '\n')
    .replace(/\n[ \t]+/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
  value = value.replace(/\u0000CODE(\d+)\u0000/g, (_match, index) => codeBlocks[Number(index)]);
  return `${value}\n`;
}

function mdxToMarkdown(source) {
  const chunks = source.split(/(```[\s\S]*?```)/g);
  return chunks.map((chunk, index) => index % 2 ? chunk.trim() : htmlFragmentToMarkdown(chunk).trim())
    .filter(Boolean).join('\n\n') + '\n';
}

export function markdownPathForRoute(route) {
  const canonical = route === '/' ? '/' : `${route.replace(/\/$/, '')}/`;
  const page = publicationRegistry.pages.find((candidate) => candidate.canonicalUrl === canonical);
  if (!page) throw new Error(`Unregistered canonical route: ${route}`);
  return page.markdownUrl;
}

function titleForHtml(id, content) {
  if (id === 'pllm.home') return 'PLLM — Private LLM Inference';
  const title = stripTags(content.match(/<h1\b[^>]*>([\s\S]*?)<\/h1>/i)?.[1] ?? '');
  if (!title) throw new Error(`Missing h1: ${id}`);
  return id === 'pllm.research' ? 'PLLM research' : title;
}

function descriptionForHtml(id, content) {
  if (id === 'pllm.home') return 'Private language model inference. Client and server guides, the protocol, and reproducible research.';
  if (id === 'pllm.research') return 'PLLM original research, technical paper, tracked papers, and implementation backlog.';
  throw new Error(`Missing HTML description: ${id}`);
}

function nearestParent(page, pages) {
  if (page.canonicalUrl === '/') return null;
  const candidates = pages.filter((candidate) => candidate.canonicalUrl !== page.canonicalUrl &&
    (candidate.canonicalUrl === '/' || page.canonicalUrl.startsWith(candidate.canonicalUrl)));
  return candidates.sort((left, right) => right.canonicalUrl.length - left.canonicalUrl.length)[0]?.id ?? 'pllm.home';
}

export function navigationForRoute(route) {
  const groups = new Map(navigation.publicationGroups.map((group, index) => [group.id, { ...group, index }]));
  const candidates = navigation.publicationGroups.flatMap((group, groupIndex) =>
    group.entries.map((entry, entryIndex) => ({
      group,
      groupIndex,
      entry,
      entryIndex,
    }))).filter(({ entry }) => route === entry.href || route.startsWith(entry.href));
  const direct = candidates.sort((left, right) => right.entry.href.length - left.entry.href.length)[0];
  if (direct) {
    return {
      navigationGroup: direct.group.id,
      navigationGroupTitle: direct.group.label,
      navigationRoot: direct.entry.href,
      navigationOrder: direct.groupIndex * 100 + direct.entryIndex,
    };
  }
  const mapping = navigation.guideMappings
    .filter((candidate) => route === candidate.prefix || route.startsWith(candidate.prefix))
    .sort((left, right) => right.prefix.length - left.prefix.length)[0];
  if (!mapping) return {
    navigationGroup: null,
    navigationGroupTitle: null,
    navigationRoot: null,
    navigationOrder: null,
  };
  const group = groups.get(mapping.group);
  if (!group) throw new Error(`Unknown navigation group: ${mapping.group}`);
  return {
    navigationGroup: group.id,
    navigationGroupTitle: group.label,
    navigationRoot: mapping.root,
    navigationOrder: group.index * 100 + 99,
  };
}

function headingSlug(value) {
  return value.toLowerCase().replace(/[`*_]/g, '').replace(/[^a-z0-9 -]/g, '')
    .trim().replace(/\s+/g, '-').replace(/-+/g, '-');
}

function anchorsFor(page) {
  const anchors = new Set([...page.content.matchAll(/\bid=["']([^"']+)["']/g)].map((match) => match[1]));
  if (page.sourcePath.endsWith('.mdx')) {
    for (const match of page.content.matchAll(/^#{1,6}\s+(.+)$/gm)) anchors.add(headingSlug(match[1]));
  }
  return anchors;
}

export function buildPublicationGraph(root = siteRoot) {
  const pages = publicationRegistry.pages.map((definition) => {
    const file = path.join(root, definition.sourcePath);
    const raw = fs.readFileSync(file, 'utf8');
    const parsed = definition.sourcePath.endsWith('.mdx')
      ? parseMdx(raw, file)
      : { title: titleForHtml(definition.id, raw), description: descriptionForHtml(definition.id, raw), content: raw };
    const aliases = [...new Set([
      ...(definition.canonicalUrl === '/' ? [] : [definition.canonicalUrl.slice(0, -1)]),
      ...(definition.aliases ?? []),
    ])].filter((alias) => alias !== definition.canonicalUrl);
    const bodyMarkdown = definition.sourcePath.endsWith('.mdx')
      ? mdxToMarkdown(parsed.content)
      : htmlFragmentToMarkdown(parsed.content);
    return {
      ...definition,
      ...navigationForRoute(definition.canonicalUrl),
      aliases,
      markdownAliases: definition.markdownAliases ?? [],
      prerequisites: definition.prerequisites ?? [],
      related: definition.related ?? [],
      publicModules: definition.publicModules ?? [],
      publicSymbols: definition.publicSymbols ?? [],
      componentIds: definition.componentIds ?? [],
      sourcePaths: definition.sourcePaths ?? [definition.sourcePath],
      testPaths: definition.testPaths ?? [],
      status: 'published',
      release: publicationRegistry.release,
      file,
      raw,
      ...parsed,
      bodyMarkdown,
    };
  });
  for (const page of pages) page.parent = nearestParent(page, pages);
  for (const page of pages) page.children = pages.filter((candidate) => candidate.parent === page.id).map((candidate) => candidate.id);
  return { ...publicationRegistry, pages };
}

export function readPages(root = siteRoot) {
  return buildPublicationGraph(root).pages.filter((page) => page.sourcePath.endsWith('.mdx')).map((page) => ({
    ...page,
    key: page.sourcePath.replace(/^content\/(?:docs|research)\//, '').replace(/\.mdx$/, '').replace(/^/, page.sourcePath.startsWith('content/research/') ? 'research/' : ''),
    url: page.canonicalUrl,
  }));
}

export function readStandalonePages(root = siteRoot) {
  return buildPublicationGraph(root).pages.filter((page) => page.sourcePath.endsWith('.html'));
}

export function readSearchPages(root = siteRoot) {
  return buildPublicationGraph(root).pages;
}

export function plain(value) {
  return value.replace(/```[^\n]*\n/g, '\n').replace(/```/g, '')
    .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1').replace(/[#*_`>|]/g, '')
    .replace(/\s+/g, ' ').trim();
}

export function plainHtml(value) {
  return plain(htmlFragmentToMarkdown(value));
}

export function whitepaperWordCount(root = siteRoot) {
  const source = fs.readFileSync(path.join(root, 'content', 'research', 'whitepaper.mdx'), 'utf8');
  return plain(source).match(/[A-Za-z0-9]+(?:[.'-][A-Za-z0-9]+)*/g)?.length ?? 0;
}

function localTarget(href) {
  return href.split(/[?#]/)[0].replace(/\/$/, '') || '/';
}

export function validate(root = siteRoot) {
  const graph = buildPublicationGraph(root);
  const routes = new Set(graph.pages.flatMap((page) => [page.canonicalUrl, ...page.aliases]).map(localTarget));
  const routePages = new Map(graph.pages.flatMap((page) =>
    [page.canonicalUrl, ...page.aliases].map((route) => [localTarget(route), page])));
  const errors = [];
  const check = (href, file) => {
    if (!href.startsWith('/') || href.startsWith('//')) return;
    const target = localTarget(href);
    if (routes.has(target)) {
      const fragment = href.includes('#') ? decodeURIComponent(href.split('#', 2)[1]) : '';
      const targetPage = routePages.get(target);
      if (fragment && targetPage && !anchorsFor(targetPage).has(fragment)) {
        errors.push(`${file}: missing fragment ${href}`);
      }
      return;
    }
    if (fs.existsSync(path.join(root, 'public', target))) return;
    errors.push(`${file}: ${href}`);
  };
  for (const page of graph.pages) {
    const pattern = page.sourcePath.endsWith('.mdx')
      ? /\]\(([^\s)]+)(?:\s[^)]*)?\)|href=["']([^"']+)["']/g
      : /href=["']([^"']+)["']/g;
    for (const match of page.content.matchAll(pattern)) check(match[1] ?? match[2], page.sourcePath);
  }
  return { pages: graph.pages.length, errors };
}
