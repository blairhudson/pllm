import { docs, research } from 'collections/server';
import type { Folder, Node, Root } from 'fumadocs-core/page-tree';
import { loader } from 'fumadocs-core/source';
import type { NavigationArea } from './navigation';
import { docsHrefForSlugs, fumadocsHref } from './docs-routes.mjs';
export const source = loader({
  baseUrl: '',
  source: docs.toFumadocsSource(),
  url: (slugs) => docsHrefForSlugs(slugs),
});
export const researchSource = loader({ baseUrl: '/research', source: research.toFumadocsSource() });

export function getPageByCanonicalHref(href: string) {
  return source.getPageByHref(fumadocsHref(href));
}

const areaOverview: Record<NavigationArea, string> = {
  Learn: '/learn',
  CLI: '/cli',
  SDK: '/sdk',
  Research: '/research',
};

const areaRootTitle: Record<NavigationArea, string> = {
  Learn: 'Learn',
  CLI: 'CLI',
  SDK: 'SDK',
  Research: 'Research guide',
};

function getAreaRoot(area: NavigationArea): Folder {
  const root = source.getPageTree().children.find((node) => {
    if (node.type !== 'folder' || node.root !== true) return false;
    return (
      node.index?.url === areaOverview[area] ||
      node.children.some(
        (child) => child.type === 'page' && child.url === areaOverview[area],
      ) ||
      (typeof node.name === 'string' && node.name === areaRootTitle[area])
    );
  });

  if (root === undefined || root.type !== 'folder') {
    throw new Error(`Missing Fumadocs root for ${area}`);
  }

  return root;
}

function nodeUrls(node: Node): string[] {
  if (node.type === 'page') return [node.url];
  if (node.type !== 'folder') return [];
  return [
    ...(node.index === undefined ? [] : [node.index.url]),
    ...node.children.flatMap(nodeUrls),
  ];
}

function researchPapers(): Folder {
  function visit(nodes: Node[]): Folder | undefined {
    for (const node of nodes) {
      if (node.type !== 'folder') continue;
      if (node.index?.url === '/research/papers' || node.children.some((child) =>
        child.type === 'page' && child.url === '/research/papers')) return node;
      const nested = visit(node.children);
      if (nested) return nested;
    }
    return undefined;
  }
  const folder = visit(source.getPageTree().children);
  if (!folder) {
    throw new Error('Missing generated research paper pages');
  }
  return { ...folder, name: 'Private Inference Papers', root: true };
}

function researchDocument(name: string, url: string): Folder {
  const page = researchSource.getPageTree().children.find((node) => nodeUrls(node).includes(url));
  if (!page) throw new Error(`Missing research document ${url}`);
  return { type: 'folder', name, root: true, children: [page] };
}

function researchFolders(): Folder[] {
  return [
    researchPapers(),
    researchDocument('PLLM Whitepaper', '/research/whitepaper'),
    researchDocument('PLLM Research Paper', '/research/paper'),
  ];
}

export type AreaSection = {
  title: string;
  url: string;
  urls: string[];
};

function areaFolders(area: NavigationArea, root: Folder): Folder[] {
  const sections: Folder[] = [
    {
      ...root,
      children: root.children.filter(
        (node) => node.type !== 'folder' || node.root !== true,
      ),
    },
    ...root.children.filter(
      (node): node is Folder => node.type === 'folder' && node.root === true,
    ),
  ];
  if (area !== 'Learn' && area !== 'SDK') return sections;

  const names = area === 'Learn'
    ? ['Start', 'Core concepts', 'Working with PLLM']
    : ['SDK', 'Experiments', 'Models', 'Inference options', 'Component options',
      'Run PLLM', 'Plans', 'Evaluate', 'Extend PLLM', 'API reference'];
  const order = new Map(names.map((name, index) => [name, index]));
  return sections.sort((left, right) => (
    order.get(String(left.name)) ?? Number.MAX_SAFE_INTEGER
  ) - (
    order.get(String(right.name)) ?? Number.MAX_SAFE_INTEGER
  ));
}

export function getAreaSections(area: NavigationArea): AreaSection[] {
  if (area === 'Research') {
    return researchFolders().map((section) => {
      const urls = nodeUrls(section);
      return { title: String(section.name), url: urls[0] ?? '/research/', urls };
    });
  }

  const root = getAreaRoot(area);
  const sections = areaFolders(area, root);

  const result = sections.map((section) => {
    if (typeof section.name !== 'string') {
      throw new Error(`Section title must be text in ${area}`);
    }
    const url = (section.name === root.name ? areaOverview[area] : section.index?.url)
      ?? section.children.find((node) => node.type === 'page')?.url;
    if (url === undefined) throw new Error(`Missing section index for ${section.name}`);
    const urls = nodeUrls(section);
    if (section.name === root.name && !urls.includes(url)) urls.unshift(url);
    return { title: section.name, url, urls };
  });

  const owners = new Map<string, string>();
  for (const section of result) {
    for (const url of new Set(section.urls)) {
      const owner = owners.get(url);
      if (owner !== undefined) {
        throw new Error(`${url} belongs to both ${owner} and ${section.title}`);
      }
      owners.set(url, section.title);
    }
  }
  return result;
}

export function getAreaPageTree(area: NavigationArea): Root {
  if (area === 'Research') {
    return { name: area, children: researchFolders() };
  }

  const root = getAreaRoot(area);

  return {
    name: area,
    children: areaFolders(area, root),
  };
}
