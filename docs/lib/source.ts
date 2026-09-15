import { docs, research } from 'collections/server';
import type { Folder, Node, Root } from 'fumadocs-core/page-tree';
import { loader } from 'fumadocs-core/source';
import type { NavigationArea } from './navigation';
import { docsHrefForSlugs } from './docs-routes.mjs';
export const source = loader({
  baseUrl: '',
  source: docs.toFumadocsSource(),
  url: (slugs) => docsHrefForSlugs(slugs),
});
export const researchSource = loader({ baseUrl: '/research', source: research.toFumadocsSource() });

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

export type AreaSection = {
  title: string;
  url: string;
  urls: string[];
};

export function getAreaSections(area: NavigationArea): AreaSection[] {
  const root = getAreaRoot(area);
  const localChildren = root.children.filter(
    (node) => node.type !== 'folder' || node.root !== true,
  );
  const sectionRoots = root.children.filter(
    (node): node is Folder => node.type === 'folder' && node.root === true,
  );
  const sections: Folder[] = [{ ...root, children: localChildren }, ...sectionRoots];

  const result = sections.map((section, index) => {
    if (typeof section.name !== 'string') {
      throw new Error(`Section title must be text in ${area}`);
    }
    const url = (index === 0 ? areaOverview[area] : section.index?.url)
      ?? section.children.find((node) => node.type === 'page')?.url;
    if (url === undefined) throw new Error(`Missing section index for ${section.name}`);
    const urls = nodeUrls(section);
    if (index === 0 && !urls.includes(url)) urls.unshift(url);
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
  const root = getAreaRoot(area);
  const localChildren = root.children.filter(
    (node) => node.type !== 'folder' || node.root !== true,
  );
  const sectionRoots = root.children.filter(
    (node) => node.type === 'folder' && node.root === true,
  );

  return {
    name: area,
    children: [
      { ...root, children: localChildren },
      ...sectionRoots,
    ],
  };
}
