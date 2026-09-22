import { getMDXComponents } from '@/components/mdx';
import { execSync } from 'node:child_process';
import { markdownPathForRoute, sourcePathForRoute, withBasePath } from '@/lib/paths.mjs';
import { getPageByCanonicalHref } from '@/lib/source';
import type { Metadata } from 'next';
import { notFound } from 'next/navigation';
import { DocsBody, DocsDescription, DocsPage, DocsTitle } from 'fumadocs-ui/layouts/docs/page';

function routeFor(slug: string[]) {
  return `/${slug.join('/')}/`;
}

let trackedFiles: Set<string> | undefined;

function editableSourceFor(canonical: string) {
  const sourcePath = sourcePathForRoute(canonical);
  if (!sourcePath) return undefined;
  const repoPath = `docs/${sourcePath}`;
  if (trackedFiles === undefined) {
    try {
      trackedFiles = new Set(
        execSync('git ls-files --full-name ":/docs/content"', { encoding: 'utf8' })
          .split('\n').filter(Boolean),
      );
    } catch {
      trackedFiles = new Set();
    }
  }
  return trackedFiles.has(repoPath) ? repoPath : undefined;
}

export function renderDocsPage(slug: string[]) {
  const canonical = routeFor(slug);
  const page = getPageByCanonicalHref(canonical)?.page;
  if (!page) notFound();
  const MDX = page.data.body;
  const editSource = editableSourceFor(canonical);
  return <DocsPage toc={page.data.toc} full={page.data.full}>
    <DocsTitle id="main-content" tabIndex={-1}>{page.data.title}</DocsTitle>
    <DocsDescription>{page.data.description}</DocsDescription>
    <div className="doc-actions"><span>PLLM documentation</span>
      <a href={withBasePath(markdownPathForRoute(canonical))}>View Markdown ↗</a>
      {editSource &&
        <a href={`https://github.com/blairhudson/pllm/edit/main/${editSource}`}>Edit this page ↗</a>}
    </div>
    <DocsBody><MDX components={getMDXComponents()} /></DocsBody>
  </DocsPage>;
}

export function docsMetadata(slug: string[]): Metadata {
  const canonical = routeFor(slug);
  const page = getPageByCanonicalHref(canonical)?.page;
  if (!page) notFound();
  return {
    title: page.data.title,
    description: page.data.description,
    alternates: {
      canonical,
      types: { 'text/markdown': markdownPathForRoute(canonical) },
    },
    openGraph: {
      type: 'article',
      url: canonical,
      title: page.data.title,
      description: page.data.description,
    },
    twitter: {
      card: 'summary_large_image',
      title: page.data.title,
      description: page.data.description,
    },
  };
}
