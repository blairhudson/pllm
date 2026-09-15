import { getMDXComponents } from '@/components/mdx';
import { markdownPathForRoute, withBasePath } from '@/lib/paths.mjs';
import { getPageByCanonicalHref } from '@/lib/source';
import type { Metadata } from 'next';
import { notFound } from 'next/navigation';
import { DocsBody, DocsDescription, DocsPage, DocsTitle } from 'fumadocs-ui/layouts/docs/page';

function routeFor(slug: string[]) {
  return `/${slug.join('/')}/`;
}

export function renderDocsPage(slug: string[]) {
  const canonical = routeFor(slug);
  const page = getPageByCanonicalHref(canonical)?.page;
  if (!page) notFound();
  const MDX = page.data.body;
  return <DocsPage toc={page.data.toc} full={page.data.full}>
    <DocsTitle id="main-content" tabIndex={-1}>{page.data.title}</DocsTitle>
    <DocsDescription>{page.data.description}</DocsDescription>
    <div className="doc-actions"><span>PLLM documentation</span><a href={withBasePath(markdownPathForRoute(canonical))}>View Markdown ↗</a></div>
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
  };
}
