import { notFound } from 'next/navigation';
import { DocsBody, DocsDescription, DocsPage, DocsTitle } from 'fumadocs-ui/page';
import { researchSource, source } from '@/lib/source';
import { getMDXComponents } from '@/components/mdx';
import { markdownPathForRoute } from '@/lib/paths.mjs';
import { docsMetadata, renderDocsPage } from '@/app/_docs-page';

export default async function Page(props: { params: Promise<{ slug: string[] }> }) {
  const params = await props.params;
  const docsSlug = ['research', ...params.slug];
  if (source.getPageByHref(`/${docsSlug.join('/')}`)) return renderDocsPage(docsSlug);
  const page = researchSource.getPage(params.slug);
  if (!page) notFound();
  const MDX = page.data.body;
  return (
    <DocsPage toc={page.data.toc} full={page.data.full}>
      <DocsTitle id="main-content" tabIndex={-1}>{page.data.title}</DocsTitle>
      <DocsDescription>{page.data.description}</DocsDescription>
      <DocsBody><MDX components={getMDXComponents()} /></DocsBody>
    </DocsPage>
  );
}

export function generateStaticParams() {
  return [
    ...source
      .getPages()
      .filter((page) => page.url.startsWith('/research/'))
      .map((page) => ({ slug: page.url.split('/').slice(2) })),
    ...researchSource.generateParams(),
  ];
}

export async function generateMetadata(props: { params: Promise<{ slug: string[] }> }) {
  const params = await props.params;
  const docsSlug = ['research', ...params.slug];
  if (source.getPageByHref(`/${docsSlug.join('/')}`)) return docsMetadata(docsSlug);
  const page = researchSource.getPage(params.slug);
  if (!page) notFound();
  return {
    title: page.data.title,
    description: page.data.description,
    alternates: {
      canonical: `/research/${params.slug.join('/')}/`,
      types: { 'text/markdown': markdownPathForRoute(`/research/${params.slug.join('/')}/`) },
    },
  };
}
