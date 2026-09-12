import { withBasePath } from '@/lib/paths.mjs';
import type { Metadata } from 'next';
import { notFound } from 'next/navigation';
import { DocsBody, DocsDescription, DocsPage, DocsTitle } from 'fumadocs-ui/layouts/docs/page';
import { getMDXComponents } from '@/components/mdx';
import { source } from '@/lib/source';
type Props = { params: Promise<{ slug?: string[] }> };
export default async function Page({ params }: Props) {
  const { slug } = await params;
  const page = source.getPage(slug);
  if (!page) notFound();
  const MDX = page.data.body;
  const key = slug?.join('/') || 'index';
  return <DocsPage toc={page.data.toc} full={page.data.full}>
    <DocsTitle id="main-content" tabIndex={-1}>{page.data.title}</DocsTitle>
    <DocsDescription>{page.data.description}</DocsDescription>
    <div className="doc-actions"><span>PLLM documentation</span><a href={withBasePath(`/markdown/${key}.md`)}>View Markdown ↗</a></div>
    <DocsBody><MDX components={getMDXComponents()} /></DocsBody>
  </DocsPage>;
}
export const dynamicParams = false;
export function generateStaticParams() { return source.generateParams(); }
export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { slug } = await params;
  const page = source.getPage(slug);
  if (!page) notFound();
  return {
    title: page.data.title,
    description: page.data.description,
    alternates: { canonical: `/docs/${slug?.join('/') ? `${slug.join('/')}/` : ''}` },
  };
}
