import { docsMetadata, renderDocsPage } from '@/app/_docs-page';
import { source } from '@/lib/source';

type Props = { params: Promise<{ slug: string[] }> };

export default async function Page({ params }: Props) {
  const { slug } = await params;
  return renderDocsPage(slug);
}

export const dynamicParams = false;

export function generateStaticParams() {
  return source
    .getPages()
    .filter((page) => !page.url.startsWith('/research'))
    .map((page) => ({ slug: page.url.slice(1).split('/') }));
}

export async function generateMetadata({ params }: Props) {
  const { slug } = await params;
  return docsMetadata(slug);
}
