import type { ReactNode } from 'react';
import { DocsShell } from '@/components/docs-shell';
import { navigationAreaForPathname } from '@/lib/navigation';

export default async function Layout({
  children,
  params,
}: {
  children: ReactNode;
  params: Promise<{ slug: string[] }>;
}) {
  const { slug } = await params;
  const pathname = `/${slug.join('/')}`;
  const area = navigationAreaForPathname(pathname);

  if (area === undefined) throw new Error(`No documentation area owns ${pathname}`);
  return <DocsShell area={area}>{children}</DocsShell>;
}
