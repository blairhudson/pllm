import type { ReactNode } from 'react';
import { DocsLayout } from 'fumadocs-ui/layouts/docs';
import { source } from '@/lib/source';
export default function Layout({ children }: { children: ReactNode }) {
  return <div className="docs-root">
    <DocsLayout tree={source.getPageTree()} nav={{ enabled: false }}
      themeSwitch={{ enabled: false }} searchToggle={{ enabled: false }}>
      {children}
    </DocsLayout>
  </div>;
}
