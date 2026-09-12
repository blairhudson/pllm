import type { ReactNode } from 'react';
import { DocsLayout } from 'fumadocs-ui/layouts/docs';
import { researchSource } from '@/lib/source';

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <div className="docs-root research-root">
      <DocsLayout
        tree={researchSource.getPageTree()}
        nav={{ enabled: false }}
        themeSwitch={{ enabled: false }}
        searchToggle={{ enabled: false }}
      >
        {children}
      </DocsLayout>
    </div>
  );
}
