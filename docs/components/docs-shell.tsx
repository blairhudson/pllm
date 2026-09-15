import type { ReactNode } from 'react';
import { DocsLayout } from 'fumadocs-ui/layouts/docs';
import type { NavigationArea } from '@/lib/navigation';
import { getAreaPageTree, getAreaSections, source } from '@/lib/source';
import { DocsMobileHeader, DocsSidebarTitle } from './docs-mobile-header';
import { DocsSectionSwitcher } from './docs-section-switcher';

export function DocsShell({
  children,
  area,
  sidebar = true,
  research = false,
}: {
  children: ReactNode;
  area?: NavigationArea;
  sidebar?: boolean;
  research?: boolean;
}) {
  const sectionSwitcher = area === undefined
    ? undefined
    : <DocsSectionSwitcher area={area} sections={getAreaSections(area)} />;

  return (
    <div className={`docs-root${research ? ' research-root' : ''}`}>
      <DocsLayout
        tree={area === undefined ? source.getPageTree() : getAreaPageTree(area)}
        tabs={false}
        nav={{ component: <DocsMobileHeader showMenu={sidebar} /> }}
        slots={{ navTitle: DocsSidebarTitle }}
        sidebar={{ enabled: sidebar, banner: sectionSwitcher }}
        themeSwitch={{ enabled: false }}
        searchToggle={{ enabled: false }}
      >
        {children}
      </DocsLayout>
    </div>
  );
}
