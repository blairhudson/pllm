'use client';

import type { NavigationArea } from '@/lib/navigation';
import type { AreaSection } from '@/lib/source';
import { basePath } from '@/lib/paths.mjs';
import Link from 'next/link';
import { usePathname } from 'next/navigation';

function normalizePath(pathname: string) {
  const withoutBase = basePath && pathname.startsWith(`${basePath}/`)
    ? pathname.slice(basePath.length)
    : pathname;
  return withoutBase.length > 1 ? withoutBase.replace(/\/$/, '') : withoutBase;
}

export function DocsSectionSwitcher({
  area,
  sections,
}: {
  area: NavigationArea;
  sections: AreaSection[];
}) {
  const pathname = normalizePath(usePathname());

  return (
    <nav className="docs-section-switcher" aria-label={`${area} sections`}>
      <div className="docs-section-switcher-list">
        {sections.map((section) => {
          const active = section.urls.some((url) => normalizePath(url) === pathname);
          return (
            <Link
              key={section.url}
              href={section.url}
              data-active={active || undefined}
              aria-current={active ? 'location' : undefined}
            >
              {section.title}
            </Link>
          );
        })}
      </div>
      <hr />
    </nav>
  );
}
