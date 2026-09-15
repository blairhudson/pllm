'use client';

import { mainNavigation, navigationItemForPathname } from '@/lib/navigation';
import { basePath } from '@/lib/paths.mjs';
import { Menu, Search, SunMoon } from 'lucide-react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { SidebarTrigger } from 'fumadocs-ui/layouts/docs/slots/sidebar';

function routeWithoutBasePath(pathname: string) {
  if (!basePath || !pathname.startsWith(`${basePath}/`)) return pathname;
  return pathname.slice(basePath.length) || '/';
}

export function DocsSidebarTitle() {
  return null;
}

export function DocsMobileHeader({ showMenu = true }: { showMenu?: boolean }) {
  const pathname = routeWithoutBasePath(usePathname());
  const active = navigationItemForPathname(pathname);

  function toggleTheme() {
    const dark = document.documentElement.classList.toggle('dark');
    try {
      localStorage.setItem('pllm-theme', dark ? 'dark' : 'light');
    } catch {}
  }

  return (
    <header id="nd-subnav" className="docs-mobile-header">
      <Link href="/" className="docs-mobile-brand" aria-label="PLLM home">
        pllm
      </Link>
      <details className="docs-area-switcher">
        <summary>{active?.label ?? 'Docs'}</summary>
        <nav aria-label="Documentation areas">
          {mainNavigation.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              aria-current={active?.href === item.href ? 'page' : undefined}
            >
              {item.label}
            </Link>
          ))}
        </nav>
      </details>
      <div className="docs-mobile-tools">
        <button
          type="button"
          onClick={() => document.dispatchEvent(new Event('pllm:search'))}
          aria-label="Search documentation"
        >
          <Search aria-hidden />
        </button>
        <button type="button" onClick={toggleTheme} aria-label="Toggle color theme">
          <SunMoon aria-hidden />
        </button>
        {showMenu ? (
          <SidebarTrigger aria-label="Open local navigation">
            <Menu aria-hidden />
          </SidebarTrigger>
        ) : null}
      </div>
    </header>
  );
}
