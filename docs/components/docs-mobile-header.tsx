'use client';

import { mainNavigation, navigationItemForPathname } from '@/lib/navigation';
import { basePath, withBasePath } from '@/lib/paths.mjs';
import { Menu, Search, SunMoon } from 'lucide-react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useEffect, useRef } from 'react';
import { useSidebar } from 'fumadocs-ui/layouts/docs/slots/sidebar';

function routeWithoutBasePath(pathname: string) {
  if (!basePath || !pathname.startsWith(`${basePath}/`)) return pathname;
  return pathname.slice(basePath.length) || '/';
}

export function DocsSidebarTitle() {
  return null;
}

function DocsMenuButton() {
  const { open, setOpen } = useSidebar();
  const trigger = useRef<HTMLButtonElement>(null);
  const wasOpen = useRef(false);

  useEffect(() => {
    if (open) {
      wasOpen.current = true;
      const drawer = document.getElementById('nd-sidebar-mobile');
      drawer?.querySelector<HTMLElement>('button, a[href]')?.focus();
    } else if (wasOpen.current) {
      wasOpen.current = false;
      trigger.current?.focus();
    }

    function closeOnEscape(event: KeyboardEvent) {
      if (event.key === 'Escape') setOpen(false);
    }
    document.addEventListener('keydown', closeOnEscape);
    return () => document.removeEventListener('keydown', closeOnEscape);
  }, [open, setOpen]);

  return (
    <button
      ref={trigger}
      type="button"
      aria-label={open ? 'Close local navigation' : 'Open local navigation'}
      aria-controls="nd-sidebar-mobile"
      aria-expanded={open}
      onClick={() => setOpen((value) => !value)}
    >
      <Menu aria-hidden />
    </button>
  );
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
        <img
          src={withBasePath('/icon.svg')}
          alt=""
          width="24"
          height="16"
        />
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
        {showMenu ? <DocsMenuButton /> : null}
      </div>
    </header>
  );
}
