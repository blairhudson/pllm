'use client';

import {
  footerNavigation,
  mainNavigation,
  navigationItemForPathname,
  searchSuggestions,
} from '@/lib/navigation';
import { basePath, withBasePath } from '@/lib/paths.mjs';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useEffect, useRef, useState, type ReactNode } from 'react';

type Entry = {
  title: string;
  description: string;
  url: string;
  section: string;
  text: string;
};

function selectPhase(button: HTMLElement) {
  const group = button.closest<HTMLElement>('[data-tab-group]') ?? document;
  group.querySelectorAll<HTMLElement>('[data-phase]').forEach((candidate) => {
    const selected = candidate === button;
    candidate.setAttribute('aria-selected', String(selected));
    candidate.tabIndex = selected ? 0 : -1;
  });
  group.querySelectorAll<HTMLElement>('[data-panel]').forEach((panel) => {
    panel.hidden = panel.dataset.panel !== button.dataset.phase;
  });
}

export function SiteChrome({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const dialog = useRef<HTMLDialogElement>(null);
  const input = useRef<HTMLInputElement>(null);
  const [pages, setPages] = useState<Entry[]>([]);
  const [query, setQuery] = useState('');
  const [failed, setFailed] = useState(false);

  function openSearch() {
    dialog.current?.showModal();
    setTimeout(() => input.current?.focus(), 0);
  }

  function closeSearch() {
    dialog.current?.close();
  }

  useEffect(() => {
    try {
      const saved = localStorage.getItem('pllm-theme');
      document.documentElement.classList.toggle('dark', saved !== 'light');
    } catch {}
  }, []);

  useEffect(() => {
    let active = true;
    fetch(withBasePath('/search-index.json'))
      .then((response) => {
        if (!response.ok) throw Error('search index unavailable');
        return response.json();
      })
      .then((data: Entry[]) => {
        if (active) setPages(data);
      })
      .catch(() => {
        if (active) setFailed(true);
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    function handleSearchRequest() {
      openSearch();
    }

    function handleKey(event: KeyboardEvent) {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault();
        openSearch();
      }
      if (
        event.target instanceof HTMLElement &&
        event.target.matches('[data-phase]') &&
        ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End'].includes(event.key)
      ) {
        const group = event.target.closest<HTMLElement>('[data-tab-group]') ?? document;
        const phases = Array.from(group.querySelectorAll<HTMLElement>('[data-phase]'));
        let next = phases.indexOf(event.target);
        if (event.key === 'Home') next = 0;
        else if (event.key === 'End') next = phases.length - 1;
        else {
          const direction = ['ArrowLeft', 'ArrowUp'].includes(event.key) ? -1 : 1;
          next = (next + direction + phases.length) % phases.length;
        }
        event.preventDefault();
        selectPhase(phases[next]);
        phases[next].focus();
      }
    }

    document.addEventListener('pllm:search', handleSearchRequest);
    document.addEventListener('keydown', handleKey);
    return () => {
      document.removeEventListener('pllm:search', handleSearchRequest);
      document.removeEventListener('keydown', handleKey);
    };
  }, []);

  useEffect(() => {
    closeSearch();
    setQuery('');
    document.querySelectorAll<HTMLElement>('[data-phase]').forEach((button) => {
      button.tabIndex = button.getAttribute('aria-selected') === 'true' ? 0 : -1;
    });

    function handleClick(event: MouseEvent) {
      const element = (event.target as Element).closest<HTMLElement>('[data-copy],[data-phase]');
      if (!element) return;
      if (element.dataset.copy !== undefined) {
        const text = element.parentElement?.querySelector('pre')?.textContent ?? '';
        navigator.clipboard
          .writeText(text)
          .then(() => {
            const original = element.textContent;
            element.textContent = 'Copied';
            setTimeout(() => {
              element.textContent = original;
            }, 1500);
          })
          .catch(() => {
            element.textContent = 'Select code';
          });
      }
      if (element.dataset.phase) selectPhase(element);
    }

    document.addEventListener('click', handleClick);
    return () => document.removeEventListener('click', handleClick);
  }, [pathname]);

  function toggleTheme() {
    const dark = document.documentElement.classList.toggle('dark');
    try {
      localStorage.setItem('pllm-theme', dark ? 'dark' : 'light');
    } catch {}
  }

  const terms = query.toLowerCase().trim().split(/\s+/).filter(Boolean);
  const results = pages
    .map((page) => ({
      page,
      score: terms.reduce(
        (total, term) =>
          total +
          (page.title.toLowerCase().includes(term) ? 10 : 0) +
          (page.description.toLowerCase().includes(term) ? 4 : 0) +
          (page.text.toLowerCase().includes(term) ? 1 : 0),
        0,
      ),
    }))
    .filter(({ page, score }) => {
      const searchable = `${page.title} ${page.description} ${page.text}`.toLowerCase();
      return !terms.length || (score > 0 && terms.every((term) => searchable.includes(term)));
    })
    .sort((left, right) => right.score - left.score)
    .slice(0, 9);
  const routePath = basePath && pathname.startsWith(`${basePath}/`)
    ? pathname.slice(basePath.length)
    : pathname;
  const activeNavigation = navigationItemForPathname(routePath);
  const hasFumadocsNavigation = activeNavigation !== undefined;

  return (
    <>
      <a href="#main-content" className="skip-link">
        Skip to content
      </a>
      <header className={`site-header${hasFumadocsNavigation ? ' content-site-header' : ''}`}>
        <div className="header-inner">
          <Link href="/" className="brand" aria-label="PLLM home">
            <span className="brand-symbol" aria-hidden="true">
              {Array.from({ length: 9 }, (_, index) => (
                <i key={index} />
              ))}
            </span>
            pllm
          </Link>
          <span className="brand-tag">
            PRIVATE INFERENCE
            <br />
            RUNTIME
          </span>
          <nav className="top-nav" aria-label="Main">
            {mainNavigation.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                aria-current={
                  activeNavigation?.href === item.href ? 'page' : undefined
                }
              >
                {item.label}
              </Link>
            ))}
          </nav>
          <div className="header-tools">
            <button
              type="button"
              className="search-button"
              onClick={openSearch}
              aria-label="Search documentation"
            >
              <span aria-hidden="true">⌕</span>
              <span className="search-label">Search documentation</span>
              <kbd>⌘ K</kbd>
            </button>
            <button
              type="button"
              className="theme-button"
              onClick={toggleTheme}
              aria-label="Toggle color theme"
            >
              ◐
            </button>
          </div>
        </div>
      </header>
      {children}
      <footer className="site-footer">
        <div className="shell footer-inner">
          <Link className="brand" href="/">
            pllm
          </Link>
          <div className="footer-nav">
            {footerNavigation.map((item) => (
              <Link key={item.href} href={item.href}>
                {item.label}
              </Link>
            ))}
            <a href={withBasePath('/downloads/paper.pdf')}>Paper PDF ↗</a>
            <a href={withBasePath('/downloads/whitepaper.pdf')}>Whitepaper PDF ↗</a>
            <a href={withBasePath('/llms.txt')}>llms.txt</a>
          </div>
        </div>
      </footer>
      <dialog
        ref={dialog}
        className="search-dialog"
        aria-label="Search documentation"
        onClick={(event) => {
          if (event.target === event.currentTarget) closeSearch();
        }}
      >
        <div className="search-top">
          <span aria-hidden="true">⌕</span>
          <input
            ref={input}
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search current product documentation…"
            aria-label="Search documentation"
          />
          <button className="search-close" onClick={closeSearch}>
            ESC
          </button>
        </div>
        <div className="search-results" aria-live="polite">
          {failed ? (
            <p className="search-empty">
              Search index unavailable. <Link href="/learn">Browse Learn instead.</Link>
            </p>
          ) : results.length ? (
            results.map(({ page }) => (
              <Link
                className="search-result"
                key={page.url}
                href={page.url}
                onClick={closeSearch}
              >
                <em>{page.section}</em>
                <strong>{page.title}</strong>
                <span>{page.description}</span>
              </Link>
            ))
          ) : (
            <p className="search-empty">
              {pages.length
                ? `No match. Try “${searchSuggestions.join('”, “')}”.`
                : 'Loading documentation index…'}
            </p>
          )}
        </div>
        <div className="search-bottom">Search runs locally in this browser.</div>
      </dialog>
    </>
  );
}
