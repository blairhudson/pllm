'use client';

import { useEffect, useState, type ReactNode } from 'react';

export function PaperTimeline({ children }: { children: ReactNode }) {
  const [implementedOnly, setImplementedOnly] = useState(false);
  useEffect(() => {
    const reveal = (event: Event) => {
      setImplementedOnly(false);
      const id = `paper-${(event as CustomEvent<string>).detail}`;
      requestAnimationFrame(() => document.getElementById(id)?.scrollIntoView({ block: 'start', behavior: 'smooth' }));
    };
    window.addEventListener('pllm-paper-focus', reveal);
    return () => window.removeEventListener('pllm-paper-focus', reveal);
  }, []);
  return <div className="paper-browser" data-implemented-only={implementedOnly} onClick={(event) => {
    const link = (event.target as HTMLElement).closest<HTMLAnchorElement>('[data-research-paper]');
    if (link) window.dispatchEvent(new CustomEvent('pllm-paper-select', { detail: link.dataset.researchPaper }));
  }}>
    <label className="paper-filter"><input type="checkbox" checked={implementedOnly}
      onChange={(event) => setImplementedOnly(event.target.checked)} /> Hide unimplemented papers</label>
    <p className="paper-filter-scope">Includes scoped references and attack regressions. Each card states its coverage;
      only measured, executable SDK configurations receive full-response scores.</p>
    {children}
  </div>;
}
