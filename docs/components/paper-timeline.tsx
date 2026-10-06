'use client';

import { useState, type ReactNode } from 'react';

export function PaperTimeline({ children }: { children: ReactNode }) {
  const [implementedOnly, setImplementedOnly] = useState(false);
  return <div className="paper-browser" data-implemented-only={implementedOnly}>
    <label className="paper-filter"><input type="checkbox" checked={implementedOnly}
      onChange={(event) => setImplementedOnly(event.target.checked)} /> Hide unimplemented papers</label>
    <p className="paper-filter-scope">Includes scoped references and attack regressions. Each card states its coverage;
      only measured, executable SDK configurations receive full-response scores.</p>
    {children}
  </div>;
}
