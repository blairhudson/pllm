'use client';
import { withBasePath } from '@/lib/paths.mjs';
import { useEffect, useRef, useState, type ReactNode } from 'react';
import { usePathname } from 'next/navigation';
import Link from 'next/link';
type Entry = { title: string; description: string; url: string; section: string; text: string };
export function SiteChrome({ children }: { children: ReactNode }) {
 const pathname=usePathname(); const dialog=useRef<HTMLDialogElement>(null);
 const input=useRef<HTMLInputElement>(null); const [pages,setPages]=useState<Entry[]>([]);
 const [query,setQuery]=useState(''); const [failed,setFailed]=useState(false);
 function openSearch(){dialog.current?.showModal();setTimeout(()=>input.current?.focus(),0);}
 function closeSearch(){dialog.current?.close();}
 useEffect(()=>{try {const saved=localStorage.getItem('pllm-theme');document.documentElement.classList.toggle('dark',saved!=='light');}catch{}},[]);
 useEffect(()=>{let active=true;fetch(withBasePath('/search-index.json')).then(r=>{if(!r.ok)throw Error();return r.json();}).then((data:Entry[])=>{if(active)setPages(data);}).catch(()=>{if(active)setFailed(true);});return()=>{active=false;};},[]);
 function selectPhase(button:HTMLElement){document.querySelectorAll<HTMLElement>('[data-phase]').forEach(b=>{const selected=b===button;b.setAttribute('aria-selected',String(selected));b.tabIndex=selected?0:-1;});document.querySelectorAll<HTMLElement>('[data-panel]').forEach(p=>p.hidden=p.dataset.panel!==button.dataset.phase);}
 useEffect(()=>{function key(e:KeyboardEvent){if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='k'){e.preventDefault();openSearch();}if(e.target instanceof HTMLElement&&e.target.matches('[data-phase]')&&['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Home','End'].includes(e.key)){const all=Array.from(document.querySelectorAll<HTMLElement>('[data-phase]'));let n=all.indexOf(e.target);n=e.key==='Home'?0:e.key==='End'?all.length-1:(n+(['ArrowLeft','ArrowUp'].includes(e.key)?-1:1)+all.length)%all.length;e.preventDefault();selectPhase(all[n]);all[n].focus();}}document.addEventListener('keydown',key);return()=>document.removeEventListener('keydown',key);},[]);
 useEffect(()=>{closeSearch();setQuery('');document.querySelectorAll<HTMLElement>('[data-phase]').forEach(b=>b.tabIndex=b.getAttribute('aria-selected')==='true'?0:-1);
  function click(e:MouseEvent){const element=(e.target as Element).closest<HTMLElement>('[data-copy],[data-phase]');if(!element)return;
   if(element.dataset.copy!==undefined){const text=element.parentElement?.querySelector('pre')?.textContent||'';navigator.clipboard.writeText(text).then(()=>{const old=element.textContent;element.textContent='Copied';setTimeout(()=>{element.textContent=old;},1500);}).catch(()=>{element.textContent='Select code';});}
   if(element.dataset.phase){selectPhase(element);}
  }document.addEventListener('click',click);return()=>document.removeEventListener('click',click);
 },[pathname]);
 function toggleTheme(){const dark=document.documentElement.classList.toggle('dark');try{localStorage.setItem('pllm-theme',dark?'dark':'light');}catch{}}
 const terms=query.toLowerCase().trim().split(/\s+/).filter(Boolean);
 const results=pages.map(p=>({p,score:terms.reduce((n,t)=>n+(p.title.toLowerCase().includes(t)?10:0)+(p.description.toLowerCase().includes(t)?4:0)+(p.text.toLowerCase().includes(t)?1:0),0)})).filter(({p,score})=>!terms.length||score>0&&terms.every(t=>(p.title+' '+p.description+' '+p.text).toLowerCase().includes(t))).sort((a,b)=>b.score-a.score).slice(0,9);
 return <>
  <a href="#main-content" className="skip-link">Skip to content</a>
  <header className="site-header"><div className="header-inner">
   <Link href="/" className="brand" aria-label="PLLM home"><span className="brand-symbol" aria-hidden="true">{Array.from({length:9},(_,i)=><i key={i}/>)}</span>pllm</Link>
   <span className="brand-tag">PRIVATE LLM<br/>INFERENCE</span>
   <nav className="top-nav" aria-label="Main"><Link href="/docs" aria-current={pathname.startsWith('/docs')?'page':undefined}>Documentation</Link><Link href="/research" aria-current={pathname==='/research'?'page':undefined}>Research</Link><Link href="/docs/deployment/overview">Deploy</Link></nav>
   <div className="header-tools"><button type="button" className="search-button" onClick={openSearch} aria-label="Search documentation"><span aria-hidden="true">⌕</span><span className="search-label">Search documentation</span><kbd>⌘ K</kbd></button><button type="button" className="theme-button" onClick={toggleTheme} aria-label="Toggle colour theme">◐</button></div>
  </div></header>
  {pathname.startsWith('/docs')&&<details className="mobile-doc-nav"><summary>Browse documentation</summary><div className="mobile-doc-links">{pages.map(p=><Link key={p.url} href={p.url}>{p.title}</Link>)}</div></details>}
  {children}
  <footer className="site-footer"><div className="shell footer-inner"><Link className="brand" href="/">pllm</Link><div className="footer-nav"><Link href="/docs/security">Security</Link><Link href="/docs/reference/compatibility">Compatibility</Link><a href={withBasePath('/downloads/paper.pdf')}>Paper PDF ↗</a><a href={withBasePath('/llms.txt')}>llms.txt</a></div><span className="build-tag">RESEARCH SOFTWARE · SOURCE EDITION 2026.09</span></div></footer>
  <dialog ref={dialog} className="search-dialog" aria-label="Search documentation" onClick={e=>{if(e.target===e.currentTarget)closeSearch();}}>
   <div className="search-top"><span aria-hidden="true">⌕</span><input ref={input} value={query} onChange={e=>setQuery(e.target.value)} placeholder="Search the docs, protocol, and paper…" aria-label="Search documentation"/><button className="search-close" onClick={closeSearch}>ESC</button></div>
   <div className="search-results" aria-live="polite">{failed?<p className="search-empty">The index could not be loaded. <Link href="/docs">Browse documentation instead.</Link></p>:results.length?results.map(({p})=><Link className="search-result" key={p.url} href={p.url} onClick={closeSearch}><em>{p.section}</em><strong>{p.title}</strong><span>{p.description}</span></Link>):<p className="search-empty">{pages.length?'No matching pages. Try “preparation” or “OpenAI”.':'Loading the documentation index…'}</p>}</div>
   <div className="search-bottom">Search runs in your browser. Queries never leave this page.</div>
  </dialog>
 </>;
}
