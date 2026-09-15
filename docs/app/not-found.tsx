import Link from 'next/link';
export default function NotFound() {
 return <main id="main-content" className="not-found"><p className="eyebrow">404 / NOT FOUND</p><h1>That path ends here.</h1><p>The guide may have moved. Start with Learn.</p><Link href="/learn" className="button primary">Open Learn →</Link></main>;
}
