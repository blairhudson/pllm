import Link from 'next/link';
export default function NotFound() {
 return <main id="main-content" className="not-found"><p className="eyebrow">404 / NOT FOUND</p><h1>That path ends here.</h1><p>The guide may have moved. Start with the documentation index.</p><Link href="/docs" className="button primary">Open documentation →</Link></main>;
}
