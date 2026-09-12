import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {readPages,readSearchPages,readStandalonePages,validate,siteRoot,whitepaperWordCount} from '../scripts/content.mjs';
const pages=readPages();
const standalone=readStandalonePages();
test('navigation has unique routes',()=>assert.equal(new Set(pages.map(p=>p.url)).size,pages.length));
test('all local content links resolve',()=>assert.deepEqual(validate().errors,[]));
test('required user journeys exist',()=>{for(const p of ['quickstart','client/install','client/python','client/openai','client/agents','server/install','server/usage','deployment/docker','research/market','research/paper']) assert.ok(pages.some(v=>v.key===p),p);});
test('standalone research and whitepaper join search content',()=>{
 assert.deepEqual(standalone.map(page=>page.url),['/research/','/research/whitepaper/']);
 assert.deepEqual(readSearchPages().slice(-2).map(page=>page.key),['research','whitepaper']);
});
test('whitepaper copy and vector equivalents meet publication constraints',()=>{
 const source=fs.readFileSync(path.join(siteRoot,'content/whitepaper.html'),'utf8');
 const tex=fs.readFileSync(path.join(siteRoot,'../paper/whitepaper.tex'),'utf8');
 const words=whitepaperWordCount();
 assert.ok(words>=750&&words<=900,`whitepaper has ${words} words`);
 assert.equal((source.match(/<svg\b/g)||[]).length,2);
 assert.equal((source.match(/<title\b/g)||[]).length,2);
 assert.equal((source.match(/<desc\b/g)||[]).length,2);
 assert.ok(source.includes('market mechanism, not a measured savings result'));
  assert.ok(source.includes('SEPARATE OPERATORS'));
  assert.ok(source.includes('class="wp-download-bar"'));
  assert.doesNotMatch(source,/Private inference \/ decision brief|Revision 1\.1|TeX source/);
  assert.doesNotMatch(source,/foundation for broader supply|energy-aware routing|price competition/);
  assert.ok(source.includes('Private inference lets organisations use remote AI compute'));
  assert.doesNotMatch(source,/PLLM separates useful computation from/);
  assert.doesNotMatch(tex,/PLLM separates useful computation from/);
  assert.doesNotMatch(source,/custod[y]/i);
  assert.doesNotMatch(tex,/custod[y]/i);
  assert.ok(source.includes('M 130 75 L 130 40 L 810 40 L 810 75'));
  assert.ok(source.includes('M 810 185 L 810 220 L 130 220 L 130 185'));
  assert.match(source, /wp-svg-order-text[^>]*>1<[^]*wp-svg-order-text[^>]*>2<[^]*wp-svg-order-text[^>]*>3<[^]*wp-svg-order-text[^>]*>4</);
  assert.ok(!source.toLowerCase().includes('gradient'));
  assert.ok(tex.includes('letterpaper'));
  assert.ok(tex.includes('\\usepackage{tikz}'));
  assert.ok(tex.includes('client.north')&&tex.includes('infer.north'));
  assert.ok(tex.includes('yshift=-.70cm'));
  assert.doesNotMatch(source,/customer/i);
  assert.doesNotMatch(tex,/customer/i);
  assert.ok(!tex.toLowerCase().includes('gradient'));
  for (const url of [
    'https://pllm.run/research/architecture/',
    'https://pllm.run/docs/security/',
    'https://pllm.run/research/reproduce/',
    'https://pllm.run/downloads/current-runtime-2026-09-11.json',
  ]) {
    assert.ok(source.includes(url),url);
    assert.ok(tex.includes(url),url);
  }
  assert.doesNotMatch(source,/ARCHITECTURE\.md|SECURITY\.md|source distribution/);
  assert.doesNotMatch(tex,/ARCHITECTURE\.md|SECURITY\.md|source distribution/);
});
test('public copy consistently names the client',()=>{
  for(const page of [...pages,...standalone]) {
    assert.doesNotMatch(page.content,/customer/i,page.url);
    assert.doesNotMatch(page.content,/custod[y]/i,page.url);
  }
  assert.doesNotMatch(fs.readFileSync(path.join(siteRoot,'content/home.html'),'utf8'),/customer/i);
});
test('research landing page leads with mission, market, paper, and participation',()=>{
  const source=fs.readFileSync(path.join(siteRoot,'content/research.html'),'utf8');
  const css=fs.readFileSync(path.join(siteRoot,'app/research/research.module.css'),'utf8');
  assert.ok(source.includes('Unlocking a global market for private AI compute.'));
  assert.ok(source.includes('Our research aims to make compute globally accessible'));
  assert.ok(source.includes('Compute is global. Trust is not.'));
  assert.ok(source.includes('Separate AI compute from data.'));
  assert.ok(source.includes('Private inference lets providers run AI workloads'));
  assert.doesNotMatch(source,/PLLM lets providers run AI workloads/);
  assert.match(css,/\.researchPage :global\(\.research-problem\)\s*{[^}]*border-bottom: 0;/s);
  assert.ok(source.includes('One mission. Four connected questions.'));
  assert.ok(source.includes('Read Whitepaper <span>→</span>'));
  assert.doesNotMatch(source,/Read PLLM Whitepaper|<span>↗<\/span>/);
  assert.doesNotMatch(source,/research-history|rh-timeline|Current result first/);
  const program=source.slice(source.indexOf('<div class="program-grid">'));
  assert.ok(program.indexOf('Global compute market')<program.indexOf('Private inference'));
  assert.ok(program.indexOf('Private inference')<program.indexOf('PLLM Research Paper'));
  assert.doesNotMatch(source,/Current evidence|Build the privacy layer first|Private demand|Global supply|mission-mark/);
});
test('custom research pages occupy the docs content grid',()=>{
  const layout=fs.readFileSync(path.join(siteRoot,'app/research/layout.tsx'),'utf8');
  const css=fs.readFileSync(path.join(siteRoot,'app/global.css'),'utf8');
  assert.ok(layout.includes('docs-root research-root'));
  assert.match(css,/\.research-root #nd-docs-layout > main\s*{[^}]*grid-area: main;[^}]*min-width: 0;/s);
});
test('research navigation excludes retired roadmap',()=>{
  assert.ok(!pages.some(page=>page.key==='research/roadmap'));
  assert.equal(pages.find(page=>page.key==='research/reproduce').title,'Run your own benchmark');
  assert.equal(pages.find(page=>page.key==='research/market').title,'Global compute market');
  assert.equal(pages.find(page=>page.key==='research/architecture').title,'Private inference');
  assert.ok(!pages.some(page=>page.key==='research/benchmarks'));
  const meta=JSON.parse(fs.readFileSync(path.join(siteRoot,'content/research/meta.json'),'utf8'));
  assert.deepEqual(meta.pages,['[Research goals](/research/)','market','architecture','[PLLM Whitepaper](/research/whitepaper/)','[PLLM Research Paper](/research/paper/)','reproduce']);
});
test('research subpages explain the general market and privacy boundaries',()=>{
  const market=pages.find(page=>page.key==='research/market').content;
  const architecture=pages.find(page=>page.key==='research/architecture').content;
  const reproduce=pages.find(page=>page.key==='research/reproduce').content;
  assert.ok(market.includes('Private inference is therefore market-enabling infrastructure'));
  assert.doesNotMatch(market,/PLLM exists|market PLLM enables|PLLM records/);
  assert.ok(architecture.includes('Main design families'));
  assert.ok(architecture.includes('Fully homomorphic encryption'));
  assert.ok(architecture.includes('Secure multiparty computation'));
  assert.ok(architecture.includes('Trusted execution environments'));
  assert.doesNotMatch(architecture,/PLLM provides|current protocol uses|current runtime/);
  assert.ok(reproduce.includes("PLLM's market mission"));
  assert.doesNotMatch(reproduce,/Historical BFV study/);
});
test('private provider is not documented as ordinary Responses base URL',()=>{const p=pages.find(p=>p.key==='client/openai').content;assert.ok(p.includes('http://127.0.0.1:8080/v1'));assert.ok(!p.includes('base_url="http://127.0.0.1:8000/v1"'));});
test('papers and evidence are downloadable',()=>{for(const f of ['paper.pdf','paper-source.zip','whitepaper.pdf','whitepaper.tex','lifecycle-summary.json','current-runtime-2026-09-11.json','evidence.zip','cli-help.txt']) assert.ok(fs.statSync(path.join(siteRoot,'public/downloads',f)).size>0,f);});
test('generated paper is valid MDX-compatible Markdown',()=>assert.doesNotMatch(fs.readFileSync(path.join(siteRoot,'content/research/paper.mdx'),'utf8'),/<https?:\/\/[^>]+>/));
test('research paper exposes only the PDF action',()=>{
  const paper=fs.readFileSync(path.join(siteRoot,'content/research/paper.mdx'),'utf8');
  assert.ok(paper.includes('/downloads/paper.pdf'));
  assert.doesNotMatch(paper,/Download evidence|Download source|paper-byline|paper-scope|\/downloads\/(?:evidence\.zip|paper-source\.zip)/);
});
test('quick start installs the published package with uv',()=>assert.ok(pages.find(p=>p.key==='quickstart').content.includes('uv pip install pllm')));
test('release setup uses one uv command',()=>assert.ok(pages.find(p=>p.key==='reference/releasing').content.includes('uv run python scripts/release.py prepare')));
test('homepage gradient is full bleed without a lower rule',()=>{
  const css=fs.readFileSync(path.join(siteRoot,'app/home.module.css'),'utf8');
  const hero=css.slice(css.indexOf('.page :global(.vision-hero)'),css.indexOf('.page :global(.vision-hero::after)'));
  assert.ok(hero.includes('overflow: visible'));
  assert.ok(!hero.includes('border-bottom'));
});
test('Agents example disables tracing',()=>assert.ok(pages.find(p=>p.key==='client/agents').content.includes('set_tracing_disabled(True)')));
test('static export and privacy friendly assets',()=>{assert.ok(fs.readFileSync(path.join(siteRoot,'next.config.mjs'),'utf8').includes("output: 'export'"));assert.ok(!fs.readFileSync(path.join(siteRoot,'app/layout.tsx'),'utf8').includes('next/font/google'));});
test('public pages use the pllm.run canonical origin',()=>{
  const layout=fs.readFileSync(path.join(siteRoot,'app/layout.tsx'),'utf8');
  assert.ok(layout.includes("metadataBase: new URL('https://pllm.run')"));
  for(const file of ['app/page.tsx','app/research/page.tsx','app/research/whitepaper/page.tsx','app/research/[...slug]/page.tsx','app/docs/[[...slug]]/page.tsx']) {
    assert.ok(fs.readFileSync(path.join(siteRoot,file),'utf8').includes('canonical'),file);
  }
});
test('whitepaper downloads match canonical artifacts',()=>{
 assert.deepEqual(fs.readFileSync(path.join(siteRoot,'public/downloads/whitepaper.tex')),fs.readFileSync(path.join(siteRoot,'../paper/whitepaper.tex')),'whitepaper.tex');
 assert.match(fs.readFileSync(path.join(siteRoot,'public/downloads/whitepaper.pdf')).subarray(0,5).toString(),/^%PDF-/);
});
