import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
export const siteRoot = fileURLToPath(new URL('../', import.meta.url));
export function walk(dir) {
 return fs.readdirSync(dir, {withFileTypes:true}).flatMap(e => e.isDirectory() ? walk(path.join(dir,e.name)) : [path.join(dir,e.name)]);
}
export function readPages(root = siteRoot) {
 const base=path.join(root,'content/docs');
 return walk(base).filter(p=>p.endsWith('.mdx')).sort().map(file=>{
  const raw=fs.readFileSync(file,'utf8');
  const match=raw.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
  if(!match) throw new Error(`Missing frontmatter: ${file}`);
  const field=name=>{const value=match[1].match(new RegExp(`^${name}: (.+)$`,'m'))?.[1]; if(!value) throw new Error(`Missing ${name}: ${file}`);return JSON.parse(value);};
  const key=path.relative(base,file).replace(/\\/g,'/').replace(/\.mdx$/,'');
  const url=key==='index'?'/docs/':`/docs/${key}/`;
  return {key,url,title:field('title'),description:field('description'),content:match[2],raw,file};
 });
}
export function plain(s) { return s.replace(/```[^\n]*\n/g,'\n').replace(/```/g,'').replace(/\[([^\]]+)\]\([^)]*\)/g,'$1').replace(/[#*_`>|]/g,'').replace(/\s+/g,' ').trim(); }
export function validate(root = siteRoot) {
 const pages=readPages(root); const routes=new Set(['/', '/research/', ...pages.map(p=>p.url)]);
 const errors=[]; const check=(href,file)=>{
  if(!href.startsWith('/')||href.startsWith('//')) return;
  const target=href.split(/[?#]/)[0];
  if(routes.has(target.endsWith('/')?target:target+'/')) return;
  if(fs.existsSync(path.join(root,'public',target))) return;
  errors.push(`${file}: ${href}`);
 };
 for(const p of pages) for(const m of p.content.matchAll(/\]\(([^\s)]+)(?:\s[^)]*)?\)/g)) check(m[1],p.key);
 for(const f of ['home.html','research.html']) {
  const p=path.join(root,'content',f);if(fs.existsSync(p)) for(const m of fs.readFileSync(p,'utf8').matchAll(/href="([^"]+)"/g)) check(m[1],f);
 }
 return {pages:pages.length,errors};
}
