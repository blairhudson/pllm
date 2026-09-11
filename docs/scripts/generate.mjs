import { withBasePath } from '../lib/paths.mjs';
import fs from 'node:fs';
import path from 'node:path';
import { readPages, readStandalonePages, plain, plainHtml } from './content.mjs';
const pages=readPages();
const standalone=readStandalonePages();
const searchable=[...pages,...standalone];
fs.mkdirSync('public/markdown',{recursive:true});
for(const p of pages){const file=path.join('public/markdown',p.key+'.md');fs.mkdirSync(path.dirname(file),{recursive:true});fs.writeFileSync(file,`# ${p.title}\n\n${p.description}\n\n${p.content}`);}
fs.writeFileSync('public/search-index.json', JSON.stringify(searchable.map(p=>({title:p.title,description:p.description,url:p.url,section:p.section??(p.key.includes('/')?p.key.split('/')[0]:'start'),text:p.format==='html'?plainHtml(p.content):plain(p.content)}))));
fs.writeFileSync('public/llms.txt','# PLLM\n\nPrivate LLM inference documentation. Public weights; provider follows the protocol.\n\n'+searchable.map(p=>`- [${p.title}](${withBasePath(p.url)}): ${p.description}`).join('\n')+'\n');
fs.writeFileSync('public/llms-full.txt',searchable.map(p=>`# ${p.title}\nURL: ${withBasePath(p.url)}\n\n${p.format==='html'?plainHtml(p.content):p.content}`).join('\n\n---\n\n'));
fs.writeFileSync('public/docs-manifest.json', JSON.stringify(pages.map(({key,url,title,description})=>({key,url,title,description})),null,2));
console.log(`Generated ${pages.length} docs and ${standalone.length} standalone pages, Markdown, and a browser search index.`);
