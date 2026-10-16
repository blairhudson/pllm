/** Tailwind 4.3.0 compatibility: prefer synchronous Node hooks (DEP0205).
 * Remove when upstream uses registerHooks. Keep its cache-id propagation and
 * legacy fallback; do not suppress warnings or change global module APIs.
 */
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const root = path.dirname(require.resolve('@tailwindcss/node'));
const version = JSON.parse(fs.readFileSync(path.join(root, '../package.json'))).version;
if (version !== '4.3.0') throw new Error('Review Tailwind Node hook compatibility patch for new version');

// Equivalent to esm-cache.loader.mjs, without async/await: registerHooks runs
// in the importing thread and nextResolve returns synchronously.
// Resolve the loader URL BEFORE registration: require.resolve inside a hook
// recursively invokes that same hook under the synchronous API.
const hook = (module, url) => `((loaderUrl)=>${module}.registerHooks({resolve(specifier,context,nextResolve){const result=nextResolve(specifier,context);if(result.url===loaderUrl||${module}.isBuiltin(result.url)||!context.parentURL)return result;const id=new URL(context.parentURL).searchParams.get("id");if(id===null)return result;const resolved=new URL(result.url);resolved.searchParams.set("id",id);return {...result,url:resolved.href}}}))(${url})`;
const patches = [
  ['index.js',
    'process.versions.bun||_t.register?.((0,Dt.pathToFileURL)(require.resolve("@tailwindcss/node/esm-cache-loader")));',
    `process.versions.bun||(_t.registerHooks?${hook('_t', '(0,Dt.pathToFileURL)(require.resolve("@tailwindcss/node/esm-cache-loader")).href')}:_t.register?.((0,Dt.pathToFileURL)(require.resolve("@tailwindcss/node/esm-cache-loader"))));`],
  ['index.mjs',
    'if(!process.versions.bun){let e=fe.createRequire(import.meta.url);fe.register?.(Xr(e.resolve("@tailwindcss/node/esm-cache-loader")))}',
    `if(!process.versions.bun){let e=fe.createRequire(import.meta.url);fe.registerHooks?${hook('fe', 'Xr(e.resolve("@tailwindcss/node/esm-cache-loader")).href')}:fe.register?.(Xr(e.resolve("@tailwindcss/node/esm-cache-loader")))}`],
];
for (const [file, before, after] of patches) {
  const target = path.join(root, file);
  const source = fs.readFileSync(target, 'utf8');
  if (source.includes(after)) continue;
  if (source.split(before).length !== 2) throw new Error(`Tailwind hook patch target changed: ${file}`);
  fs.writeFileSync(target, source.replace(before, after));
}
