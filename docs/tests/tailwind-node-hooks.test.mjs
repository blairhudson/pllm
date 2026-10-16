import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { pathToFileURL } from 'node:url';

for (const mode of ['require', 'import']) {
  test(`Tailwind ${mode} uses nondeprecated hooks and preserves transitive cache ids`, () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'pllm-tailwind-hook-'));
    try {
      fs.writeFileSync(path.join(directory, 'child.mjs'), 'export default import.meta.url;');
      fs.writeFileSync(path.join(directory, 'parent.mjs'), 'import "node:fs"; export { default } from "./child.mjs";');
      const parent = pathToFileURL(path.join(directory, 'parent.mjs')).href;
      const result = spawnSync(process.execPath, ['--trace-deprecation', '--input-type=module', '-e', `
        import assert from 'node:assert/strict';
        import { createRequire } from 'node:module';
        ${mode === 'require' ? 'createRequire(import.meta.url)("@tailwindcss/node");' : 'await import("@tailwindcss/node");'}
        const first = await import(${JSON.stringify(parent + '?id=first')});
        const second = await import(${JSON.stringify(parent + '?id=second')});
        assert.match(first.default, /child\\.mjs\\?id=first$/);
        assert.match(second.default, /child\\.mjs\\?id=second$/);
      `], { cwd: new URL('..', import.meta.url), encoding: 'utf8' });
      assert.equal(result.status, 0, result.stderr);
      assert.doesNotMatch(result.stderr, /DEP0205|module\.register\(\).*deprecated/);
    } finally {
      fs.rmSync(directory, { recursive: true, force: true });
    }
  });
}
