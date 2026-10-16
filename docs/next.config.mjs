import { createMDX } from 'fumadocs-mdx/next';
import { normalizeBasePath } from './lib/paths.mjs';
const withMDX = createMDX();
const config = {
  // Isolate dev locks/cache. With static export, a custom production distDir
  // changes the export destination rather than isolating the build cache.
  distDir: process.env.NODE_ENV === 'development' ? process.env.NEXT_DIST_DIR ?? '.next' : '.next',
  trailingSlash: true,
  reactStrictMode: true,
  allowedDevOrigins: ['127.0.0.1', 'macbookblair.local', 'macbookblair.tail858b09.ts.net'],
  basePath: normalizeBasePath(process.env.NEXT_PUBLIC_BASE_PATH ?? ''),
  images: { unoptimized: true },
  poweredByHeader: false,
};

if (process.env.NODE_ENV !== 'development') config.output = 'export';

export default withMDX(config);
