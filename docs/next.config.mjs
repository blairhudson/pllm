import { createMDX } from 'fumadocs-mdx/next';
import { normalizeBasePath } from './lib/paths.mjs';
const withMDX = createMDX();
export default withMDX({
  output: 'export',
  trailingSlash: true,
  reactStrictMode: true,
  basePath: normalizeBasePath(process.env.NEXT_PUBLIC_BASE_PATH ?? ''),
  images: { unoptimized: true },
  poweredByHeader: false,
});
