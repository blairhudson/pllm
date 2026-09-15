import defaultMdxComponents from 'fumadocs-ui/mdx';
import type { MDXComponents } from 'mdx/types';
import Link from 'next/link';
import { withBasePath } from '@/lib/paths.mjs';
import { CapabilityStatus } from './capability-status';

export function getMDXComponents(components?: MDXComponents): MDXComponents {
  return {
    ...defaultMdxComponents,
    CapabilityStatus,
    a: ({ href = '', children, ...props }) => {
      if (href.startsWith('/')) {
        return <Link href={href} {...props}>{children}</Link>;
      }
      return <a href={withBasePath(href)} {...props}>{children}</a>;
    },
    ...components,
  };
}
