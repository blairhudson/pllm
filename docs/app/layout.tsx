import { withBasePath } from '@/lib/paths.mjs';
import type { ReactNode } from 'react';
import type { Metadata } from 'next';
import { RootProvider } from 'fumadocs-ui/provider/next';
import { SiteChrome } from '@/components/site-chrome';
import './global.css';
export const metadata: Metadata = {
  metadataBase: new URL('https://pllm.run'),
  title: { default: 'PLLM — Private LLM Inference', template: '%s · PLLM' },
  description: 'Private language model inference. Client and server guides, the protocol, and reproducible research.',
  icons: { icon: withBasePath('/icon.svg') },
  robots: { index: true, follow: true },
  openGraph: {
    type: 'website',
    siteName: 'PLLM',
    images: [{ url: withBasePath('/og.png'), width: 1200, height: 630, alt: 'PLLM — Private LLM Inference' }],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'PLLM — Private LLM Inference',
    description: 'Private language model inference. Client and server guides, the protocol, and reproducible research.',
    images: [withBasePath('/og.png')],
  },
};
export default function RootLayout({ children }: { children: ReactNode }) {
  return <html lang="en" className="dark" suppressHydrationWarning>
    <body>
      <RootProvider theme={{ enabled: false }} search={{ enabled: false }}>
        <SiteChrome>{children}</SiteChrome>
      </RootProvider>
    </body>
  </html>;
}
