import navigation from '@/navigation.json';

export type NavigationItem = {
  href: string;
  label: string;
  owns?: readonly string[];
};

export const navigationAreas = ['Learn', 'CLI', 'SDK', 'Research'] as const;
export type NavigationArea = (typeof navigationAreas)[number];

export const mainNavigation: readonly NavigationItem[] = navigation.main;

function ownsPath(pathname: string, prefix: string) {
  return pathname === prefix || pathname.startsWith(`${prefix}/`);
}

export function navigationItemForPathname(pathname: string) {
  return mainNavigation
    .flatMap((item) => (item.owns ?? [item.href]).map((prefix) => ({ item, prefix })))
    .filter(({ prefix }) => ownsPath(pathname, prefix))
    .sort((left, right) => right.prefix.length - left.prefix.length)[0]?.item;
}

export function navigationAreaForPathname(pathname: string): NavigationArea | undefined {
  const label = navigationItemForPathname(pathname)?.label;
  return navigationAreas.find((area) => area === label);
}

export const footerNavigation: readonly NavigationItem[] = [
  { href: '/sdk/research/assurance/', label: 'Privacy and assurance' },
  { href: '/sdk/reference/status/', label: 'Current support' },
  { href: '/research/', label: 'Research methods' },
];

export const searchSuggestions = ['private inference', 'Python SDK', 'research methods'];
