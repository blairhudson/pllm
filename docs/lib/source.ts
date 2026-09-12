import { docs, research } from 'collections/server';
import { loader } from 'fumadocs-core/source';
export const source = loader({ baseUrl: '/docs', source: docs.toFumadocsSource() });
export const researchSource = loader({ baseUrl: '/research', source: research.toFumadocsSource() });
