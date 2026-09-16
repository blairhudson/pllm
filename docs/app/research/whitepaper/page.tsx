import type { Metadata } from 'next';
import Link from 'next/link';
import { markdownPathForRoute } from '@/lib/paths.mjs';
import styles from './whitepaper.module.css';

export const metadata: Metadata = {
  title: 'PLLM Whitepaper',
  description: 'A concise guide to PLLM private inference and its research harness.',
  alternates: {
    canonical: '/research/whitepaper/',
    types: { 'text/markdown': markdownPathForRoute('/research/whitepaper/') },
  },
};

const Arrow = () => <span className={styles.arrow} aria-hidden="true">→</span>;

export default function WhitepaperPage() {
  return (
    <main className={styles.document}>
      <section className={`${styles.page} ${styles.cover}`}>
        <header className={styles.header}>
          <Link href="/" className={styles.brand}>PLLM</Link>
          <span>Whitepaper · September 2026</span>
        </header>

        <div className={styles.coverCopy}>
          <p className={styles.kicker}>Private inference, built to improve</p>
          <h1>Run an LLM without giving one provider the plaintext conversation.</h1>
          <p className={styles.lead}>
            PLLM is a high-performance private LLM multi-party inference runtime and
            autonomous research harness. It keeps prompts, activations, and model
            state inside the client boundary while separate services perform the
            expensive linear work.
          </p>
        </div>

        <div className={styles.principles}>
          <article>
            <strong>Keep plaintext local</strong>
            <p>The client owns prompts, decoding, private state, masks, and token boundaries.</p>
          </article>
          <article>
            <strong>Prepare before chat</strong>
            <p>One-time masked corrections are produced offline, then consumed during inference.</p>
          </article>
          <article>
            <strong>Improve with evidence</strong>
            <p>Research components are composed into typed plans and compared by the same benchmark contract.</p>
          </article>
        </div>

        <div className={styles.roleStrip}>
          <div><span>Trusted</span><strong>Client</strong><small>plaintext + state</small></div>
          <Arrow />
          <div><span>Offline</span><strong>Preparation</strong><small>masked corrections</small></div>
          <Arrow />
          <div><span>Online</span><strong>Inference</strong><small>masked tensors</small></div>
        </div>

        <p className={styles.footnote}>
          Privacy requires preparation and inference to follow the protocol and not collude.
          PLLM does not claim protection against arbitrary malicious providers.
        </p>
      </section>

      <section className={styles.page}>
        <header className={styles.sectionHeader}>
          <span>01</span>
          <div>
            <p>Runtime + research harness</p>
            <h2>One system for execution and improvement</h2>
          </div>
        </header>

        <div className={styles.twoColumn}>
          <div>
            <h3>Private inference</h3>
            <ol className={styles.steps}>
              <li><b>Prepare.</b> The client authorizes an inventory. Preparation expands fresh seeds, computes <code>W·r−s</code>, and pushes the corrections to inference.</li>
              <li><b>Reserve.</b> Before a response, the client reserves one-use rows for every remote stage. Cancellation and failure burn unused rows.</li>
              <li><b>Execute.</b> The client sends a ticket and <code>x−r</code>. Inference consumes the matching correction and returns <code>W·x−s</code>. The client adds <code>s</code>.</li>
              <li><b>Decode.</b> Token lookup, output projection, sampling, and model state remain with the client.</li>
            </ol>
          </div>

          <div className={styles.runtimeBox}>
            <p className={styles.boxLabel}>Online path</p>
            <div className={styles.flow}>
              <div><strong>Client</strong><small>ticket + masked activation</small></div>
              <Arrow />
              <div><strong>Inference</strong><small>one-time masked result</small></div>
              <Arrow />
              <div><strong>Client</strong><small>reconstruct + decode</small></div>
            </div>
            <p>Preparation is idle while a response runs. The online path is client-to-inference only.</p>
          </div>
        </div>

        <div className={styles.harness}>
          <div>
            <p className={styles.boxLabel}>Autonomous research loop</p>
            <h3>Build a library of interchangeable research components.</h3>
            <p>
              Papers are reimplemented behind capability-based contracts: cache
              policies, numeric approximations, nonlinear protocols, matrix
              protocols, preparation schemes, kernels, and placement strategies.
              New capability families can be added as research evolves.
            </p>
          </div>
          <div className={styles.loop} aria-label="Research workflow">
            {['Specify', 'Implement', 'Assure', 'Benchmark', 'Compose', 'Search'].map((item, index) => (
              <span key={item}><b>{String(index + 1).padStart(2, '0')}</b>{item}</span>
            ))}
          </div>
        </div>

        <p className={styles.rule}>
          Like-for-like components live together under stable typed interfaces.
          Paper names remain attribution, not runtime architecture. Compatibility
          rules reject invalid combinations before execution.
        </p>
      </section>

      <section className={styles.page}>
        <header className={styles.sectionHeader}>
          <span>02</span>
          <div>
            <p>Current position</p>
            <h2>Working runtime, incomplete full-model coverage</h2>
          </div>
        </header>

        <div className={styles.statusTable}>
          <div><strong>Available</strong><p>Responses API and Chat Completions API through a trusted loopback gateway; prepared public-weight transport; Qwen runtime path; semantic model planning; benchmark dashboard and evidence records.</p></div>
          <div><strong>Experimental</strong><p>Bounded Q7 SiLU arithmetic garbling and MPCache plan transformation. These components do not establish complete-model execution or production security.</p></div>
          <div><strong>Not complete</strong><p>The compiler cannot yet execute a complete Qwen plan under the preferred protected profile. Multi-host, WAN, GPU, energy, price, quality, and adversarial studies remain unperformed.</p></div>
        </div>

        <div className={styles.nextGrid}>
          <article>
            <span>1</span>
            <h3>Complete Qwen coverage</h3>
            <p>Close every semantic, numeric, protected, placement, state, and token-feedback gap. Partial plans continue to fail closed.</p>
          </article>
          <article>
            <span>2</span>
            <h3>Grow the component library</h3>
            <p>Reimplement relevant papers as reusable components, validate each source-scoped claim, and preserve its trust model.</p>
          </article>
          <article>
            <span>3</span>
            <h3>Search compatible plans</h3>
            <p>Start with constrained grid and randomized search, then add smarter search only when repeated benchmark data justifies it.</p>
          </article>
        </div>

        <div className={styles.evidence}>
          <div>
            <p className={styles.boxLabel}>Evidence rule</p>
            <h3>Compare complete plans, not isolated claims.</h3>
          </div>
          <p>
            Every result binds model, graph, component versions, numeric profile,
            placement, trust assumptions, machine, workload, and cold/warm state.
            Functional correctness, privacy review, model quality, latency,
            communication, memory, and cost remain separate measurements.
          </p>
        </div>

        <footer className={styles.footer}>
          <div>
            <strong>Read the implementation detail</strong>
            <p>The technical paper, source-by-source catalog, and live status matrix state what is implemented and what remains open.</p>
          </div>
          <nav>
            <Link href="/research/paper/">Technical paper</Link>
            <Link href="/research/papers/">Papers</Link>
            <Link href="/research/backlog/">Backlog</Link>
            <Link href="/sdk/reference/status/">Status</Link>
          </nav>
        </footer>
      </section>
    </main>
  );
}
