import { createResource, createSignal, For, Show, type JSX } from "solid-js";
import { BrandLockup, MarkShapes, RouteMotif } from "../components/Brand";
import { Icon, type IconName } from "../components/icons";
import { ScreenHeader, Section } from "../components/Screen";
import { Term } from "../components/Term";
import { LegalDialog } from "../legal/LegalDialog";
import { LEGAL_DOCS, type LegalDocId } from "../legal/types";
import type { Platform } from "../platform/platform";

/**
 * About Cerebro: what it does for you, this copy's version, what the mark means, and the documents
 * (terms, privacy, licences). The routing motif sits behind the hero only (BRAND.md "Motif").
 */
export function About(props: { platform: Platform }): JSX.Element {
  const [info] = createResource(() => props.platform.appInfo());
  const [legal, setLegal] = createSignal<LegalDocId | null>(null);

  const promises: Array<{ icon: IconName; title: string; text: JSX.Element }> = [
    { icon: "tune", title: "Measured, not guessed", text: "Every model gets the fastest configuration that passes Cerebro's checks on this machine. Sometimes that is the defaults, and Cerebro says so." },
    { icon: "security", title: "Protected by default", text: "Integrity check before every start, process-tree cleanup, a clean environment. Each protection can be switched off." },
    {
      icon: "offline",
      title: "Local by default",
      text: (
        <>
          Models run on this computer and the <Term k="api" /> answers on <Term k="loopback" /> only by default. Network features are optional.
        </>
      ),
    },
  ];

  return (
    <div class="screen">
      <ScreenHeader route="about" title="About Cerebro" lede="The fastest safe configuration for every model on this machine, measured, then run with its protections on." />

      <section class="about-hero card reveal" data-reveal aria-label="Cerebro">
        <RouteMotif class="about-hero__motif" />
        <div class="about-hero__body">
          <BrandLockup height={48} class="about-hero__lockup" />
          <p class="about-hero__meta">
            <span class="chip" data-tone="neutral">
              Version <span class="mono">{info()?.version ?? "…"}</span>
            </span>
            <span class="chip" data-tone="neutral">
              <Show when={info()} fallback="…">
                {(i) => (i().mode === "desktop" ? "Desktop app" : "Browser preview with illustrative data")}
              </Show>
            </span>
            <span class="chip" data-tone="neutral">Apache-2.0</span>
          </p>
        </div>
      </section>

      <Section id="what" title="What is included">
        <ul class="about-promises">
          <For each={promises}>
            {(p) => (
              <li class="about-promise">
                <span class="about-promise__icon" aria-hidden="true">
                  <Icon name={p.icon} size={18} />
                </span>
                <span class="about-promise__title">{p.title}</span>
                <span class="about-promise__text">{p.text}</span>
              </li>
            )}
          </For>
        </ul>
      </Section>

      <Section
        id="mark"
        title="The mark"
        description={
          <>
            A <Term k="moe" /> model sends each token to a few of its experts. The mark draws that moment: seven idle experts, the router at the centre, and the one route it
            took.
          </>
        }
      >
        <figure class="mark-anatomy">
          <svg class="mark-anatomy__mark" viewBox="-2 -2 28 28" width="140" height="140" aria-hidden="true">
            <MarkShapes />
          </svg>
          <figcaption class="mark-anatomy__legend">
            <span class="mark-anatomy__item">
              <span class="mark-anatomy__swatch mark-anatomy__swatch--idle" aria-hidden="true" />
              <span>
                <strong>Experts</strong> idle this token, in ink
              </span>
            </span>
            <span class="mark-anatomy__item">
              <span class="mark-anatomy__swatch mark-anatomy__swatch--route" aria-hidden="true" />
              <span>
                <strong>Router, route and picked expert</strong>, the only blue
              </span>
            </span>
          </figcaption>
        </figure>
      </Section>

      <Section id="documents" title="Documents" description="The accepted terms, how Cerebro treats data, and the licences of everything it ships.">
        <div class="about-docs">
          <For each={LEGAL_DOCS}>
            {(d) => (
              <button type="button" class="btn btn--secondary about__doc" data-press aria-haspopup="dialog" onClick={() => setLegal(d.id)}>
                <Icon name="documents" size={16} />
                {d.label}
              </button>
            )}
          </For>
        </div>
      </Section>
      <LegalDialog doc={legal()} onSelect={setLegal} onClose={() => setLegal(null)} />
    </div>
  );
}
