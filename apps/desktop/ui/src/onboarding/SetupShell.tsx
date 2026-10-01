import { createResource, createSignal, ErrorBoundary, onMount, Show, type JSX } from "solid-js";
import { Icon } from "../components/icons";
import { TitleBar } from "../components/TitleBar";
import { LegalDialog } from "../legal/LegalDialog";
import type { LegalDocId } from "../legal/types";
import type { Platform } from "../platform/platform";
import { TERMS_VERSION, termsAccepted, writeRecord, type OnboardingRecord } from "./progress";

/**
 * Everything around first-run setup that ships in the main bundle (small on purpose):
 * - the setup module itself (Onboarding.tsx, its steps and onboarding.css) loads as its own chunk,
 *   so every launch after the first pays nothing for it;
 * - while that chunk loads, the empty setup frame shows (title bar, rail, stage), never a blank page;
 * - an ErrorBoundary around setup: if anything in it fails (the chunk cannot load, a service answers
 *   with something unexpected), a fallback screen still offers the terms and Accept, so a broken
 *   setup can never lock anyone out of the app.
 */

type SetupProps = { platform: Platform; record: OnboardingRecord | null; onComplete: (record: OnboardingRecord) => void };

export function Setup(props: SetupProps): JSX.Element {
  return (
    <ErrorBoundary fallback={(err, reset) => <SetupFallback {...props} error={err} onRetry={reset} />}>
      <SetupLoader {...props} />
    </ErrorBoundary>
  );
}

function SetupLoader(props: SetupProps): JSX.Element {
  const [mod] = createResource(() => import("./Onboarding"));
  return (
    <>
      {(() => {
        // Reading an errored resource throws. Here that is the point: a chunk that failed to load
        // reaches the ErrorBoundary above and its fallback, instead of leaving the frame forever.
        const m = mod();
        return m ? <m.Onboarding platform={props.platform} record={props.record} onComplete={props.onComplete} /> : <SetupFrame platform={props.platform} />;
      })()}
    </>
  );
}

/** The setup frame with nothing in it yet: the same layout the steps arrive into. */
function SetupFrame(props: { platform: Platform }): JSX.Element {
  return (
    <div class="onb onb--loading" aria-busy="true">
      <TitleBar platform={props.platform} />
      <div class="onb__body">
        <div class="onb-frame__rail" />
        <main class="onb-frame__stage" id="main" tabindex="-1" aria-label="Setting up Cerebro" />
      </div>
    </div>
  );
}

/** First-run setup failed. The app still opens: read the terms, accept, and set the rest up later. */
function SetupFallback(props: SetupProps & { error: unknown; onRetry: () => void }): JSX.Element {
  const [legal, setLegal] = createSignal<LegalDocId | null>(null);
  const accepted = termsAccepted(props.record);
  const detail = (): string => {
    const e = props.error;
    const text = e instanceof Error ? `${e.name}: ${e.message}` : String(e);
    return text.length > 300 ? `${text.slice(0, 300)}…` : text;
  };
  const open = () => {
    const record: OnboardingRecord = { v: 1, completed: true, termsVersion: TERMS_VERSION, model: props.record?.model ?? null };
    writeRecord(record);
    props.onComplete(record);
  };
  let heading: HTMLHeadingElement | undefined;
  onMount(() => {
    document.title = "Setup · Set up Cerebro";
    requestAnimationFrame(() => heading?.focus({ preventScroll: true }));
  });
  return (
    <div class="onb onb--fallback">
      <TitleBar platform={props.platform} />
      <main class="setup-fallback" id="main" tabindex="-1" aria-labelledby="setup-fallback-title">
        <div class="setup-fallback__card card">
          <p class="eyebrow">
            <Icon name="alert" size={14} />
            Setup
          </p>
          <h1 class="setup-fallback__title" id="setup-fallback-title" tabindex="-1" ref={heading}>
            Setup could not finish.
          </h1>
          <p class="setup-fallback__lede">
            Cerebro itself is fine. Try setup again, or {accepted ? "open Cerebro" : "accept the terms and open Cerebro"} and choose a model later in Model Hub.
            Protections are on by default either way.
          </p>
          <div class="setup-fallback__actions">
            <button type="button" class="btn btn--primary" data-press data-sweep onClick={open}>
              {accepted ? "Open Cerebro" : "Accept and open Cerebro"}
              <span class="btn__arrow" aria-hidden="true">
                <Icon name="arrowRight" size={16} />
              </span>
            </button>
            <button type="button" class="btn btn--secondary" data-press onClick={() => props.onRetry()}>
              <Icon name="refresh" size={16} />
              Try setup again
            </button>
            <button type="button" class="btn btn--tertiary" data-press aria-haspopup="dialog" onClick={() => setLegal("terms")}>
              <Icon name="documents" size={16} />
              Read the terms
            </button>
          </div>
          <Show when={!accepted}>
            <p class="setup-fallback__note">
              <strong>Accept and open Cerebro</strong> agrees to the Terms of use, version {TERMS_VERSION}. The choice is remembered on this computer.
            </p>
          </Show>
          <details class="setup-fallback__detail">
            <summary>Technical detail</summary>
            <p class="mono">{detail()}</p>
          </details>
        </div>
      </main>
      <LegalDialog doc={legal()} onSelect={setLegal} onClose={() => setLegal(null)} />
    </div>
  );
}
