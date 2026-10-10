import { createResource, ErrorBoundary, type JSX } from "solid-js";
import { Icon } from "../components/icons";
import { ScreenHeader } from "../components/Screen";

/**
 * The Tune screen loads as its own chunk (screens/tune/), so launch pays nothing for the live charts
 * and the run view until the screen is opened. A tune keeps running in the service while the screen
 * is closed; opening it again replays the run. If the chunk fails to load, the screen says so and
 * offers a retry.
 */
export function Tune(): JSX.Element {
  return (
    <ErrorBoundary fallback={(err, reset) => {
        // Kept in the console so a failure can be diagnosed; the screen shows a plain sentence.
        console.error(err);
        return <TuneFallback onRetry={reset} />;
      }}>
      <TuneLoader />
    </ErrorBoundary>
  );
}

function TuneLoader(): JSX.Element {
  const [mod] = createResource(() => import("./tune/TuneScreen"));
  return (
    <>
      {(() => {
        const m = mod();
        return m ? (
          <m.TuneScreen />
        ) : (
          <div class="screen">
            <ScreenHeader route="tune" />
            <p class="sr-only" role="status">
              Opening Tune…
            </p>
          </div>
        );
      })()}
    </>
  );
}

function TuneFallback(props: { onRetry: () => void }): JSX.Element {
  return (
    <div class="screen">
      <ScreenHeader route="tune" />
      <div class="notice" role="alert">
        <Icon name="alert" size={18} />
        <p>The Tune screen could not be opened. Nothing was changed, and a running tune keeps running.</p>
        <button type="button" class="btn btn--secondary" data-press onClick={() => props.onRetry()}>
          Try again
        </button>
      </div>
    </div>
  );
}
