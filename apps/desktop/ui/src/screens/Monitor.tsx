import { createResource, ErrorBoundary, type JSX } from "solid-js";
import { Icon } from "../components/icons";
import { ScreenHeader } from "../components/Screen";

/**
 * The Monitor loads as its own chunk (screens/monitor/), so launch pays nothing for its charts and
 * inspector until the screen is opened. The header pill and alerts run app-wide from the shared
 * metrics feed (service/metrics.tsx) either way.
 */
export function Monitor(): JSX.Element {
  return (
    <ErrorBoundary fallback={(err, reset) => {
        // Kept in the console so a failure can be diagnosed; the screen shows a plain sentence.
        console.error(err);
        return <MonitorFallback onRetry={reset} />;
      }}>
      <MonitorLoader />
    </ErrorBoundary>
  );
}

function MonitorLoader(): JSX.Element {
  const [mod] = createResource(() => import("./monitor/MonitorScreen"));
  return (
    <>
      {(() => {
        const m = mod();
        return m ? (
          <m.MonitorScreen />
        ) : (
          <div class="screen">
            <ScreenHeader route="monitor" />
            <p class="sr-only" role="status">
              Opening Monitor…
            </p>
          </div>
        );
      })()}
    </>
  );
}

function MonitorFallback(props: { onRetry: () => void }): JSX.Element {
  return (
    <div class="screen">
      <ScreenHeader route="monitor" />
      <div class="notice" role="alert">
        <Icon name="alert" size={18} />
        <p>The Monitor could not be opened. The model keeps running; the status in the title bar stays current.</p>
        <button type="button" class="btn btn--secondary" data-press onClick={() => props.onRetry()}>
          Try again
        </button>
      </div>
    </div>
  );
}
