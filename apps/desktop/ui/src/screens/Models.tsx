import { createResource, ErrorBoundary, type JSX } from "solid-js";
import { Icon } from "../components/icons";
import { ScreenHeader } from "../components/Screen";

/**
 * The Models screen loads as its own chunk (screens/models/), so the rest of the app pays nothing
 * for the library, load panel and Tweak Studio until the screen is opened. While the chunk loads,
 * the header and an empty library frame show; if it fails, the screen says so and offers a retry.
 */
export function Models(): JSX.Element {
  return (
    <ErrorBoundary fallback={(err, reset) => {
        // Kept in the console so a failure can be diagnosed; the screen shows a plain sentence.
        console.error(err);
        return <ModelsFallback onRetry={reset} />;
      }}>
      <ModelsLoader />
    </ErrorBoundary>
  );
}

function ModelsLoader(): JSX.Element {
  const [mod] = createResource(() => import("./models/ModelsScreen"));
  return (
    <>
      {(() => {
        const m = mod();
        return m ? (
          <m.ModelsScreen />
        ) : (
          <div class="screen">
            <ScreenHeader route="models" />
            <p class="sr-only" role="status">
              Opening the library…
            </p>
          </div>
        );
      })()}
    </>
  );
}

function ModelsFallback(props: { onRetry: () => void }): JSX.Element {
  return (
    <div class="screen">
      <ScreenHeader route="models" />
      <div class="notice" role="alert">
        <Icon name="alert" size={18} />
        <p>The Models screen could not be opened. Nothing was changed.</p>
        <button type="button" class="btn btn--secondary" data-press onClick={() => props.onRetry()}>
          Try again
        </button>
      </div>
    </div>
  );
}
