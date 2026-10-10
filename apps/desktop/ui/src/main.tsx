import "@design/tokens.css";
import "@design/fonts.css";
import "@design/motion.css";
import "@design/components.css";
import "./styles/app.css";
import "./styles/preview.css";
import "./styles/setup-frame.css";

import { initMotion } from "@design/motion";
import { render } from "solid-js/web";
import { App } from "./App";
import { createMockService, mockOptionsFromQuery } from "./mock/mock-service";
import { createDesktopService } from "./service/desktop-service";
import { createPlatform } from "./platform/platform";

const root = document.getElementById("root");
if (!root) throw new Error("#root is missing from index.html");

// The desktop app talks to Cerebro Engine through IPC (service/desktop-service.ts); the mock only
// keeps the user's local settings there. The web build (headless tests, the browser preview) uses
// the typed mock, which can fail on purpose (`?mock=fail-hw`, see mockOptionsFromQuery).
const LATENCY = { library: 450, load: 1100, test: 1200, scan: 700 };
const service = __DESKTOP__
  ? createDesktopService(createMockService({}))
  : createMockService({ latency: LATENCY, ...mockOptionsFromQuery(window.location.search) });
const platform = await createPlatform();

render(() => <App service={service} platform={platform} />, root);

// Watch mode arms motion on everything mounted later: route changes, toasts, and portalled
// layers (palette, tooltips) that render outside #root.
initMotion(document.body, { watch: true });
