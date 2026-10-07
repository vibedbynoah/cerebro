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
import { createPlatform } from "./platform/platform";

const root = document.getElementById("root");
if (!root) throw new Error("#root is missing from index.html");

// All data comes from the typed mock until the IPC bridge implements CerebroService. The web build
// (headless tests, the browser preview) can ask the mock to fail on purpose (`?mock=fail-hw`, see
// mockOptionsFromQuery) so failure paths are exercised; the desktop build compiles that branch out.
// Realistic latency so loading states are seen; a `?mock=` switch can override it in the web build.
const LATENCY = { library: 450, load: 1100, test: 1200, scan: 700 };
const service = createMockService(__DESKTOP__ ? { latency: LATENCY } : { latency: LATENCY, ...mockOptionsFromQuery(window.location.search) });
const platform = await createPlatform();

render(() => <App service={service} platform={platform} />, root);

// Watch mode arms motion on everything mounted later: route changes, toasts, and portalled
// layers (palette, tooltips) that render outside #root.
initMotion(document.body, { watch: true });
