import "@design/tokens.css";
import "@design/fonts.css";
import "@design/motion.css";
import "@design/components.css";
import "./styles/app.css";
import "./styles/preview.css";

import { initMotion } from "@design/motion";
import { render } from "solid-js/web";
import { App } from "./App";
import { createMockService } from "./mock/mock-service";
import { createPlatform } from "./platform/platform";

const root = document.getElementById("root");
if (!root) throw new Error("#root is missing from index.html");

// All data comes from the typed mock until the IPC bridge implements CerebroService.
const service = createMockService();
const platform = await createPlatform();

render(() => <App service={service} platform={platform} />, root);

// Watch mode arms motion on everything mounted later: route changes, toasts, and portalled
// layers (palette, tooltips) that render outside #root.
initMotion(document.body, { watch: true });
