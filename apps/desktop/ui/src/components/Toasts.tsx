import { Toast, Toaster } from "@ark-ui/solid/toast";
import { Show, type JSX } from "solid-js";
import { toaster } from "../lib/toast";
import { Icon, type IconName } from "./icons";

const ICONS: Record<string, IconName> = {
  success: "check",
  warning: "alert",
  error: "alert",
  info: "info",
};

/** Renders the app toaster. Title and description are text children (never HTML). */
export function Toasts(): JSX.Element {
  return (
    <Toaster toaster={toaster} class="toaster">
      {(toast) => (
        <Toast.Root class="toast" data-kind={toast().type ?? "info"}>
          <Icon name={ICONS[toast().type ?? "info"] ?? "info"} size={18} class="toast__icon" />
          <div class="toast__body">
            <Toast.Title class="toast__title">{toast().title}</Toast.Title>
            <Show when={toast().description}>
              <Toast.Description class="toast__desc">{toast().description}</Toast.Description>
            </Show>
          </div>
          <Toast.CloseTrigger class="toast__close" aria-label="Dismiss notification" data-press>
            <Icon name="close" size={14} />
          </Toast.CloseTrigger>
        </Toast.Root>
      )}
    </Toaster>
  );
}
