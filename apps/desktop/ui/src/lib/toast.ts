import { createToaster } from "@ark-ui/solid/toast";

/** One app-wide toaster; <Toasts> renders it. Toast text is always plain text, never HTML. */
export const toaster = createToaster({
  placement: "bottom-end",
  gap: 12,
  max: 4,
  overlap: false,
  offsets: { top: "16px", bottom: "44px", left: "16px", right: "16px" },
});

export type ToastKind = "info" | "success" | "warning" | "error";

export function notify(title: string, description?: string, kind: ToastKind = "info"): void {
  toaster.create({
    title,
    ...(description !== undefined ? { description } : {}),
    type: kind,
    duration: kind === "error" ? 8000 : 4000,
  });
}
