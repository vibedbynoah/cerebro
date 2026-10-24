import {
  createEffect,
  createSignal,
  onCleanup,
  onMount,
  Show,
  type JSX,
} from "solid-js";
import { firingExpert, useMoeActivity } from "../../lib/moe-activity";

/**
 * A mixture-of-experts model's experts as they fire: one row per layer, one cell per expert, the
 * brighter the more the recent tokens used it; the latest token's picks are outlined. Drawn on a
 * canvas in the theme's accent colour, read from the CSS tokens so it follows light and dark.
 */
export function ExpertHeatmap(): JSX.Element {
  const moe = useMoeActivity();
  let canvas: HTMLCanvasElement | undefined;
  let box: HTMLDivElement | undefined;
  const [width, setWidth] = createSignal(0);

  const draw = () => {
    const a = moe.activity();
    const c = canvas;
    if (!a || !c || a.experts === 0) return;
    const css = getComputedStyle(c);
    const accent =
      css.getPropertyValue("--color-accent").trim() || "currentColor";
    const ink =
      css.getPropertyValue("--color-text-primary").trim() || "currentColor";
    const base =
      css.getPropertyValue("--color-surface-2").trim() || "transparent";
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(120, width());
    const rowH = Math.max(3, Math.min(8, Math.floor(220 / a.layers)));
    const h = rowH * a.layers;
    c.width = Math.round(w * dpr);
    c.height = Math.round(h * dpr);
    c.style.width = `${w}px`;
    c.style.height = `${h}px`;
    const g = c.getContext("2d");
    if (!g) return;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.fillStyle = base;
    g.fillRect(0, 0, w, h);
    const cw = w / a.experts;
    g.fillStyle = accent;
    for (let l = 0; l < a.layers; l++) {
      for (let e = 0; e < a.experts; e++) {
        const v = a.heat[l * a.experts + e] ?? 0;
        if (v === 0) continue;
        // Square root: a cell used once in a while still shows beside the busiest ones.
        g.globalAlpha = Math.sqrt(v / 255);
        g.fillRect(e * cw, l * rowH, Math.max(1, cw - 0.5), rowH - 1);
      }
    }
    g.globalAlpha = 1;
    g.strokeStyle = ink;
    g.lineWidth = 1;
    a.last.forEach((row, l) => {
      for (const e of row)
        g.strokeRect(
          e * cw + 0.5,
          l * rowH + 0.5,
          Math.max(1, cw - 1),
          rowH - 2,
        );
    });
  };

  onMount(() => {
    if (!box) return;
    const ro = new ResizeObserver(() => setWidth(box!.clientWidth));
    ro.observe(box);
    onCleanup(() => ro.disconnect());
  });
  createEffect(draw);

  return (
    <div class="heatmap" ref={box}>
      <Show
        when={moe.activity()}
        fallback={
          <p class="dock__note">
            The map fills in as the model reads and writes.
          </p>
        }
      >
        {(a) => (
          <>
            <canvas
              ref={canvas}
              role="img"
              aria-label={`Experts firing: ${a().layers} layers of ${a().experts} experts; brighter means used more in the last tokens.`}
            />
            <p class="dock__note">
              <span class="mono">{a().layers}</span> layers ×{" "}
              <span class="mono">{a().experts}</span> experts
              {moe.active() && firingExpert(a()) !== null ? (
                <>
                  {" "}
                  · firing now: expert{" "}
                  <span class="mono">{firingExpert(a())}</span>
                </>
              ) : (
                " · resting"
              )}
            </p>
          </>
        )}
      </Show>
    </div>
  );
}
