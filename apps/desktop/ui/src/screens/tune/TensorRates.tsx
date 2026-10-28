import { Show, type Accessor, type JSX } from "solid-js";
import type { HardwareProfile } from "../../service/types";

/**
 * The graphics card's tensor-core rates as measured on this run, in a dropdown. An overclock raises
 * them all; an FP16 unlock mod (GPU Unlock on GeForce RTX 30) doubles the FP16 one. Cerebro reads
 * prompts with INT8 because it is the fastest math these model files allow: INT4 would be twice as
 * fast, but one INT4 operation sums 64 values and the files change scale every 32.
 */
export function TensorRates(props: { hardware: Accessor<HardwareProfile | null> }): JSX.Element {
  return (
    <Show when={props.hardware()?.gpu?.tensor}>
      {(t) => (
        <details class="tensor-rates">
          <summary>
            {props.hardware()?.gpu?.name}: measured speed{t().smClockMhz ? ` at ${t().smClockMhz} MHz` : ""}
          </summary>
          <dl class="tensor-rates__body">
            <dt>INT8</dt>
            <dd>
              {Math.round(t().int8Tops)} TOPS <span class="hint">reads prompts (the fastest these model files allow)</span>
            </dd>
            <dt>INT4</dt>
            <dd>
              {Math.round(t().int4Tops)} TOPS{" "}
              <span class="hint">not used: one INT4 step sums 64 values, and the files change scale every 32</span>
            </dd>
            <dt>FP16</dt>
            <dd>
              {Math.round(t().f16Tflops)} TFLOPS{" "}
              <span class="hint">attention; {t().f16FullRate ? "full rate (unlocked)" : "stock GeForce half rate (an unlock mod doubles it)"}</span>
            </dd>
          </dl>
        </details>
      )}
    </Show>
  );
}
