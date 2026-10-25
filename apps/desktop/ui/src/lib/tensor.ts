import type { TensorInfo } from "../service/types";

/** "INT8 154 TOPS, FP16 38 TFLOPS (stock rate) at 1635 MHz": what the card measured. */
export function tensorLine(t: TensorInfo): string {
  const clock = t.smClockMhz ? ` at ${t.smClockMhz} MHz` : "";
  return `Measured INT8 ${Math.round(t.int8Tops)} TOPS, FP16 ${Math.round(t.f16Tflops)} TFLOPS (${t.f16FullRate ? "full rate, unlocked" : "stock rate"})${clock}`;
}
