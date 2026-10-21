/**
 * End of speech from the microphone's level (voice mode, screens/chat/VoiceMode.tsx): speech starts
 * when the level stays above the threshold for a moment, and ends after `silenceMs` below it.
 * Short clicks and pops never count as speech. Pure, so it is unit-tested.
 */

export type VadState = "waiting" | "speaking" | "done";

export interface VadOptions {
  /** Level (0..1, the recording's recent peak) that counts as speech. */
  threshold: number;
  /** Quiet this long after speech ends the turn, ms. */
  silenceMs: number;
  /** Speech must last this long before it counts, ms. */
  minSpeechMs: number;
}

export const DEFAULT_VAD: VadOptions = { threshold: 0.06, silenceMs: 800, minSpeechMs: 250 };

/** Sensitivity 0..1 (the voice mode slider) to a threshold: more sensitive hears quieter speech. */
export function thresholdFor(sensitivity: number): number {
  const s = Math.min(1, Math.max(0, sensitivity));
  return 0.18 - s * 0.16;
}

export class EndOfSpeech {
  private state: VadState = "waiting";
  private aboveSince: number | null = null;
  private belowSince: number | null = null;
  private speechStart: number | null = null;

  constructor(private readonly o: VadOptions = DEFAULT_VAD) {}

  /** One level reading at time `t` (ms); returns the state after it. */
  feed(level: number, t: number): VadState {
    if (this.state === "done") return "done";
    const loud = level >= this.o.threshold;
    if (this.state === "waiting") {
      if (loud) {
        this.aboveSince ??= t;
        if (t - this.aboveSince >= this.o.minSpeechMs) {
          this.state = "speaking";
          this.speechStart = this.aboveSince;
          this.belowSince = null;
        }
      } else {
        this.aboveSince = null;
      }
      return this.state;
    }
    // Speaking: quiet long enough ends the turn; any loud reading resets the quiet.
    if (loud) {
      this.belowSince = null;
    } else {
      this.belowSince ??= t;
      if (t - this.belowSince >= this.o.silenceMs) this.state = "done";
    }
    return this.state;
  }

  /** When speech started (ms), once it has. */
  started(): number | null {
    return this.speechStart;
  }
}
