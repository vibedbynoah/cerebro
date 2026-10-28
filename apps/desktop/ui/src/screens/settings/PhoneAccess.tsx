import "../../styles/phone.css";
import { createSignal, For, onCleanup, Show, type JSX } from "solid-js";
import { Section } from "../../components/Screen";
import { SettingSwitch } from "../../components/SettingSwitch";
import { notify } from "../../lib/toast";
import { useService } from "../../service/context";
import type { RemoteDevice, RemoteView } from "../../service/types";

/** 64 short words, so a generated password can be read off a screen and typed on a phone keyboard. */
const WORDS = [
  "amber", "anchor", "apple", "arrow", "aspen", "atlas", "autumn", "badge",
  "basil", "beacon", "birch", "bison", "bloom", "breeze", "brick", "brook",
  "canyon", "cedar", "chalk", "clover", "cobalt", "comet", "coral", "crane",
  "creek", "crystal", "dawn", "delta", "denim", "dune", "ember", "falcon",
  "fern", "flame", "forest", "fossil", "galaxy", "garden", "glacier", "granite",
  "harbor", "harvest", "hawk", "hollow", "ivory", "jade", "kettle", "lagoon",
  "lantern", "lemon", "lotus", "maple", "marble", "meadow", "mist", "moss",
  "north", "nova", "ocean", "olive", "orbit", "pebble", "pine", "plum",
] as const;

/** Six words from 64: 36 bits, on top of the one-time QR code, Argon2 and the sign-in lockout. */
const WORD_COUNT = 6;
const MIN_PASSWORD = 8;

/**
 * Phone access (Settings screen, id="phone"): a password, one switch, a QR pairing code and the
 * list of phones allowed in. Everything stays on the local network — nothing is published.
 */
export function PhoneAccess(): JSX.Element {
  const service = useService();
  const errMessage = (e: unknown): string => (e instanceof Error ? e.message : "Unknown error");

  const [view, setView] = createSignal<RemoteView | null>(null);
  const [now, setNow] = createSignal(Date.now());

  const [password, setPassword] = createSignal("");
  const [confirm, setConfirm] = createSignal("");
  const [generated, setGenerated] = createSignal<string | null>(null);
  const [savingPassword, setSavingPassword] = createSignal(false);

  const [pair, setPair] = createSignal<{ url: string; qrSvg: string; expiresAt: number } | null>(null);
  const [pairing, setPairing] = createSignal(false);

  // Polled 3 times a second (never overlapping, skipped while hidden): last-seen times, the running
  // state and the QR countdown change under the reader's eyes. A failure is reported once.
  let pollFailed = false;
  let inFlight = false;
  const load = async () => {
    setNow(Date.now());
    if (inFlight || document.visibilityState === "hidden") return;
    inFlight = true;
    try {
      setView(await service.remoteGet());
      pollFailed = false;
    } catch (e) {
      if (!pollFailed) notify("Could not read phone access", errMessage(e), "error");
      pollFailed = true;
    } finally {
      inFlight = false;
    }
  };

  void load();
  const poll = setInterval(() => void load(), 333);
  onCleanup(() => clearInterval(poll));

  const statusState = (): "on" | "off" | "error" => {
    const v = view();
    if (!v) return "off";
    if (v.error) return "error";
    return v.enabled && v.running ? "on" : "off";
  };

  const statusText = (): string => {
    const v = view();
    if (!v) return "Checking…";
    if (v.error) return v.error;
    if (!v.enabled) return "Off";
    if (!v.running || !v.url) return "Starting…";
    return `On at ${v.url}`;
  };

  // Strength is length only: it says how much typing there is, never how safe this is.
  const strength = (): { label: string; level: string } => {
    const n = password().length;
    if (n === 0) return { label: `At least ${MIN_PASSWORD} characters.`, level: "none" };
    if (n < MIN_PASSWORD) return { label: `Too short — ${MIN_PASSWORD} characters minimum.`, level: "weak" };
    if (n < 12) return { label: "Weak.", level: "weak" };
    if (n < 16) return { label: "Fair.", level: "fair" };
    return { label: "Strong.", level: "strong" };
  };

  const copy = async (value: string, what: string) => {
    try {
      await navigator.clipboard.writeText(value);
      notify("Copied", what, "success");
    } catch {
      notify("Could not copy", "Read it off the screen and type it instead.", "warning");
    }
  };

  const generate = () => {
    const draws = new Uint32Array(WORD_COUNT);
    crypto.getRandomValues(draws);
    const value = Array.from(draws, (d) => WORDS[Math.floor((d / 2 ** 32) * WORDS.length)] ?? "amber").join("-");
    setPassword(value);
    setConfirm(value);
    setGenerated(value);
  };

  const submitPassword = async (event: SubmitEvent) => {
    event.preventDefault();
    const value = password();
    if (value.length < MIN_PASSWORD) {
      notify("That password is too short", `Use at least ${MIN_PASSWORD} characters.`, "warning");
      return;
    }
    if (value !== confirm()) {
      notify("The two passwords are different", "Type the same password twice.", "warning");
      return;
    }
    setSavingPassword(true);
    try {
      setView(await service.remoteSetPassword(value));
      setPassword("");
      setConfirm("");
      setGenerated(null);
      notify("Password set", "Turn phone access on to pair a phone.");
    } catch (e) {
      notify("Could not set the password", errMessage(e), "error");
    } finally {
      setSavingPassword(false);
    }
  };

  const toggle = async (on: boolean) => {
    try {
      setView(await service.remoteEnable(on));
      notify(
        on ? "Phone access is on" : "Phone access is off",
        on ? "Paired phones can reach Cerebro again." : "Paired phones cannot connect until it is turned back on.",
        "success",
      );
    } catch (e) {
      notify(on ? "Could not turn phone access on" : "Could not turn phone access off", errMessage(e), "error");
    }
  };

  const pairPhone = async () => {
    setPairing(true);
    try {
      setPair(await service.remotePair());
    } catch (e) {
      notify("Could not create a pairing code", errMessage(e), "error");
    } finally {
      setPairing(false);
    }
  };

  const revoke = async (device: RemoteDevice | null) => {
    try {
      setView(await service.remoteRevoke(device ? device.id : "*"));
      notify(
        device ? `${device.name} signed out` : "All phones signed out",
        "A signed-out phone has to pair again.",
        "success",
      );
    } catch (e) {
      notify(device ? `Could not sign out ${device.name}` : "Could not sign out all phones", errMessage(e), "error");
    }
  };

  const saveOptions = async (
    patch: Partial<{ allowChat: boolean; allowTailscale: boolean; address: string | null; sessionDays: number }>,
    title: string,
    message: string,
  ) => {
    const current = view();
    if (!current) return;
    try {
      setView(
        await service.remoteOptions({
          allowChat: current.allowChat,
          allowTailscale: current.allowTailscale,
          address: current.addressChoice,
          sessionDays: current.sessionDays,
          ...patch,
        }),
      );
      notify(title, message, "success");
    } catch (e) {
      notify("Could not save that option", errMessage(e), "error");
    }
  };

  const relative = (at: number): string => {
    const seconds = Math.max(0, Math.round((now() - at) / 1000));
    if (seconds < 45) return "just now";
    const minutes = Math.round(seconds / 60);
    if (minutes < 60) return `${minutes} min ago`;
    const hours = Math.round(minutes / 60);
    if (hours < 24) return `${hours} ${hours === 1 ? "hour" : "hours"} ago`;
    const days = Math.round(hours / 24);
    return `${days} ${days === 1 ? "day" : "days"} ago`;
  };

  const countdown = (): number => {
    const p = pair();
    return p ? Math.max(0, Math.floor((p.expiresAt - now()) / 1000)) : 0;
  };

  const countdownLabel = (): string => {
    const s = countdown();
    if (s <= 0) return "Expired";
    const m = Math.floor(s / 60);
    return m > 0 ? `${m}:${String(s % 60).padStart(2, "0")}` : `${s}s`;
  };

  const passwordFields = (submitLabel: string): JSX.Element => (
    <div class="phone__field-grid">
      <label class="phone__field">
        <span>Password</span>
        <input
          class="input"
          type="password"
          autocomplete="off"
          aria-label="Phone access password"
          aria-describedby="phone-strength"
          value={password()}
          onInput={(e) => setPassword(e.currentTarget.value)}
        />
      </label>
      <label class="phone__field">
        <span>Confirm</span>
        <input
          class="input"
          type="password"
          autocomplete="off"
          aria-label="Phone access password again"
          value={confirm()}
          onInput={(e) => setConfirm(e.currentTarget.value)}
        />
      </label>
      <p class="phone__strength" id="phone-strength" data-level={strength().level}>
        {strength().label}
      </p>
      <div class="phone__actions">
        <button class="btn btn--tertiary btn--sm" type="button" onClick={generate}>
          Generate
        </button>
        <button class="btn btn--primary btn--sm" type="submit" disabled={savingPassword()} aria-busy={savingPassword()}>
          {submitLabel}
        </button>
      </div>
      <Show when={generated()}>
        {(value) => (
          <div class="phone__generated">
            <code class="mono phone__generated-value">{value()}</code>
            <button class="btn btn--tertiary btn--sm" type="button" onClick={() => void copy(value(), "Type it into the password fields on the phone.")}>
              Copy
            </button>
            <p class="hint">Shown here only until it is set.</p>
          </div>
        )}
      </Show>
    </div>
  );

  return (
    <Section
      id="phone"
      title="Phone access"
      description="Watch and use the model from a phone over the local Wi-Fi. Nothing is reachable from the internet."
    >
      <div class="phone" data-testid="phone-access">
        <div class="phone__status-row">
          <p class="phone__status" data-state={statusState()} data-testid="phone-status">
            <span class="phone__dot" aria-hidden="true" />
            {statusText()}
          </p>
        </div>

        <Show
          when={view()?.passwordSet}
          fallback={
            <div class="phone__card">
              <h3 class="phone__title">Set a password first</h3>
              <p class="hint">Phone access stays off until there is a password to ask for.</p>
              <form class="phone__form" onSubmit={submitPassword}>
                {passwordFields("Set password")}
              </form>
            </div>
          }
        >
          <details class="phone__disclosure">
            <summary>Change password</summary>
            <p class="hint phone__warn">Changing it signs out every phone.</p>
            <form class="phone__form" onSubmit={submitPassword}>
              {passwordFields("Change password")}
            </form>
          </details>
        </Show>

        <SettingSwitch
          id="phone-enabled"
          label="Phone access"
          note="A small server listens on this computer only, on the local network. Off is a real off: no port is open."
          checked={view()?.enabled ?? false}
          disabled={!view()?.passwordSet}
          onChange={(on) => void toggle(on)}
        />

        <Show when={view()?.running}>
          <div class="phone__pair">
            <div class="phone__pair-actions">
              <button class="btn btn--primary btn--sm" type="button" onClick={() => void pairPhone()} disabled={pairing()} aria-busy={pairing()}>
                Pair a phone
              </button>
              <Show when={pair()}>
                <button class="btn btn--tertiary btn--sm" type="button" onClick={() => void pairPhone()} disabled={pairing()} aria-busy={pairing()}>
                  New code
                </button>
              </Show>
            </div>
            <Show when={pair()} keyed>
              {(code) => (
                <div class="phone__code">
                  <figure class="phone__qr">
                    <div class="phone__qr-card">
                      <img
                        class="phone__qr-img"
                        src={"data:image/svg+xml;charset=utf-8," + encodeURIComponent(code.qrSvg)}
                        alt="QR code containing the pairing link"
                        width="200"
                        height="200"
                        data-testid="phone-qr"
                      />
                    </div>
                    <figcaption class="phone__qr-caption">Expires in {countdownLabel()}</figcaption>
                  </figure>
                  <div class="phone__code-text">
                    <code class="mono phone__url" data-testid="phone-url">
                      {code.url}
                    </code>
                    <div class="phone__pair-actions">
                      <button class="btn btn--tertiary btn--sm" type="button" onClick={() => void copy(code.url, "Open it on a phone on the same network.")}>
                        Copy
                      </button>
                    </div>
                    <p class="hint">
                      The phone warns about the certificate the first time; it is made on this computer. Fingerprint:{" "}
                      <span class="mono">{view()?.fingerprint ?? "not available yet"}</span>
                    </p>
                    <p class="hint">Windows may ask to allow Cerebro on private networks. Allow it for private networks only.</p>
                  </div>
                </div>
              )}
            </Show>
          </div>
        </Show>

        <Show when={(view()?.devices.length ?? 0) > 0}>
          <div class="phone__devices">
            <h3 class="phone__title">Paired phones</h3>
            <ul class="phone__device-list">
              <For each={view()?.devices}>
                {(device) => (
                  <li class="phone__device" data-testid={`phone-device-${device.id}`}>
                    <div class="phone__device-text">
                      <span class="phone__device-name">{device.name}</span>
                      <span class="mono phone__device-ip">{device.ip}</span>
                      <span class="hint">last seen {relative(device.lastSeenAt)}</span>
                    </div>
                    <button class="btn btn--tertiary btn--sm" type="button" onClick={() => void revoke(device)}>
                      Revoke
                    </button>
                  </li>
                )}
              </For>
            </ul>
            <Show when={(view()?.devices.length ?? 0) > 1}>
              <button class="btn btn--tertiary btn--sm" type="button" onClick={() => void revoke(null)}>
                Sign out all phones
              </button>
            </Show>
          </div>
        </Show>

        <details class="phone__advanced">
          <summary>Advanced</summary>
          <div class="phone__options">
            <SettingSwitch
              id="phone-chat"
              label="Allow chat from the phone"
              note="Off: the phone only sees status and can stop a reply."
              checked={view()?.allowChat ?? false}
              onChange={(on) =>
                void saveOptions({ allowChat: on }, on ? "Chat from the phone is on" : "Chat from the phone is off", on ? "The phone can start and send replies." : "The phone can only watch and stop a reply.")
              }
            />
            <SettingSwitch
              id="phone-tailscale"
              label="Also allow Tailscale addresses"
              note="Lets a phone reach this computer through a Tailscale network."
              checked={view()?.allowTailscale ?? false}
              onChange={(on) =>
                void saveOptions(
                  { allowTailscale: on },
                  on ? "Tailscale addresses allowed" : "Tailscale addresses not allowed",
                  on ? "Cerebro also answers on its Tailscale address." : "Cerebro answers on the local network only.",
                )
              }
            />
            <label class="phone__field">
              <span>Address</span>
              <select
                aria-label="Address the phone server answers on"
                value={view()?.addressChoice ?? ""}
                onChange={(e) => {
                  const next = e.currentTarget.value || null;
                  void saveOptions({ address: next }, "Address saved", next ? `Cerebro answers on ${next}.` : "Cerebro picks the address automatically.");
                }}
              >
                <option value="">Automatic</option>
                <For each={view()?.addresses ?? []}>{(address) => <option value={address}>{address}</option>}</For>
              </select>
            </label>
            <label class="phone__field">
              <span>Sign phones out after N idle days (0 = never)</span>
              <input
                class="input"
                type="number"
                min="0"
                max="3650"
                step="1"
                inputmode="numeric"
                value={view()?.sessionDays ?? 0}
                onChange={(e) => {
                  const days = Math.min(3650, Math.max(0, Math.round(Number(e.currentTarget.value) || 0)));
                  void saveOptions(
                    { sessionDays: days },
                    "Idle timeout saved",
                    days === 0 ? "Phones stay signed in until revoked." : `Phones are signed out after ${days} idle ${days === 1 ? "day" : "days"}.`,
                  );
                }}
              />
            </label>
          </div>
        </details>
        <Show when={service.remoteTunnelStart}>
          <InternetShare running={view()?.running === true} passwordSet={view()?.passwordSet === true} />
        </Show>
      </div>
    </Section>
  );
}

/**
 * Sharing over the internet: a Cloudflare quick tunnel to this page's server. The public link is
 * new each time; whoever opens it also needs the one-time sign-in link and the password.
 */
function InternetShare(props: { running: boolean; passwordSet: boolean }): JSX.Element {
  const service = useService();
  const [busy, setBusy] = createSignal(false);
  const [link, setLink] = createSignal<{ url: string; pairUrl: string } | null>(null);
  const [error, setError] = createSignal<string | null>(null);
  void service.remoteTunnelGet?.().then((u) => u && setLink({ url: u, pairUrl: "" }));
  const start = async () => {
    setBusy(true);
    setError(null);
    try {
      setLink(await service.remoteTunnelStart!());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };
  const stop = async () => {
    await service.remoteTunnelStop?.();
    setLink(null);
  };
  const copy = (text: string) => void navigator.clipboard.writeText(text).then(() => notify("Copied", undefined, "success"));
  return (
    <div class="phone__internet card">
      <h3 class="phone__internet-title">Share over the internet</h3>
      <p class="hint">
        A public link through a Cloudflare tunnel for chatting with this computer's model from anywhere. Whoever opens it still needs the one-time sign-in link and the password; wrong
        passwords lock it for a while. The link changes every time and stops working when sharing stops or Cerebro closes.
      </p>
      <Show
        when={link()}
        fallback={
          <>
            <button type="button" class="btn btn--primary btn--sm" disabled={busy() || !props.running || !props.passwordSet} onClick={() => void start()}>
              {busy() ? "Opening the tunnel…" : "Create a public link"}
            </button>
            <Show when={!props.running || !props.passwordSet}>
              <p class="hint">{!props.passwordSet ? "Set a password above first." : "Turn phone access on first."}</p>
            </Show>
          </>
        }
      >
        {(l) => (
          <div class="phone__internet-link">
            <label class="phone__field">
              <span>Public link</span>
              <span class="phone__internet-row">
                <input class="input mono" readonly value={l().url} />
                <button type="button" class="btn btn--secondary btn--sm" onClick={() => copy(l().url)}>
                  Copy
                </button>
              </span>
            </label>
            <Show when={l().pairUrl}>
              <label class="phone__field">
                <span>One-time sign-in link (send it once; it expires in a few minutes)</span>
                <span class="phone__internet-row">
                  <input class="input mono" readonly value={l().pairUrl} />
                  <button type="button" class="btn btn--secondary btn--sm" onClick={() => copy(l().pairUrl)}>
                    Copy
                  </button>
                </span>
              </label>
            </Show>
            <button type="button" class="btn btn--tertiary btn--sm" onClick={() => void stop()}>
              Stop sharing
            </button>
          </div>
        )}
      </Show>
      <Show when={error()}>
        <p class="phone__error" role="alert">
          {error()}
        </p>
      </Show>
    </div>
  );
}
