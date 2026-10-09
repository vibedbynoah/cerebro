import "../styles/security.css";
import { createSignal, For, onCleanup, Show, type JSX } from "solid-js";
import { Icon } from "../components/icons";
import { IllustrativeFlag, ScreenHeader, Section } from "../components/Screen";
import { Term } from "../components/Term";
import { hrefFor } from "../lib/routes";
import { SETTINGS_META } from "../lib/settings-meta";
import { notify } from "../lib/toast";
import { ACTIVITY, FOLDER_AUDIT, generateKey, MANIFEST, maskKey, SANDBOX, UPDATE, type ActivityEntry } from "../mock/security";
import { useAppState } from "../state";

/**
 * Security: what protects the model process, the evidence for each protection, and the API key.
 * Sensitive values (hashes, the key) are masked until the user asks; a new key is shown once and
 * held only in this screen's memory, never in storage or logs. Every protection can be switched off
 * in Settings; this screen never argues about it.
 */
export function Security(): JSX.Element {
  const { settings } = useAppState();
  const protections = SETTINGS_META.filter((m) => m.group === "protections");
  const onCount = () => {
    const s = settings();
    return s ? protections.filter((p) => s[p.key]).length : 0;
  };

  return (
    <div class="screen security">
      <ScreenHeader
        route="security"
        actions={
          <>
            <IllustrativeFlag />
            <a class="btn btn--secondary" href={hrefFor("settings")} data-press>
              <Icon name="settings" size={16} />
              Change in Settings
            </a>
          </>
        }
      />
      <Section
        id="protections"
        title={settings() ? `${onCount()} of ${protections.length} protections on` : "Protections"}
        description="Each entry says what it does and, where it matters, what it does not do."
      >
        <ul class="protection-list">
          <For each={protections}>
            {(p) => {
              const on = () => settings()?.[p.key] ?? false;
              return (
                <li class="protection card" data-setting={p.key} data-on={on()}>
                  <span class="protection__icon" aria-hidden="true">
                    <Icon name={on() ? "security" : "lock"} size={20} />
                  </span>
                  <div class="protection__text">
                    <h3 class="protection__title">
                      <Show when={p.term} fallback={p.label}>
                        {(t) => <Term k={t()}>{p.label}</Term>}
                      </Show>
                    </h3>
                    <p class="protection__note">{p.note}</p>
                    <Show when={p.limits}>
                      <p class="protection__limits">
                        <span class="protection__limits-label">Limits:</span> {p.limits}
                      </p>
                    </Show>
                  </div>
                  <span class="chip" data-tone={on() ? "on" : "off"}>
                    {on() ? "On" : "Off"}
                  </span>
                </li>
              );
            }}
          </For>
        </ul>
      </Section>

      <Integrity on={settings()?.integrityCheck ?? true} />
      <FolderAudit />
      <Sandbox on={settings()?.processContainment ?? true} />
      <ApiKey />
      <Updates />
      <Activity />
    </div>
  );
}

function Integrity(props: { on: boolean }): JSX.Element {
  const [shown, setShown] = createSignal(false);
  return (
    <Section id="integrity" title="Backend integrity" description={<>The engine's files are checked against a signed manifest before every start (<Term k="integrity" />).</>}>
      <div class="sec-card card">
        <div class="sec-card__status" data-ok={props.on}>
          <Icon name={props.on ? "check" : "alert"} size={18} />
          <span>{props.on ? `Verified: ${MANIFEST.files.filter((f) => f.ok).length} of ${MANIFEST.files.length} files match` : "Check switched off in Settings: files are not verified before start"}</span>
        </div>
        <dl class="sec-facts">
          <div>
            <dt>Backend</dt>
            <dd>{MANIFEST.backend}</dd>
          </div>
          <div>
            <dt>Manifest signed by</dt>
            <dd class="mono">{MANIFEST.signedBy}</dd>
          </div>
          <div>
            <dt>Last verified</dt>
            <dd>{MANIFEST.verifiedAt}</dd>
          </div>
        </dl>
        <button class="btn btn--tertiary btn--sm" type="button" aria-expanded={shown()} aria-controls="hash-table" onClick={() => setShown(!shown())}>
          {shown() ? "Hide file hashes" : "Show file hashes"}
        </button>
        <Show when={shown()}>
          <table class="sec-table" id="hash-table">
            <thead>
              <tr>
                <th scope="col">File</th>
                <th scope="col">SHA-256</th>
                <th scope="col">Match</th>
              </tr>
            </thead>
            <tbody>
              <For each={MANIFEST.files}>
                {(f) => (
                  <tr>
                    <th scope="row" class="mono">
                      {f.path}
                    </th>
                    <td class="mono sec-hash">{f.sha256}</td>
                    <td>{f.ok ? "Yes" : "No"}</td>
                  </tr>
                )}
              </For>
            </tbody>
          </table>
        </Show>
      </div>
    </Section>
  );
}

function FolderAudit(): JSX.Element {
  const unexpected = FOLDER_AUDIT.grants.filter((g) => !g.expected);
  return (
    <Section id="audit" title="Folder permissions" description="Who can change the engine's files. Anyone else with write access could swap the engine.">
      <div class="sec-card card">
        <div class="sec-card__status" data-ok={unexpected.length === 0}>
          <Icon name={unexpected.length ? "alert" : "check"} size={18} />
          <span>{unexpected.length ? `${unexpected.length} unexpected account${unexpected.length > 1 ? "s" : ""} can write` : "Only you, SYSTEM and Administrators can write"}</span>
        </div>
        <p class="mono sec-path">{FOLDER_AUDIT.folder}</p>
        <ul class="sec-grants" role="list">
          <For each={FOLDER_AUDIT.grants}>
            {(g) => (
              <li data-expected={g.expected}>
                <span>{g.principal}</span>
                <span class="hint">{g.rights}</span>
              </li>
            )}
          </For>
        </ul>
      </div>
    </Section>
  );
}

function Sandbox(props: { on: boolean }): JSX.Element {
  return (
    <Section id="sandbox" title="Process containment" description={<>How the model process is started and ended (<Term k="jobObject" />, <Term k="cleanEnv" />).</>}>
      <div class="sec-card card">
        <div class="sec-card__status" data-ok={props.on}>
          <Icon name={props.on ? "check" : "info"} size={18} />
          <span>{props.on ? "The model process and anything it starts end when Cerebro stops" : "Containment switched off in Settings"}</span>
        </div>
        <dl class="sec-facts">
          <div>
            <dt>Ends with Cerebro</dt>
            <dd>{SANDBOX.jobObject.killOnClose ? "Yes" : "No"}</dd>
          </div>
          <div>
            <dt>Child processes may escape</dt>
            <dd>{SANDBOX.jobObject.breakawayBlocked ? "No" : "Yes"}</dd>
          </div>
          <div>
            <dt>Process limit</dt>
            <dd>{SANDBOX.jobObject.activeProcessLimit}</dd>
          </div>
        </dl>
        <p class="hint">
          Environment passed to the model: {SANDBOX.envAllowlist.length} variables; {SANDBOX.envDropped} others are left out.
        </p>
        <ul class="sec-env" role="list">
          <For each={SANDBOX.envAllowlist}>{(v) => <li class="mono">{v}</li>}</For>
        </ul>
      </div>
    </Section>
  );
}

function ApiKey(): JSX.Element {
  // The key exists only in this signal: never written to storage, never logged, gone when the
  // screen closes. The engine keeps only a hash of it.
  const [fresh, setFresh] = createSignal<string | null>(null);
  const [revealed, setRevealed] = createSignal(false);
  const [masked, setMasked] = createSignal("cbr_9f2a…c21e");
  const [confirmCopy, setConfirmCopy] = createSignal(false);
  onCleanup(() => setFresh(null));

  const rotate = () => {
    const k = generateKey();
    setFresh(k);
    setMasked(maskKey(k));
    setRevealed(false);
    setConfirmCopy(false);
    notify("New API key created", "The old key stopped working. Copy the new one now: it is shown only once.");
  };
  const copy = async () => {
    const k = fresh();
    if (!k) return;
    if (!confirmCopy()) {
      setConfirmCopy(true);
      return;
    }
    try {
      await navigator.clipboard.writeText(k);
      notify("API key copied", "Paste it into your tool. Cerebro will not show it again.");
    } catch {
      notify("Clipboard unavailable", "Reveal the key and copy it by hand.", "warning");
    }
    setConfirmCopy(false);
  };

  return (
    <Section id="api-key" title="API key" description={<>Every request to the local <Term k="api" /> needs this key, reads included.</>}>
      <div class="sec-card card">
        <div class="sec-key">
          <span class="mono sec-key__value" data-testid="api-key">
            {fresh() && revealed() ? fresh() : masked()}
          </span>
          <Show when={fresh()}>
            <button class="btn btn--tertiary btn--sm" type="button" onClick={() => setRevealed(!revealed())} aria-pressed={revealed()}>
              <Icon name="eye" size={14} />
              <span class="btn__label">{revealed() ? "Hide" : "Reveal"}</span>
            </button>
            <button class="btn btn--secondary btn--sm" type="button" onClick={() => void copy()} data-press>
              <Icon name="copy" size={14} />
              <span class="btn__label">{confirmCopy() ? "Click again to copy" : "Copy"}</span>
            </button>
          </Show>
          <button class="btn btn--secondary btn--sm" type="button" onClick={rotate} data-press>
            <Icon name="refresh" size={14} />
            <span class="btn__label">{fresh() ? "Rotate again" : "Create new key"}</span>
          </button>
        </div>
        <p class="hint">
          <Show when={fresh()} fallback="The current key is never shown again. Create a new one to see it; the old one stops working at once.">
            Shown once. When you leave this screen it is gone from the app; Cerebro keeps only its hash.
          </Show>
        </p>
      </div>
    </Section>
  );
}

function Updates(): JSX.Element {
  return (
    <Section id="updates" title="Updates" description="Every update is checked against Cerebro's release key before it is installed.">
      <div class="sec-card card">
        <div class="sec-card__status" data-ok={UPDATE.signature === "Valid"}>
          <Icon name="check" size={18} />
          <span>
            Version {UPDATE.current} is current. Signature: {UPDATE.signature.toLowerCase()}.
          </span>
        </div>
        <dl class="sec-facts">
          <div>
            <dt>Channel</dt>
            <dd>{UPDATE.channel}</dd>
          </div>
          <div>
            <dt>Signed by</dt>
            <dd class="mono">{UPDATE.signedBy}</dd>
          </div>
          <div>
            <dt>Last checked</dt>
            <dd>{UPDATE.checkedAt}</dd>
          </div>
        </dl>
      </div>
    </Section>
  );
}

const KIND_ICON: Record<ActivityEntry["kind"], "play" | "check" | "folder" | "lock" | "settings" | "download"> = {
  start: "play",
  integrity: "check",
  audit: "folder",
  key: "lock",
  setting: "settings",
  update: "download",
};

function Activity(): JSX.Element {
  const [q, setQ] = createSignal("");
  const rows = () => ACTIVITY.filter((a) => a.text.toLowerCase().includes(q().toLowerCase().trim()));
  return (
    <Section id="activity" title="Security activity" description="What Cerebro checked and changed, newest first.">
      <input class="sec-search" type="search" placeholder="Filter" aria-label="Filter security activity" value={q()} onInput={(e) => setQ(e.currentTarget.value)} />
      <ol class="sec-log" role="list">
        <For each={rows()} fallback={<li class="hint">Nothing matches.</li>}>
          {(a) => (
            <li>
              <Icon name={KIND_ICON[a.kind]} size={16} />
              <span class="sec-log__at mono">{a.at}</span>
              <span>{a.text}</span>
            </li>
          )}
        </For>
      </ol>
    </Section>
  );
}
