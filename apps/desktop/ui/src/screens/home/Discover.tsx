import "../../styles/home-discover.css";
import { createSignal, For, Show, type JSX } from "solid-js";
import { Icon, type IconName } from "../../components/icons";
import { Section } from "../../components/Screen";
import { loadConversations } from "../../lib/chat/store";
import { loadGallery } from "../../lib/images";
import { loadInstalled } from "../../lib/plugins";
import { hrefFor, type RouteId } from "../../lib/routes";
import { readPref, writePref } from "../../lib/storage";

/**
 * The discovery half of Home: quick actions, a feature spotlight that changes every day, what's
 * new, a first-steps checklist that ticks itself from what the user has done, and recent activity.
 * Loaded after first paint so Home's frame stays light.
 */

interface Spotlight {
  title: string;
  text: string;
  route: RouteId;
  icon: IconName;
  cta: string;
}

const SPOTLIGHTS: readonly Spotlight[] = [
  { title: "Watch the experts work", text: "Monitor shows how many of a MoE model's experts fire per token and how often the graphics card's cache already has them.", route: "monitor", icon: "activity", cta: "Open Monitor" },
  { title: "Builds made for this machine", text: "Turbo compiles the engine for your exact processor and graphics card, and keeps it only if it beats the official release here.", route: "turbo", icon: "bolt", cta: "Open Turbo" },
  { title: "Native reasoning, no added prompt", text: "Chat can use a model's own reasoning control with nothing added to your system prompt, or Cerebro's verification passes, or both.", route: "chat", icon: "chat", cta: "Try it in Chat" },
  { title: "Images, built in", text: "Image Studio makes and edits images with FLUX, SD 3.5 and Qwen-Image on this computer. No plugin needed.", route: "images", icon: "images", cta: "Open Image Studio" },
  { title: "Ask your own files", text: "Drop PDFs and notes into Documents and ask questions; the answering passage comes back with its file.", route: "documents", icon: "documents", cta: "Open Documents" },
  { title: "Every action, your call", text: "Tools preview the exact command, diff or URL before they run, and you decide per tool: ask, allow for the session, always, or never.", route: "tools", icon: "tools", cta: "Set tool rules" },
  { title: "Soak-tested speed", text: "Tune runs the fastest settings for minutes, not seconds, and rejects any that slowly run out of graphics memory.", route: "tune", icon: "tune", cta: "Run a tune" },
];

const NEWS = [
  { date: "Oct 9", text: "Image Studio: inpainting with a brush mask and 4× upscaling." },
  { date: "Oct 8", text: "Turbo: machine profiles with measured before and after, and one-click revert." },
  { date: "Oct 7", text: "Monitor: MoE expert share, cache hits and draft acceptance, live." },
  { date: "Oct 5", text: "Chat: reasoning source per message, including the raw model with no added prompt." },
];

const DONE_KEY = "cerebro-checklist-dismissed";

export function dayIndex(now: Date, n: number): number {
  const day = Math.floor(Date.UTC(now.getFullYear(), now.getMonth(), now.getDate()) / 86_400_000);
  return ((day % n) + n) % n;
}

export function Discover(props: { tuned: boolean; models: number }): JSX.Element {
  const [offset, setOffset] = createSignal(0);
  const spot = () => SPOTLIGHTS[(dayIndex(new Date(), SPOTLIGHTS.length) + offset()) % SPOTLIGHTS.length]!;
  const chats = loadConversations();
  const gallery = loadGallery();
  const plugins = loadInstalled();
  const [dismissed, setDismissed] = createSignal(readPref(DONE_KEY) === "1");

  const steps = [
    { id: "model", label: "Get your first model", done: props.models > 0, route: "hub" as RouteId },
    { id: "tune", label: "Run your first tune", done: props.tuned, route: "tune" as RouteId },
    { id: "chat", label: "Start a chat", done: chats.some((c) => c.messages.length > 0), route: "chat" as RouteId },
    { id: "image", label: "Make an image", done: gallery.length > 0, route: "images" as RouteId },
    { id: "plugin", label: "Install a plugin", done: Object.keys(plugins.installed).length + plugins.local.length > 2, route: "plugins" as RouteId },
  ];
  const doneCount = steps.filter((s) => s.done).length;

  const recent = [
    ...chats.slice(0, 3).map((c) => ({ icon: "chat" as IconName, text: c.title, route: "chat" as RouteId, at: c.updatedAt })),
    ...gallery.slice(0, 2).map((g) => ({ icon: "images" as IconName, text: g.prompt, route: "images" as RouteId, at: g.at })),
  ]
    .sort((a, b) => b.at - a.at)
    .slice(0, 5);

  return (
    <>
      <Section id="quick" title="Jump in">
        <div class="quick">
          <a class="quick__action" href={hrefFor("chat")} data-press>
            <Icon name="chat" size={20} />
            <span>New chat</span>
          </a>
          <a class="quick__action" href={hrefFor("tune")} data-press>
            <Icon name="tune" size={20} />
            <span>Tune a model</span>
          </a>
          <a class="quick__action" href={hrefFor("hub")} data-press>
            <Icon name="plus" size={20} />
            <span>Add a model</span>
          </a>
          <a class="quick__action" href={hrefFor("images")} data-press>
            <Icon name="images" size={20} />
            <span>Make an image</span>
          </a>
        </div>
      </Section>

      <Section id="spotlight" title="Did you know?">
        <article class="spotlight card" aria-live="polite">
          <span class="spotlight__icon" aria-hidden="true">
            <Icon name={spot().icon} size={24} />
          </span>
          <div class="spotlight__body">
            <h3>{spot().title}</h3>
            <p>{spot().text}</p>
            <div class="spotlight__actions">
              <a class="btn btn--primary btn--sm" href={hrefFor(spot().route)} data-press>
                {spot().cta}
              </a>
              <button type="button" class="btn btn--tertiary btn--sm" onClick={() => setOffset((o) => o + 1)}>
                Show another
              </button>
            </div>
          </div>
        </article>
      </Section>

      <div class="home-columns">
        <Show when={!dismissed() && doneCount < steps.length}>
          <Section id="checklist" title={`First steps · ${doneCount} of ${steps.length}`}>
            <ol class="checklist" role="list">
              <For each={steps}>
                {(s) => (
                  <li class="checklist__item" data-done={s.done}>
                    <span class="checklist__mark" aria-hidden="true">
                      <Show when={s.done}>
                        <Icon name="check" size={12} />
                      </Show>
                    </span>
                    <Show when={!s.done} fallback={<span class="checklist__label">{s.label}</span>}>
                      <a class="checklist__label text-link" href={hrefFor(s.route)}>
                        {s.label}
                      </a>
                    </Show>
                    <span class="sr-only">{s.done ? "Done" : "Not done yet"}</span>
                  </li>
                )}
              </For>
            </ol>
            <button
              type="button"
              class="btn btn--tertiary btn--sm"
              onClick={() => {
                writePref(DONE_KEY, "1");
                setDismissed(true);
              }}
            >
              Hide first steps
            </button>
          </Section>
        </Show>

        <Section id="news" title="What's new">
          <ul class="news" role="list">
            <For each={NEWS}>
              {(n) => (
                <li>
                  <span class="mono news__date">{n.date}</span>
                  <span>{n.text}</span>
                </li>
              )}
            </For>
          </ul>
        </Section>

        <Section id="recent" title="Recent">
          <ul class="recent" role="list">
            <For each={recent} fallback={<li class="hint">Chats and images you make show up here.</li>}>
              {(r) => (
                <li>
                  <Icon name={r.icon} size={14} />
                  <a class="text-link recent__text" href={hrefFor(r.route)}>
                    {r.text}
                  </a>
                </li>
              )}
            </For>
          </ul>
        </Section>
      </div>
    </>
  );
}
