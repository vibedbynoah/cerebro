import "../../styles/home-discover.css";
import { For, type JSX } from "solid-js";
import { Icon, type IconName } from "../../components/icons";
import { Section } from "../../components/Screen";
import { loadConversations } from "../../lib/chat/store";
import { loadGallery } from "../../lib/images";
import { hrefFor, type RouteId } from "../../lib/routes";

/**
 * Home's actions: the common starting points and the latest chats and images, each a link to where
 * it lives. Loaded after first paint so Home's frame stays light.
 */
export function Discover(): JSX.Element {
  const chats = loadConversations();
  const gallery = loadGallery();
  const actions: readonly { label: string; icon: IconName; route: RouteId }[] = [
    { label: "New chat", icon: "chat", route: "chat" },
    { label: "Add a model", icon: "plus", route: "hub" },
    { label: "Tune a model", icon: "tune", route: "tune" },
    { label: "Make an image", icon: "images", route: "images" },
  ];
  const recent = [
    ...chats.slice(0, 4).map((c) => ({ icon: "chat" as IconName, text: c.title, route: "chat" as RouteId, at: c.updatedAt })),
    ...gallery.slice(0, 2).map((g) => ({ icon: "images" as IconName, text: g.prompt, route: "images" as RouteId, at: g.at })),
  ]
    .sort((a, b) => b.at - a.at)
    .slice(0, 6);

  return (
    <>
      <Section id="quick" title="Start">
        <div class="quick">
          <For each={actions}>
            {(a) => (
              <a class="quick__action" href={hrefFor(a.route)} data-press>
                <Icon name={a.icon} size={20} />
                <span>{a.label}</span>
              </a>
            )}
          </For>
        </div>
      </Section>

      <Section id="recent" title="Recent">
        <ul class="recent" role="list">
          <For each={recent} fallback={<li class="hint">No chats or images yet.</li>}>
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
    </>
  );
}
