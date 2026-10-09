import { Show, type JSX } from "solid-js";
import { CodeBlock } from "../components/CodeBlock";
import { Icon } from "../components/icons";
import { Preview } from "../components/Preview";
import { ScreenHeader, Section } from "../components/Screen";
import { Term } from "../components/Term";
import { hrefFor } from "../lib/routes";
import { useAppState } from "../state";

/**
 * Screens that arrive in a later build. Each says what it will do, marks itself as planned, and
 * points at what already works today (a real `cerebro` command) where there is one.
 */

function SettingsLink(props: { children: JSX.Element }): JSX.Element {
  return (
    <a class="text-link" href={hrefFor("settings")}>
      {props.children}
    </a>
  );
}

export function Chat(): JSX.Element {
  return (
    <div class="screen">
      <ScreenHeader route="chat" />
      <Preview
        art="chat"
        title="Conversations that never leave this computer"
        lede="Chat and a coding agent, running on the model you tuned, with the speed Cerebro measured for it."
        features={[
          { icon: "chat", title: "Chat", text: "Ask, draft and rewrite with a local model. No account, no upload." },
          { icon: "tools", title: "Coding agent", text: "Reads and edits the files you choose, asking before each action." },
          { icon: "bolt", title: "Tuned speed", text: "Uses the fastest safe settings measured on this machine." },
        ]}
      />
      <Section id="today" title="Works today" description={<>Chat in a terminal now, or serve the model as an <Term k="api" /> on <Term k="loopback" /> for the tools you already use.</>}>
        <CodeBlock lines={["cerebro chat", "cerebro serve qwen3-coder-30b-a3b-q4km"]} />
      </Section>
    </div>
  );
}

export function ImageStudio(): JSX.Element {
  return (
    <div class="screen">
      <ScreenHeader route="images" />
      <Preview
        art="gallery"
        title="Make images on your own hardware"
        lede="Generate and edit images with models that run on this machine. Nothing you make is uploaded anywhere."
        features={[
          { icon: "images", title: "Generate", text: "Describe an image and refine it in a few passes." },
          { icon: "tune", title: "Measured settings", text: "Settings are measured on this machine, not guessed." },
          { icon: "lock", title: "Private by design", text: "Prompts and results stay in a folder you choose." },
        ]}
      />
    </div>
  );
}

export function Documents(): JSX.Element {
  return (
    <div class="screen">
      <ScreenHeader route="documents" />
      <Preview
        art="documents"
        title="Ask questions about your own files"
        lede={
          <>
            Point Cerebro at a folder and ask. Answers come from your files through <Term k="rag" />, with the passages they used.
          </>
        }
        features={[
          { icon: "documents", title: "Your folders", text: "Indexes the folders you pick, and nothing else." },
          { icon: "search", title: "Cited answers", text: "Every answer links to the passages it came from." },
          { icon: "offline", title: "Local index", text: "The search index is stored on this machine and works offline." },
        ]}
      />
    </div>
  );
}

export function Plugins(): JSX.Element {
  const { settings } = useAppState();
  return (
    <div class="screen">
      <ScreenHeader route="plugins" />
      <Preview
        art="plugins"
        title="Add capabilities on your terms"
        lede="Plugins add tools and screens. Installing one needs a network; an installed plugin runs offline."
        features={[
          { icon: "plugins", title: "Install and remove", text: "Each plugin is one switch, off as easily as on." },
          { icon: "security", title: "Declared access", text: "Each plugin says what it needs before you install it." },
          { icon: "offline", title: "Runs offline", text: "After installing, a plugin never needs the network again." },
        ]}
      >
        <Show when={settings()?.allowPluginInstalls === false}>
          <p class="hint">
            <Icon name="info" size={16} />
            <span>
              Plugin installs are switched off right now. <SettingsLink>Change this in Settings</SettingsLink>.
            </span>
          </p>
        </Show>
      </Preview>
    </div>
  );
}

export function Tools(): JSX.Element {
  const { settings } = useAppState();
  const promptsOn = () => settings()?.permissionPrompts !== false;
  return (
    <div class="screen">
      <ScreenHeader route="tools" />
      <Preview
        art="tools"
        title="Actions a model takes only with your say-so"
        lede="Tools let a model act, such as reading a file you chose. Each tool is listed here with exactly what it can reach."
        features={[
          { icon: "tools", title: "One list", text: "Every tool a model can call, and what it can reach." },
          { icon: "lock", title: "Permission prompts", text: "With prompts on, Cerebro asks before each action." },
          { icon: "settings", title: "Your call", text: "Allow a tool once, always, or never." },
        ]}
      >
        <p class="hint">
          <Icon name="info" size={16} />
          <span>
            Permission prompts are <strong>{promptsOn() ? "on" : "off"}</strong>. <SettingsLink>Change this in Settings</SettingsLink>.
          </span>
        </p>
      </Preview>
    </div>
  );
}
