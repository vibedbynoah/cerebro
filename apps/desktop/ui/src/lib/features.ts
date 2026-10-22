import type { IconName } from "../components/icons";
import type { RouteId } from "./routes";

/**
 * What Cerebro does, for the feature guide that opens on first entry (components/FeatureGuide.tsx)
 * and from the command palette. Only what this build really does: each entry names the screen it
 * lives on so "Open" can take the user there. Bump GUIDE_VERSION when the list changes so the
 * guide shows once more.
 */

export const GUIDE_VERSION = "2026-10-22";

export interface Feature {
  title: string;
  text: string;
  icon: IconName;
  route: RouteId;
  /** Shown with a "New" badge. */
  isNew?: boolean;
}

export interface FeatureGroup {
  title: string;
  features: readonly Feature[];
}

export const FEATURE_GROUPS: readonly FeatureGroup[] = [
  {
    title: "Run models on this computer",
    features: [
      { title: "Its own inference engine", text: "Cerebro runs models with its own engine on NVIDIA cards, other graphics cards and the processor, and splits a model between them when it does not fit.", icon: "chip", route: "models" },
      { title: "Model Hub", text: "Recommended models ranked by how well they fit this computer, browse by task, search everything on Hugging Face, and download with progress, stop and resume. A Verified badge marks publishers on Cerebro's checked list, and each model's licence is shown first.", icon: "hub", route: "hub", isNew: true },
      { title: "Cluster: one model, several computers", text: "Share a computer with a pairing code, then run a model across your computers over Wi-Fi, Ethernet or a USB network cable. The reply is the same as on one machine.", icon: "link", route: "cluster", isNew: true },
      { title: "Tune for this machine", text: "Measure the fastest settings for a model on your hardware and keep them.", icon: "tune", route: "tune" },
      { title: "Monitor", text: "Speed, memory, temperature and every request, live.", icon: "activity", route: "monitor" },
      { title: "Benchmark lab", text: "Measure a model on this computer: prompt reading and writing speed at three lengths, first-token time, memory and twenty exact questions, with history and a report card. /bench opens it.", icon: "tune", route: "monitor", isNew: true },
    ],
  },
  {
    title: "Chat",
    features: [
      { title: "Speed or accuracy, one slider", text: "From a one-line prompt with greedy decoding on the graphics card, to long thinking with review passes, to Orchestrate. Moving it tells you what changed.", icon: "bolt", route: "chat", isNew: true },
      { title: "/ commands", text: "Type / in a message for 19 commands: switch model, change speed, make an image, read aloud, export, and more. They run here and never reach the model.", icon: "code", route: "chat", isNew: true },
      { title: "Arena", text: "One question, two to four models side by side, names hidden until you vote; your votes build a leaderboard per kind of task. /arena opens it.", icon: "chat", route: "chat", isNew: true },
      { title: "Settings per model", text: "Save the current settings for a model from the live panel; they come back whenever you pick that model.", icon: "sliders", route: "chat", isNew: true },
      { title: "Live panel", text: "Speed, context, graphics memory, temperature and power beside the chat, with the settings you change most. Ctrl . opens it.", icon: "activity", route: "chat", isNew: true },
      { title: "Cloud models with your own keys", text: "Add models you already pay for elsewhere. Keys stay in the Windows Credential Manager, and the chat says clearly when it goes to the cloud.", icon: "online", route: "settings", isNew: true },
      { title: "Thinking and review passes", text: "Reasoning models think within a budget you set, and answers can be checked and corrected before you see them.", icon: "check", route: "chat", isNew: true },
      { title: "Branches, search and saved prompts", text: "Edit an earlier message or write a reply again to branch the chat and switch between versions; search every chat at once; keep prompts with fill-in fields.", icon: "search", route: "chat", isNew: true },
      { title: "Orchestrate", text: "The model plans, splits the work across fresh instances and checks it through a shared ledger, for hard problems.", icon: "plugins", route: "chat" },
      { title: "Images in chat", text: "Attach an image and a vision model describes or answers about it.", icon: "images", route: "chat" },
      { title: "Voice mode", text: "Talk hands-free: it listens until you pause, answers aloud, and listens again. Speech is recognised and spoken on this computer.", icon: "mic", route: "chat", isNew: true },
      { title: "Talk and listen", text: "Push to talk with speech recognition on this computer, and replies read aloud.", icon: "chat", route: "chat", isNew: true },
      { title: "Tools and the MCP library", text: "Models call tools and MCP servers, each behind your permission. Browse a checked list of servers, or search everything in the official MCP registry and add one in a click.", icon: "tools", route: "tools", isNew: true },
    ],
  },
  {
    title: "Create",
    features: [
      { title: "Image Studio", text: "Make a 512 x 512 image in about a fifth of a second on a recent NVIDIA card, with the seed kept so you can make it again.", icon: "images", route: "images", isNew: true },
      { title: "Documents", text: "Search your own files on this machine.", icon: "documents", route: "documents" },
      { title: "Train", text: "Fine-tune a model on your own examples.", icon: "chip", route: "train" },
    ],
  },
  {
    title: "Your machine, your rules",
    features: [
      { title: "Private by default", text: "Models run here. Only what you start uses the network: downloads, tools you allow, and the features you turn on.", icon: "lock", route: "settings" },
      { title: "Every protection is a switch", text: "Settings explains each protection and lets you turn it off.", icon: "settings", route: "settings" },
    ],
  },
];

export const GUIDE_SEEN_KEY = "cerebro-guide-seen";
