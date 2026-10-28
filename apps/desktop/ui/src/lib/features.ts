import type { IconName } from "../components/icons";
import type { RouteId } from "./routes";
import { COMMANDS } from "./chat/commands";

/**
 * What Cerebro does, for the feature guide that opens on first entry (components/FeatureGuide.tsx)
 * and from the command palette. Only what this build really does: each entry names the screen it
 * lives on so "Open" can take the user there. Bump GUIDE_VERSION when the list changes so the
 * guide shows once more.
 */

export { GUIDE_SEEN_KEY, GUIDE_VERSION } from "./guide";

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
      { title: "Cluster: one model, several computers", text: "Share a computer with a pairing code, then run a model across several computers over Wi-Fi, Ethernet or a USB network cable. The reply is the same as on one machine.", icon: "link", route: "cluster", isNew: true },
      { title: "Tune for this machine", text: "Measures the fastest settings for a model on this hardware and keeps them.", icon: "tune", route: "tune" },
      { title: "Monitor", text: "Speed, memory, temperature and every request, live.", icon: "activity", route: "monitor" },
      { title: "Benchmark lab", text: "Measure a model on this computer: prompt reading and writing speed at three lengths, first-token time, memory and twenty exact questions, with history and a report card. /bench opens it.", icon: "tune", route: "monitor", isNew: true },
    ],
  },
  {
    title: "Chat",
    features: [
      { title: "Speed or accuracy, one slider", text: "From a one-line prompt with greedy decoding on the graphics card, to long thinking with review passes, to Orchestrate. Moving it shows what changed.", icon: "bolt", route: "chat", isNew: true },
      { title: "/ commands", text: `Type / in a message for ${COMMANDS.length} commands: switch model, change speed, answer from documents, make an image, read aloud, export, and more. They run here and never reach the model.`, icon: "code", route: "chat", isNew: true },
      { title: "Answer from documents", text: "Turn on \"Use my documents\" and the passages that match each message are sent with it, numbered. The reply shows the files it used, and a chip opens the passage.", icon: "documents", route: "chat", isNew: true },
      { title: "Arena", text: "One question, two to four models side by side, names hidden until the vote; votes build a leaderboard per kind of task. /arena opens it.", icon: "chat", route: "chat", isNew: true },
      { title: "Settings per model", text: "Save the current settings for a model from the live panel; they return whenever that model is picked.", icon: "sliders", route: "chat", isNew: true },
      { title: "Live panel", text: "Speed, context, graphics memory, temperature and power beside the chat, with the most-changed settings. Ctrl . opens it.", icon: "activity", route: "chat", isNew: true },
      { title: "Cloud models with own keys", text: "Adds models paid for elsewhere. Keys stay in the Windows Credential Manager, and the chat says clearly when it goes to the cloud.", icon: "online", route: "settings", isNew: true },
      { title: "Thinking and review passes", text: "Reasoning models think within a set budget, and answers can be checked and corrected before they are shown.", icon: "check", route: "chat", isNew: true },
      { title: "Branches, search and saved prompts", text: "Edit an earlier message or write a reply again to branch the chat and switch between versions; search every chat at once; keep prompts with fill-in fields.", icon: "search", route: "chat", isNew: true },
      { title: "Orchestrate", text: "The model plans, splits the work across fresh instances and checks it through a shared ledger, for hard problems.", icon: "plugins", route: "chat" },
      { title: "Images in chat", text: "Attach an image and a vision model describes or answers about it.", icon: "images", route: "chat" },
      { title: "Voice mode", text: "Talk hands-free: it listens until a pause, answers aloud, and listens again. Speech is recognised and spoken on this computer.", icon: "mic", route: "chat", isNew: true },
      { title: "Talk and listen", text: "Push to talk with speech recognition on this computer, and replies read aloud.", icon: "chat", route: "chat", isNew: true },
      { title: "Tools and the MCP library", text: "Models call tools and MCP servers, each behind a permission rule. Browse a checked list of servers, or search everything in the official MCP registry and add one in a click.", icon: "tools", route: "tools", isNew: true },
    ],
  },
  {
    title: "Create",
    features: [
      { title: "Image Studio", text: "Make a 512 x 512 image in about a fifth of a second on a recent NVIDIA card, with the seed kept for remaking it.", icon: "images", route: "images", isNew: true },
      { title: "Documents", text: "Searches local files on this machine.", icon: "documents", route: "documents" },
      { title: "Documents as collections", text: "Group indexed files in named collections, and let Chat search one collection instead of everything.", icon: "folder", route: "documents", isNew: true },
      { title: "Converters", text: "Turn PDF, Word, HTML, Markdown, CSV, JSON and plain text into clean text on this computer, and save the result as Markdown or plain text.", icon: "download", route: "documents", isNew: true },
      { title: "Quick tools", text: "Summarise, translate, rewrite, extract, explain code, compare two documents, take meeting notes, read aloud and describe an image. One step, one answer.", icon: "bolt", route: "documents", isNew: true },
      { title: "Train", text: "Fine-tunes a model on custom examples.", icon: "chip", route: "train" },
    ],
  },
  {
    title: "Local by design",
    features: [
      { title: "Private by default", text: "Models run here. Only started actions use the network: downloads, allowed tools and features that are turned on.", icon: "lock", route: "settings" },
      { title: "Every protection is a switch", text: "Settings explains each protection and can turn it off.", icon: "settings", route: "settings" },
    ],
  },
];

