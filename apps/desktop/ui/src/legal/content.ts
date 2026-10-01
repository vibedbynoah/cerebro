/**
 * The full Terms of use, Privacy notice and Licenses, bundled into the app so they open offline.
 * Loaded on demand (LegalDialog imports this module lazily), so they cost nothing at start-up.
 *
 * Every statement here must stay true of the shipped code (docs/CLAIMS.md): no telemetry exists in
 * the app, network use is limited to the optional features named, and the protections are
 * described exactly as settings-meta.ts and the glossary describe them, including what they do not do.
 */
import { apache as apacheText, bundled, bundledComplete, rust } from "virtual:cerebro-licenses";
import interOfl from "@design/fonts/inter/OFL.txt?raw";
import frauncesOfl from "@design/fonts/fraunces/OFL.txt?raw";
import jetbrainsOfl from "@design/fonts/jetbrains-mono/OFL.txt?raw";
import { TERMS_VERSION } from "../onboarding/progress";
import type { LegalDoc, LegalDocId } from "./types";

const TERMS: LegalDoc = {
  id: "terms",
  title: "Terms of use",
  summary: `Version ${TERMS_VERSION}. The plain-language summary shown during setup is a short form of these terms.`,
  sections: [
    {
      heading: "The software and its licence",
      blocks: [
        {
          kind: "p",
          text: "Cerebro Engine is free, open-source software licensed under the Apache License, Version 2.0. You may use, copy, change and share it under that licence, whose full text is on the Licenses page. If anything in these terms conflicts with the licence, the licence governs.",
        },
      ],
    },
    {
      heading: "Provided as is",
      blocks: [
        {
          kind: "p",
          text: "Cerebro is provided as is, without warranties or conditions of any kind, express or implied. To the extent the law allows, its contributors are not liable for any damage or loss arising from its use, as set out in sections 7 and 8 of the Apache License.",
        },
      ],
    },
    {
      heading: "Models are yours to choose, and yours to answer for",
      blocks: [
        {
          kind: "p",
          text: "Cerebro does not include a model. Every model you download or add is made by a third party and comes with its own licence, and often an acceptable-use policy. Those terms apply to your use of that model, alongside these. You are responsible for reading and following them.",
        },
      ],
    },
    {
      heading: "Plugins and tools",
      blocks: [
        {
          kind: "p",
          text: "Plugins are code written by other people that you choose to install. Permission prompts ask before a plugin or tool takes an action, unless you switch prompts off. You are responsible for the plugins you install and the actions you allow.",
        },
      ],
    },
    {
      heading: "What models generate",
      blocks: [
        {
          kind: "p",
          text: "Language models can produce text or code that is wrong, harmful, biased or unlawful, and can present it confidently. You are responsible for checking what they generate and for how you use it.",
        },
      ],
    },
    {
      heading: "Your settings",
      blocks: [
        {
          kind: "p",
          text: "Cerebro starts with every protection on. You can switch any of them off in Settings, including process containment, the integrity check, permission prompts and loopback-only serving. When you do, the effect of that change is yours to accept.",
        },
      ],
    },
    {
      heading: "Network features",
      blocks: [
        {
          kind: "p",
          text: "Model downloads and plugin installs are optional and connect only to the source you pick. That source has its own terms. Offline mode, or switching either feature off in Settings, stops them.",
        },
      ],
    },
    {
      heading: "Changes to these terms",
      blocks: [
        {
          kind: "p",
          text: "If these terms change, Cerebro shows the new plain-language summary once and asks you to accept it. Nothing else about your setup changes.",
        },
      ],
    },
  ],
};

const PRIVACY: LegalDoc = {
  id: "privacy",
  title: "Privacy",
  summary: "Cerebro has no accounts, no analytics and no telemetry. What you do stays on this computer.",
  sections: [
    {
      heading: "What stays on this computer",
      blocks: [
        {
          kind: "p",
          text: "Your prompts, chats, documents, generated output, models, settings and tuning measurements are stored on this computer only. Cerebro does not send them anywhere.",
        },
      ],
    },
    {
      heading: "When Cerebro uses a network",
      blocks: [
        {
          kind: "list",
          items: [
            "Only when you use a network feature: downloading a model or installing a plugin.",
            "Only to the source you pick. Like any download, the request shows that source your IP address.",
            "Offline mode switches both features off. Everything else keeps working without a network.",
          ],
        },
      ],
    },
    {
      heading: "The local API",
      blocks: [
        {
          kind: "p",
          text: "The OpenAI-compatible API listens on 127.0.0.1 (loopback) by default. Other devices cannot reach that address, but any program running on this computer can. If you switch off API on loopback only, other devices on your network can connect to it too.",
        },
      ],
    },
    {
      heading: "Plugins",
      blocks: [
        {
          kind: "p",
          text: "A plugin is third-party code and may have its own data practices. Permission prompts ask before a plugin takes an action, unless you switch them off. Check a plugin's own notice before installing it.",
        },
      ],
    },
    {
      heading: "Removing your data",
      blocks: [
        {
          kind: "p",
          text: "Because everything Cerebro stores is on this computer, you stay in control of it: delete a model, a chat or the app's data folders at any time and it is gone. There is no copy anywhere else to request.",
        },
      ],
    },
    {
      heading: "The browser preview",
      blocks: [
        {
          kind: "p",
          text: "The browser preview of this interface shows illustrative data. It does not read your hardware, and it loads nothing except its own files.",
        },
      ],
    },
  ],
};

const LICENSES: LegalDoc = {
  id: "licenses",
  title: "Licenses",
  summary: "Cerebro's own licence, and the open-source work it is built on. The lists of packages are generated from what the build actually ships.",
  sections: [
    {
      heading: "Cerebro Engine",
      blocks: [{ kind: "license", name: "Cerebro Engine", detail: "This app", license: "Apache-2.0", text: apacheText }],
    },
    {
      heading: "Bundled in this interface",
      blocks: [
        {
          kind: "notices",
          id: "bundled",
          title: "JavaScript packages",
          detail: "Every package with code in this interface's files, with the licence texts it publishes.",
          notices: bundled,
          complete: bundledComplete,
        },
        { kind: "license", name: "Inter", detail: "Typeface, via Fontsource", license: "SIL Open Font License 1.1", text: interOfl },
        { kind: "license", name: "Fraunces", detail: "Typeface, via Fontsource", license: "SIL Open Font License 1.1", text: frauncesOfl },
        { kind: "license", name: "JetBrains Mono", detail: "Typeface, via Fontsource", license: "SIL Open Font License 1.1", text: jetbrainsOfl },
      ],
    },
    {
      heading: "Compiled into the desktop app",
      blocks: [
        {
          kind: "notices",
          id: "rust",
          title: "Rust crates",
          detail: "Every crate built into the desktop program, read from its locked dependency list, with the licence texts each one publishes.",
          notices: rust,
          complete: true,
        },
      ],
    },
    {
      heading: "Run by Cerebro",
      blocks: [{ kind: "license", name: "llama.cpp", detail: "The program that runs models. Cerebro configures it; it does not fork or change it.", license: "MIT" }],
    },
    {
      heading: "Models",
      blocks: [
        {
          kind: "p",
          text: "Models are not part of Cerebro. Each one has its own licence from its publisher, shown where you download it. Read it before you use the model: some limit commercial use or require attribution.",
        },
      ],
    },
  ],
};

export const DOCS: Readonly<Record<LegalDocId, LegalDoc>> = { terms: TERMS, privacy: PRIVACY, licenses: LICENSES };
