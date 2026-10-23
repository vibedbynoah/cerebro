import { createContext, createEffect, createSignal, onMount, useContext, type Accessor, type JSX } from "solid-js";
import { useService } from "./service/context";
import type { NetworkState, SettingKey, Settings } from "./service/types";
import { notify } from "./lib/toast";
import { readChannel, readLevel, writeChannel, writeLevel, type Channel, type ExperienceLevel } from "./lib/experience";

/** Settings, network state, experience level and channel, shared by setup and every screen. */
export interface AppState {
  settings: Accessor<Settings | null>;
  network: Accessor<NetworkState | null>;
  setSetting<K extends SettingKey>(key: K, value: Settings[K], confirmation: string): Promise<void>;
  /** Beginner or Advanced (lib/experience.ts). Saved the moment it changes, so a switch made in the
   *  middle of setup survives a restart of setup. */
  level: Accessor<ExperienceLevel>;
  setLevel(level: ExperienceLevel): void;
  channel: Accessor<Channel>;
  setChannel(channel: Channel): void;
}

const StateContext = createContext<AppState>();

export function AppStateProvider(props: { children: JSX.Element }): JSX.Element {
  const service = useService();
  const [settings, setSettings] = createSignal<Settings | null>(null);
  const [network, setNetwork] = createSignal<NetworkState | null>(null);
  const [level, setLevelSignal] = createSignal<ExperienceLevel>(readLevel());
  const [channel, setChannelSignal] = createSignal<Channel>(readChannel());

  onMount(() => {
    service
      .getSettings()
      .then(setSettings)
      .catch(() => notify("Settings could not be loaded", "The defaults are in effect until they load.", "error"));
    service
      .getNetwork()
      .then(setNetwork)
      .catch(() => setNetwork("offline"));
  });

  // The engine follows Offline mode (it blocks the llama.cpp download while it is on).
  createEffect(() => {
    const on = settings()?.offlineMode;
    if (on !== undefined) void service.setOffline(on).catch(() => undefined);
  });

  const state: AppState = {
    settings,
    network,
    level,
    setLevel(next) {
      setLevelSignal(next);
      writeLevel(next);
    },
    channel,
    setChannel(next) {
      setChannelSignal(next);
      writeChannel(next);
    },
    async setSetting(key, value, confirmation) {
      const before = settings();
      if (before) setSettings({ ...before, [key]: value }); // optimistic: the switch moves at once
      try {
        setSettings(await service.updateSetting(key, value));
        notify(confirmation);
      } catch {
        setSettings(before);
        notify("That change was not saved", "The setting is unchanged. Try again.", "error");
      }
    },
  };

  return <StateContext.Provider value={state}>{props.children}</StateContext.Provider>;
}

export function useAppState(): AppState {
  const state = useContext(StateContext);
  if (!state) throw new Error("useAppState() called outside <AppStateProvider>");
  return state;
}
