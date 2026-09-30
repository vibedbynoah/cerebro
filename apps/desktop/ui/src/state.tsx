import { createContext, createSignal, onMount, useContext, type Accessor, type JSX } from "solid-js";
import { useService } from "./service/context";
import type { NetworkState, SettingKey, Settings } from "./service/types";
import { notify } from "./lib/toast";

/** Settings and network state shared by the status bar, Security and Settings. */
export interface AppState {
  settings: Accessor<Settings | null>;
  network: Accessor<NetworkState | null>;
  setSetting<K extends SettingKey>(key: K, value: Settings[K], confirmation: string): Promise<void>;
}

const StateContext = createContext<AppState>();

export function AppStateProvider(props: { children: JSX.Element }): JSX.Element {
  const service = useService();
  const [settings, setSettings] = createSignal<Settings | null>(null);
  const [network, setNetwork] = createSignal<NetworkState | null>(null);

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

  const state: AppState = {
    settings,
    network,
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
