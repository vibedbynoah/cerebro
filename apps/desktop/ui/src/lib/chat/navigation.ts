import { createSignal } from "solid-js";
import type { Conversation } from "./store";

/**
 * Small pieces of state shared by the shell and the Chat screen: the global chat-search dialog,
 * a request to open a chat at a message, and the chats Chat currently has in memory so search sees
 * a just-sent message before the debounced save reaches local storage.
 */

export interface ChatJump {
  conversationId: string;
  messageId: string | null;
  query: string;
}

export const [chatSearchOpen, setChatSearchOpen] = createSignal(false);
export const [chatJump, setChatJump] = createSignal<ChatJump | null>(null);
export const [liveChats, setLiveChats] = createSignal<Conversation[] | null>(null);

export function publishChats(list: readonly Conversation[]): void {
  setLiveChats([...list]);
}
