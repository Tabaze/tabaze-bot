import { MessageRole } from '../enums/message-role.enum.js';

export { MessageRole };

export interface Message {
  readonly role: MessageRole;
  readonly content: string;
}

export type MessageList = readonly Message[];

export function systemMessage(content: string): Message {
  return { role: MessageRole.System, content };
}

export function userMessage(content: string): Message {
  return { role: MessageRole.User, content };
}

export function assistantMessage(content: string): Message {
  return { role: MessageRole.Assistant, content };
}
