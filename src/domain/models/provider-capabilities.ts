/**
 * Advertises which normalized GenerationConfig fields a given
 * provider/model combination actually honors. The router and application
 * layer use this to avoid sending parameters a provider would reject or
 * silently misinterpret.
 */
export interface ProviderCapabilities {
  readonly streaming: boolean;
  readonly temperature: boolean;
  readonly topP: boolean;
  readonly presencePenalty: boolean;
  readonly frequencyPenalty: boolean;
  readonly stopSequences: boolean;
  readonly systemMessages: boolean;
  readonly toolCalls: boolean;
  readonly vision: boolean;
}

export const DEFAULT_CAPABILITIES: ProviderCapabilities = {
  streaming: true,
  temperature: true,
  topP: true,
  presencePenalty: false,
  frequencyPenalty: false,
  stopSequences: true,
  systemMessages: true,
  toolCalls: false,
  vision: false,
};
