import { LLMProvider } from '../../../domain/enums/provider.enum.js';
import { DEFAULT_CAPABILITIES, ProviderCapabilities } from '../../../domain/models/provider-capabilities.js';

export interface ModelDescriptor {
  readonly contextLimit: number;
  readonly capabilities: ProviderCapabilities;
}

/** Applied to any provider/model not found in the registry below. */
const FALLBACK_DESCRIPTOR: ModelDescriptor = {
  contextLimit: 8_192,
  capabilities: DEFAULT_CAPABILITIES,
};

/**
 * Known model capabilities and context windows. This is a registry, not a
 * hard-coded switch statement: unknown provider/model pairs fall back to
 * conservative defaults instead of throwing, so new models work
 * out-of-the-box and can be registered here for tighter accuracy without
 * touching any adapter or router code.
 */
export class ModelRegistry {
  private readonly descriptors = new Map<string, ModelDescriptor>();

  constructor() {
    this.registerDefaults();
  }

  private key(provider: LLMProvider, model: string): string {
    return `${provider}::${model}`;
  }

  register(provider: LLMProvider, model: string, descriptor: ModelDescriptor): void {
    this.descriptors.set(this.key(provider, model), descriptor);
  }

  describe(provider: LLMProvider, model: string): ModelDescriptor {
    return this.descriptors.get(this.key(provider, model)) ?? FALLBACK_DESCRIPTOR;
  }

  getContextLimit(provider: LLMProvider, model: string): number {
    return this.describe(provider, model).contextLimit;
  }

  getCapabilities(provider: LLMProvider, model: string): ProviderCapabilities {
    return this.describe(provider, model).capabilities;
  }

  private registerDefaults(): void {
    const openaiCaps: ProviderCapabilities = {
      ...DEFAULT_CAPABILITIES,
      presencePenalty: true,
      frequencyPenalty: true,
      vision: true,
    };
    this.register(LLMProvider.OpenAI, 'gpt-4o', { contextLimit: 128_000, capabilities: openaiCaps });
    this.register(LLMProvider.OpenAI, 'gpt-4o-mini', { contextLimit: 128_000, capabilities: openaiCaps });
    this.register(LLMProvider.OpenAI, 'gpt-4-turbo', { contextLimit: 128_000, capabilities: openaiCaps });

    const anthropicCaps: ProviderCapabilities = {
      ...DEFAULT_CAPABILITIES,
      presencePenalty: false,
      frequencyPenalty: false,
      vision: true,
    };
    this.register(LLMProvider.Anthropic, 'claude-3-5-sonnet-20241022', { contextLimit: 200_000, capabilities: anthropicCaps });
    this.register(LLMProvider.Anthropic, 'claude-3-5-sonnet', { contextLimit: 200_000, capabilities: anthropicCaps });
    this.register(LLMProvider.Anthropic, 'claude-3-opus-20240229', { contextLimit: 200_000, capabilities: anthropicCaps });
    this.register(LLMProvider.Anthropic, 'claude-3-haiku-20240307', { contextLimit: 200_000, capabilities: anthropicCaps });

    const geminiCaps: ProviderCapabilities = {
      ...DEFAULT_CAPABILITIES,
      presencePenalty: true,
      frequencyPenalty: true,
      vision: true,
    };
    this.register(LLMProvider.Gemini, 'gemini-1.5-pro', { contextLimit: 2_000_000, capabilities: geminiCaps });
    this.register(LLMProvider.Gemini, 'gemini-1.5-flash', { contextLimit: 1_000_000, capabilities: geminiCaps });
    this.register(LLMProvider.Gemini, 'gemini-2.0-flash', { contextLimit: 1_000_000, capabilities: geminiCaps });
  }
}
