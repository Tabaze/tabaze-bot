import { LLMProvider } from '../../../domain/enums/provider.enum.js';
import { ConfigurationError } from '../../../domain/errors/configuration-error.js';
import { BaseLLMAdapter } from '../adapters/base-llm.adapter.js';
import { OpenAIAdapter } from '../adapters/openai.adapter.js';
import { AnthropicAdapter } from '../adapters/anthropic.adapter.js';
import { GeminiAdapter } from '../adapters/gemini.adapter.js';
import { OpenAICompatibleAdapter } from '../adapters/openai-compatible.adapter.js';
import type { LLMConfig } from '../configuration/llm.config.js';
import { ModelRegistry } from '../configuration/model-registry.js';
import { TokenizerService } from '../tokenization/tokenizer.service.js';
import type { ILogger } from '../../observability/logger.js';

export interface LLMFactoryDependencies {
  readonly modelRegistry: ModelRegistry;
  readonly tokenizer: TokenizerService;
  readonly logger: ILogger;
}

/**
 * Builds the correct adapter for a provider from configuration. This is
 * the only place in the codebase that maps LLMProvider -> concrete adapter
 * class; nothing else branches on provider identity.
 */
export class LLMFactory {
  constructor(
    private readonly config: LLMConfig,
    private readonly deps: LLMFactoryDependencies,
  ) {}

  create(provider: LLMProvider): BaseLLMAdapter {
    switch (provider) {
      case LLMProvider.OpenAI: {
        if (!this.config.openai) {
          throw new ConfigurationError('LLMFactory: OpenAI configuration is missing.');
        }
        return new OpenAIAdapter(this.config.openai, this.deps.modelRegistry, this.deps.tokenizer, this.deps.logger);
      }
      case LLMProvider.Anthropic: {
        if (!this.config.anthropic) {
          throw new ConfigurationError('LLMFactory: Anthropic configuration is missing.');
        }
        return new AnthropicAdapter(this.config.anthropic, this.deps.modelRegistry, this.deps.tokenizer, this.deps.logger);
      }
      case LLMProvider.Gemini: {
        if (!this.config.gemini) {
          throw new ConfigurationError('LLMFactory: Gemini configuration is missing.');
        }
        return new GeminiAdapter(this.config.gemini, this.deps.modelRegistry, this.deps.tokenizer, this.deps.logger);
      }
      case LLMProvider.Local: {
        if (!this.config.local) {
          throw new ConfigurationError('LLMFactory: Local provider configuration is missing.');
        }
        return new OpenAICompatibleAdapter(this.config.local, this.deps.modelRegistry, this.deps.tokenizer, this.deps.logger);
      }
      default: {
        const exhaustiveCheck: never = provider;
        throw new ConfigurationError(`LLMFactory: unknown provider "${String(exhaustiveCheck)}".`);
      }
    }
  }
}
