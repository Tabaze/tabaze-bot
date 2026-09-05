import { describe, expect, it } from 'vitest';
import { LLMFactory } from '../../src/infrastructure/llm/factory/llm.factory.js';
import { OpenAIAdapter } from '../../src/infrastructure/llm/adapters/openai.adapter.js';
import { AnthropicAdapter } from '../../src/infrastructure/llm/adapters/anthropic.adapter.js';
import { GeminiAdapter } from '../../src/infrastructure/llm/adapters/gemini.adapter.js';
import { OpenAICompatibleAdapter } from '../../src/infrastructure/llm/adapters/openai-compatible.adapter.js';
import { ModelRegistry } from '../../src/infrastructure/llm/configuration/model-registry.js';
import { TokenizerService } from '../../src/infrastructure/llm/tokenization/tokenizer.service.js';
import { NullLogger } from '../../src/infrastructure/observability/logger.js';
import { LLMProvider } from '../../src/domain/enums/provider.enum.js';
import { ConfigurationError } from '../../src/domain/errors/configuration-error.js';
import type { LLMConfig } from '../../src/infrastructure/llm/configuration/llm.config.js';

function buildFactory(config: LLMConfig): LLMFactory {
  return new LLMFactory(config, { modelRegistry: new ModelRegistry(), tokenizer: new TokenizerService(), logger: new NullLogger() });
}

const BASE_CONFIG: LLMConfig = {
  provider: LLMProvider.OpenAI,
  activeModel: 'gpt-4o-mini',
  openai: { apiKey: 'k', baseUrl: 'https://api.openai.com/v1' },
  anthropic: { apiKey: 'k', baseUrl: 'https://api.anthropic.com/v1' },
  gemini: { apiKey: 'k', baseUrl: 'https://generativelanguage.googleapis.com/v1beta' },
  local: { baseUrl: 'http://localhost:11434/v1', model: 'llama3' },
  routing: { primaryProvider: LLMProvider.OpenAI, primaryModel: 'gpt-4o-mini' },
  retry: { maxRetries: 2, baseDelayMs: 250, maxDelayMs: 8000 },
  circuitBreaker: { failureThreshold: 5, cooldownMs: 30_000, halfOpenMaxAttempts: 1 },
  defaultTimeoutMs: 60_000,
};

describe('LLMFactory', () => {
  it('creates an OpenAIAdapter for the openai provider', () => {
    expect(buildFactory(BASE_CONFIG).create(LLMProvider.OpenAI)).toBeInstanceOf(OpenAIAdapter);
  });

  it('creates an AnthropicAdapter for the anthropic provider', () => {
    expect(buildFactory(BASE_CONFIG).create(LLMProvider.Anthropic)).toBeInstanceOf(AnthropicAdapter);
  });

  it('creates a GeminiAdapter for the gemini provider', () => {
    expect(buildFactory(BASE_CONFIG).create(LLMProvider.Gemini)).toBeInstanceOf(GeminiAdapter);
  });

  it('creates an OpenAICompatibleAdapter for the local provider', () => {
    expect(buildFactory(BASE_CONFIG).create(LLMProvider.Local)).toBeInstanceOf(OpenAICompatibleAdapter);
  });

  it('throws ConfigurationError when the requested provider has no configuration', () => {
    const factory = buildFactory({ ...BASE_CONFIG, openai: undefined });
    expect(() => factory.create(LLMProvider.OpenAI)).toThrow(ConfigurationError);
  });

  it('throws ConfigurationError for an unrecognized provider value', () => {
    const factory = buildFactory(BASE_CONFIG);
    expect(() => factory.create('made-up' as LLMProvider)).toThrow(ConfigurationError);
  });
});
