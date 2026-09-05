import { describe, expect, it } from 'vitest';
import { loadLLMConfig } from '../../src/infrastructure/llm/configuration/llm.config.js';
import { ConfigurationError } from '../../src/domain/errors/configuration-error.js';
import { LLMProvider } from '../../src/domain/enums/provider.enum.js';

const BASE_ENV = {
  LLM_PROVIDER: 'openai',
  ACTIVE_MODEL: 'gpt-4o-mini',
  OPENAI_API_KEY: 'sk-test',
};

describe('loadLLMConfig', () => {
  it('loads a minimal valid configuration with sensible defaults', () => {
    const config = loadLLMConfig(BASE_ENV);
    expect(config.provider).toBe(LLMProvider.OpenAI);
    expect(config.openai?.apiKey).toBe('sk-test');
    expect(config.openai?.baseUrl).toBe('https://api.openai.com/v1');
    expect(config.routing.primaryProvider).toBe(LLMProvider.OpenAI);
    expect(config.routing.primaryModel).toBe('gpt-4o-mini');
    expect(config.routing.fallbackProvider).toBeUndefined();
  });

  it('fails fast when LLM_PROVIDER is missing', () => {
    expect(() => loadLLMConfig({ ACTIVE_MODEL: 'gpt-4o-mini' })).toThrow(ConfigurationError);
  });

  it('fails fast on an unknown provider', () => {
    expect(() => loadLLMConfig({ ...BASE_ENV, LLM_PROVIDER: 'made-up-provider' })).toThrow(ConfigurationError);
  });

  it('fails fast when the provider-specific API key is missing', () => {
    expect(() => loadLLMConfig({ LLM_PROVIDER: 'openai', ACTIVE_MODEL: 'gpt-4o-mini' })).toThrow(ConfigurationError);
  });

  it('requires FALLBACK_PROVIDER and FALLBACK_MODEL to be set together', () => {
    expect(() => loadLLMConfig({ ...BASE_ENV, FALLBACK_PROVIDER: 'anthropic' })).toThrow(ConfigurationError);
    expect(() => loadLLMConfig({ ...BASE_ENV, FALLBACK_MODEL: 'claude-3-5-sonnet-20241022' })).toThrow(ConfigurationError);
  });

  it('builds provider config only for providers actually referenced by routing', () => {
    const config = loadLLMConfig({
      ...BASE_ENV,
      FALLBACK_PROVIDER: 'anthropic',
      FALLBACK_MODEL: 'claude-3-5-sonnet-20241022',
      ANTHROPIC_API_KEY: 'anthropic-key',
    });
    expect(config.anthropic?.apiKey).toBe('anthropic-key');
    expect(config.gemini).toBeUndefined();
    expect(config.local).toBeUndefined();
  });

  it('requires the fallback provider API key when a fallback is configured', () => {
    expect(() =>
      loadLLMConfig({ ...BASE_ENV, FALLBACK_PROVIDER: 'anthropic', FALLBACK_MODEL: 'claude-3-5-sonnet-20241022' }),
    ).toThrow(ConfigurationError);
  });

  it('loads local provider configuration from LOCAL_LLM_* variables', () => {
    const config = loadLLMConfig({
      LLM_PROVIDER: 'local',
      ACTIVE_MODEL: 'llama3',
      LOCAL_LLM_BASE_URL: 'http://localhost:11434/v1',
      LOCAL_LLM_MODEL: 'llama3',
    });
    expect(config.local?.baseUrl).toBe('http://localhost:11434/v1');
    expect(config.local?.apiKey).toBeUndefined();
  });

  it('applies retry and circuit breaker defaults when not overridden', () => {
    const config = loadLLMConfig(BASE_ENV);
    expect(config.retry.maxRetries).toBe(2);
    expect(config.circuitBreaker.failureThreshold).toBe(5);
  });

  it('honors overridden retry and circuit breaker values', () => {
    const config = loadLLMConfig({ ...BASE_ENV, LLM_RETRY_MAX_ATTEMPTS: '5', LLM_CIRCUIT_BREAKER_FAILURE_THRESHOLD: '10' });
    expect(config.retry.maxRetries).toBe(5);
    expect(config.circuitBreaker.failureThreshold).toBe(10);
  });
});
