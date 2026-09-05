import { describe, expect, it } from 'vitest';
import { LLMRouter } from '../../src/infrastructure/llm/routing/llm-router.js';
import { FallbackManager } from '../../src/infrastructure/llm/routing/fallback-manager.js';
import { ProviderHealthRegistry } from '../../src/infrastructure/llm/routing/provider-health.js';
import { RetryPolicy } from '../../src/infrastructure/llm/routing/retry-policy.js';
import { NullLogger } from '../../src/infrastructure/observability/logger.js';
import { InMemoryMetricsRecorder } from '../../src/infrastructure/observability/metrics.js';
import { BaseLLMAdapter } from '../../src/infrastructure/llm/adapters/base-llm.adapter.js';
import { LLMProvider } from '../../src/domain/enums/provider.enum.js';
import { FinishReason } from '../../src/domain/enums/finish-reason.enum.js';
import { DEFAULT_CAPABILITIES } from '../../src/domain/models/provider-capabilities.js';
import { createGenerationConfig } from '../../src/domain/models/generation-config.js';
import { userMessage } from '../../src/domain/models/message.js';
import type { ModelResponse } from '../../src/domain/models/model-response.js';
import type { StreamChunk } from '../../src/domain/models/stream-chunk.js';
import type { LLMConfig } from '../../src/infrastructure/llm/configuration/llm.config.js';
import type { LLMFactory } from '../../src/infrastructure/llm/factory/llm.factory.js';

class FakeAdapter extends BaseLLMAdapter {
  constructor(
    readonly provider: LLMProvider,
    private readonly contextLimit = 8_192,
  ) {
    super();
  }

  async generate(): Promise<ModelResponse> {
    return {
      id: 'gen-1',
      provider: this.provider,
      model: 'fake-model',
      content: 'ok',
      usage: { promptTokens: 1, completionTokens: 1, totalTokens: 2 },
      finishReason: FinishReason.Stop,
    };
  }

  async *generateStream(): AsyncIterable<StreamChunk> {
    yield { id: 'gen-1', provider: this.provider, model: 'fake-model', delta: 'ok', finishReason: FinishReason.Stop, usage: null };
  }

  async tokenizeAndCount(): Promise<number> {
    return 42;
  }

  getCapabilities() {
    return DEFAULT_CAPABILITIES;
  }

  getContextLimit(): number {
    return this.contextLimit;
  }
}

function buildRouter(config: LLMConfig, adaptersByProvider: Partial<Record<LLMProvider, BaseLLMAdapter>>): LLMRouter {
  const fakeFactory: Pick<LLMFactory, 'create'> = {
    create: (provider: LLMProvider) => {
      const adapter = adaptersByProvider[provider];
      if (!adapter) throw new Error(`no fake adapter registered for ${provider}`);
      return adapter;
    },
  };
  const fallbackManager = new FallbackManager(
    new ProviderHealthRegistry({ failureThreshold: 5, cooldownMs: 30_000, halfOpenMaxAttempts: 1 }),
    new RetryPolicy({ maxRetries: 1, baseDelayMs: 1, maxDelayMs: 1 }),
    new NullLogger(),
    new InMemoryMetricsRecorder(),
  );
  return new LLMRouter(config, fakeFactory as LLMFactory, fallbackManager);
}

const BASE_CONFIG: LLMConfig = {
  provider: LLMProvider.OpenAI,
  activeModel: 'gpt-4o-mini',
  routing: { primaryProvider: LLMProvider.OpenAI, primaryModel: 'gpt-4o-mini' },
  retry: { maxRetries: 1, baseDelayMs: 1, maxDelayMs: 1 },
  circuitBreaker: { failureThreshold: 5, cooldownMs: 30_000, halfOpenMaxAttempts: 1 },
  defaultTimeoutMs: 60_000,
};

describe('LLMRouter', () => {
  it('routes generate() to the configured primary provider', async () => {
    const router = buildRouter(BASE_CONFIG, { [LLMProvider.OpenAI]: new FakeAdapter(LLMProvider.OpenAI) });
    const result = await router.generate([userMessage('hi')], createGenerationConfig());
    expect(result.provider).toBe(LLMProvider.OpenAI);
  });

  it('exposes the primary model context limit and capabilities', () => {
    const router = buildRouter(BASE_CONFIG, { [LLMProvider.OpenAI]: new FakeAdapter(LLMProvider.OpenAI, 128_000) });
    expect(router.getContextLimit()).toBe(128_000);
    expect(router.getCapabilities()).toEqual(DEFAULT_CAPABILITIES);
    expect(router.getActiveModel()).toBe('gpt-4o-mini');
  });

  it('delegates token counting to the primary adapter', async () => {
    const router = buildRouter(BASE_CONFIG, { [LLMProvider.OpenAI]: new FakeAdapter(LLMProvider.OpenAI) });
    await expect(router.tokenizeAndCount([userMessage('hi')])).resolves.toBe(42);
  });

  it('constructs both a primary and fallback adapter when routing configures a fallback', async () => {
    const config: LLMConfig = {
      ...BASE_CONFIG,
      routing: {
        primaryProvider: LLMProvider.OpenAI,
        primaryModel: 'gpt-4o-mini',
        fallbackProvider: LLMProvider.Anthropic,
        fallbackModel: 'claude-3-5-sonnet-20241022',
      },
    };
    const router = buildRouter(config, {
      [LLMProvider.OpenAI]: new FakeAdapter(LLMProvider.OpenAI),
      [LLMProvider.Anthropic]: new FakeAdapter(LLMProvider.Anthropic),
    });
    const result = await router.generate([userMessage('hi')], createGenerationConfig());
    expect(result.provider).toBe(LLMProvider.OpenAI);
  });

  it('streams chunks from the primary provider', async () => {
    const router = buildRouter(BASE_CONFIG, { [LLMProvider.OpenAI]: new FakeAdapter(LLMProvider.OpenAI) });
    const chunks: StreamChunk[] = [];
    for await (const chunk of router.generateStream([userMessage('hi')], createGenerationConfig({ stream: true }))) {
      chunks.push(chunk);
    }
    expect(chunks).toHaveLength(1);
    expect(chunks[0]?.provider).toBe(LLMProvider.OpenAI);
  });
});
