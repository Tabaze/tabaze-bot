import { describe, expect, it } from 'vitest';
import { FallbackManager, type RouteTarget } from '../../src/infrastructure/llm/routing/fallback-manager.js';
import { ProviderHealthRegistry } from '../../src/infrastructure/llm/routing/provider-health.js';
import { RetryPolicy } from '../../src/infrastructure/llm/routing/retry-policy.js';
import { NullLogger } from '../../src/infrastructure/observability/logger.js';
import { InMemoryMetricsRecorder, METRIC_NAMES } from '../../src/infrastructure/observability/metrics.js';
import { BaseLLMAdapter, type GenerateOptions } from '../../src/infrastructure/llm/adapters/base-llm.adapter.js';
import { LLMProvider } from '../../src/domain/enums/provider.enum.js';
import { FinishReason } from '../../src/domain/enums/finish-reason.enum.js';
import { DEFAULT_CAPABILITIES } from '../../src/domain/models/provider-capabilities.js';
import { createGenerationConfig } from '../../src/domain/models/generation-config.js';
import { userMessage } from '../../src/domain/models/message.js';
import type { ModelResponse } from '../../src/domain/models/model-response.js';
import type { StreamChunk } from '../../src/domain/models/stream-chunk.js';
import { AuthenticationError, CancelledError, InvalidRequestError, RateLimitError } from '../../src/domain/errors/llm-error.js';
import { StreamInterruptedError } from '../../src/domain/errors/stream-interrupted-error.js';

type GenerateImpl = (options?: GenerateOptions) => Promise<ModelResponse>;
type StreamImpl = (options?: GenerateOptions) => AsyncIterable<StreamChunk>;

class FakeAdapter extends BaseLLMAdapter {
  readonly provider: LLMProvider;
  generateCallCount = 0;
  streamCallCount = 0;

  constructor(
    provider: LLMProvider,
    private readonly generateImpl?: GenerateImpl,
    private readonly streamImpl?: StreamImpl,
  ) {
    super();
    this.provider = provider;
  }

  async generate(_messages: unknown, _model: string, _config: unknown, options?: GenerateOptions): Promise<ModelResponse> {
    this.generateCallCount += 1;
    if (!this.generateImpl) throw new Error('generate not configured on FakeAdapter');
    return this.generateImpl(options);
  }

  generateStream(_messages: unknown, _model: string, _config: unknown, options?: GenerateOptions): AsyncIterable<StreamChunk> {
    this.streamCallCount += 1;
    if (!this.streamImpl) throw new Error('generateStream not configured on FakeAdapter');
    return this.streamImpl(options);
  }

  async tokenizeAndCount(): Promise<number> {
    return 10;
  }

  getCapabilities() {
    return DEFAULT_CAPABILITIES;
  }

  getContextLimit(): number {
    return 8_192;
  }
}

function okResponse(provider: LLMProvider): ModelResponse {
  return {
    id: 'gen-1',
    provider,
    model: 'test-model',
    content: 'hello',
    usage: { promptTokens: 1, completionTokens: 1, totalTokens: 2 },
    finishReason: FinishReason.Stop,
  };
}

async function* singleChunkStream(provider: LLMProvider): AsyncIterable<StreamChunk> {
  yield { id: 'gen-1', provider, model: 'test-model', delta: 'hi', finishReason: null, usage: null };
  yield { id: 'gen-1', provider, model: 'test-model', delta: '', finishReason: FinishReason.Stop, usage: null };
}

function buildManager(retry = { maxRetries: 1, baseDelayMs: 1, maxDelayMs: 1 }) {
  const health = new ProviderHealthRegistry({ failureThreshold: 5, cooldownMs: 30_000, halfOpenMaxAttempts: 1 });
  const metrics = new InMemoryMetricsRecorder();
  const manager = new FallbackManager(health, new RetryPolicy(retry), new NullLogger(), metrics);
  return { manager, metrics };
}

const CONFIG = createGenerationConfig();
const MESSAGES = [userMessage('hi')];

describe('FallbackManager.generate', () => {
  it('returns the primary result without touching the fallback on success', async () => {
    const { manager } = buildManager();
    const primaryAdapter = new FakeAdapter(LLMProvider.OpenAI, async () => okResponse(LLMProvider.OpenAI));
    const fallbackAdapter = new FakeAdapter(LLMProvider.Anthropic, async () => okResponse(LLMProvider.Anthropic));

    const result = await manager.generate(
      { adapter: primaryAdapter, model: 'gpt-4o-mini' },
      { adapter: fallbackAdapter, model: 'claude' },
      MESSAGES,
      CONFIG,
    );

    expect(result.provider).toBe(LLMProvider.OpenAI);
    expect(fallbackAdapter.generateCallCount).toBe(0);
  });

  it('falls back after the primary exhausts retries on a retryable, fallbackable error', async () => {
    const { manager, metrics } = buildManager();
    const primaryAdapter = new FakeAdapter(LLMProvider.OpenAI, async () => {
      throw new RateLimitError('slow down');
    });
    const fallbackAdapter = new FakeAdapter(LLMProvider.Anthropic, async () => okResponse(LLMProvider.Anthropic));

    const result = await manager.generate(
      { adapter: primaryAdapter, model: 'gpt-4o-mini' },
      { adapter: fallbackAdapter, model: 'claude' },
      MESSAGES,
      CONFIG,
    );

    expect(result.provider).toBe(LLMProvider.Anthropic);
    expect(primaryAdapter.generateCallCount).toBeGreaterThan(1);
    expect(metrics.getCounter(METRIC_NAMES.FALLBACK_TOTAL, { from: LLMProvider.OpenAI, to: LLMProvider.Anthropic })).toBe(1);
  });

  it('never retries or falls back on an invalid request', async () => {
    const { manager } = buildManager();
    const error = new InvalidRequestError('malformed payload');
    const primaryAdapter = new FakeAdapter(LLMProvider.OpenAI, async () => {
      throw error;
    });
    const fallbackAdapter = new FakeAdapter(LLMProvider.Anthropic, async () => okResponse(LLMProvider.Anthropic));

    await expect(
      manager.generate({ adapter: primaryAdapter, model: 'gpt-4o-mini' }, { adapter: fallbackAdapter, model: 'claude' }, MESSAGES, CONFIG),
    ).rejects.toBe(error);

    expect(primaryAdapter.generateCallCount).toBe(1);
    expect(fallbackAdapter.generateCallCount).toBe(0);
  });

  it('falls back on an authentication failure without retrying the same provider', async () => {
    const { manager } = buildManager();
    const primaryAdapter = new FakeAdapter(LLMProvider.OpenAI, async () => {
      throw new AuthenticationError('bad key');
    });
    const fallbackAdapter = new FakeAdapter(LLMProvider.Anthropic, async () => okResponse(LLMProvider.Anthropic));

    const result = await manager.generate(
      { adapter: primaryAdapter, model: 'gpt-4o-mini' },
      { adapter: fallbackAdapter, model: 'claude' },
      MESSAGES,
      CONFIG,
    );

    expect(result.provider).toBe(LLMProvider.Anthropic);
    expect(primaryAdapter.generateCallCount).toBe(1);
  });

  it('propagates the error when there is no fallback target configured', async () => {
    const { manager } = buildManager();
    const error = new RateLimitError('slow down');
    const primaryAdapter = new FakeAdapter(LLMProvider.OpenAI, async () => {
      throw error;
    });

    await expect(
      manager.generate({ adapter: primaryAdapter, model: 'gpt-4o-mini' }, undefined, MESSAGES, CONFIG),
    ).rejects.toBe(error);
  });
});

describe('FallbackManager.generateStream', () => {
  async function drain(iterable: AsyncIterable<StreamChunk>): Promise<StreamChunk[]> {
    const chunks: StreamChunk[] = [];
    for await (const chunk of iterable) chunks.push(chunk);
    return chunks;
  }

  it('streams the primary provider straight through on success', async () => {
    const { manager } = buildManager();
    const primaryAdapter = new FakeAdapter(LLMProvider.OpenAI, undefined, () => singleChunkStream(LLMProvider.OpenAI));
    const fallbackAdapter = new FakeAdapter(LLMProvider.Anthropic, undefined, () => singleChunkStream(LLMProvider.Anthropic));

    const chunks = await drain(
      manager.generateStream({ adapter: primaryAdapter, model: 'gpt-4o-mini' }, { adapter: fallbackAdapter, model: 'claude' }, MESSAGES, CONFIG),
    );

    expect(chunks.every((c) => c.provider === LLMProvider.OpenAI)).toBe(true);
    expect(fallbackAdapter.streamCallCount).toBe(0);
  });

  it('falls back when the primary fails before emitting any chunk', async () => {
    const { manager, metrics } = buildManager();
    async function* failsImmediately(): AsyncIterable<StreamChunk> {
      throw new RateLimitError('slow down');
      // eslint-disable-next-line no-unreachable
      yield undefined as never;
    }
    const primaryAdapter = new FakeAdapter(LLMProvider.OpenAI, undefined, failsImmediately);
    const fallbackAdapter = new FakeAdapter(LLMProvider.Anthropic, undefined, () => singleChunkStream(LLMProvider.Anthropic));

    const chunks = await drain(
      manager.generateStream({ adapter: primaryAdapter, model: 'gpt-4o-mini' }, { adapter: fallbackAdapter, model: 'claude' }, MESSAGES, CONFIG),
    );

    expect(chunks.every((c) => c.provider === LLMProvider.Anthropic)).toBe(true);
    expect(metrics.getCounter(METRIC_NAMES.FALLBACK_TOTAL, { from: LLMProvider.OpenAI, to: LLMProvider.Anthropic })).toBe(1);
  });

  it('terminates with StreamInterruptedError -- and does NOT fall back -- once content has already been emitted', async () => {
    const { manager } = buildManager();
    async function* emitsThenFails(): AsyncIterable<StreamChunk> {
      yield { id: 'gen-1', provider: LLMProvider.OpenAI, model: 'test-model', delta: 'Hello, I can help', finishReason: null, usage: null };
      throw new RateLimitError('connection dropped mid-stream');
    }
    const primaryAdapter = new FakeAdapter(LLMProvider.OpenAI, undefined, emitsThenFails);
    const fallbackAdapter = new FakeAdapter(LLMProvider.Anthropic, undefined, () => singleChunkStream(LLMProvider.Anthropic));

    const iterable = manager.generateStream(
      { adapter: primaryAdapter, model: 'gpt-4o-mini' },
      { adapter: fallbackAdapter, model: 'claude' },
      MESSAGES,
      CONFIG,
    );

    await expect(drain(iterable)).rejects.toThrow(StreamInterruptedError);
    expect(fallbackAdapter.streamCallCount).toBe(0);
  });

  it('exposes the already-emitted content on StreamInterruptedError', async () => {
    const { manager } = buildManager();
    async function* emitsThenFails(): AsyncIterable<StreamChunk> {
      yield { id: 'gen-1', provider: LLMProvider.OpenAI, model: 'test-model', delta: 'partial answer', finishReason: null, usage: null };
      throw new RateLimitError('dropped');
    }
    const primaryAdapter = new FakeAdapter(LLMProvider.OpenAI, undefined, emitsThenFails);

    try {
      await drain(manager.generateStream({ adapter: primaryAdapter, model: 'gpt-4o-mini' }, undefined, MESSAGES, CONFIG));
      expect.unreachable('expected StreamInterruptedError to be thrown');
    } catch (error) {
      expect(error).toBeInstanceOf(StreamInterruptedError);
      expect((error as StreamInterruptedError).emittedContent).toBe('partial answer');
    }
  });

  it('propagates a cancellation without retrying or falling back', async () => {
    const { manager } = buildManager();
    const controller = new AbortController();
    async function* neverEmits(): AsyncIterable<StreamChunk> {
      controller.abort();
      throw new CancelledError('client disconnected');
    }
    const primaryAdapter = new FakeAdapter(LLMProvider.OpenAI, undefined, neverEmits);
    const fallbackAdapter = new FakeAdapter(LLMProvider.Anthropic, undefined, () => singleChunkStream(LLMProvider.Anthropic));

    await expect(
      drain(
        manager.generateStream(
          { adapter: primaryAdapter, model: 'gpt-4o-mini' },
          { adapter: fallbackAdapter, model: 'claude' },
          MESSAGES,
          CONFIG,
          { signal: controller.signal },
        ),
      ),
    ).rejects.toBeInstanceOf(CancelledError);
    expect(fallbackAdapter.streamCallCount).toBe(0);
    expect(primaryAdapter.streamCallCount).toBe(1);
  });
});
