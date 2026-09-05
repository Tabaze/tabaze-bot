import { describe, expect, it, vi } from 'vitest';
import { LLMService } from '../../src/application/services/llm.service.js';
import { NullLogger } from '../../src/infrastructure/observability/logger.js';
import { ContextLengthError } from '../../src/domain/errors/llm-error.js';
import { userMessage } from '../../src/domain/models/message.js';
import { FinishReason } from '../../src/domain/enums/finish-reason.enum.js';
import { LLMProvider } from '../../src/domain/enums/provider.enum.js';
import type { LLMRouter } from '../../src/infrastructure/llm/routing/llm-router.js';
import type { ModelResponse } from '../../src/domain/models/model-response.js';
import type { StreamChunk } from '../../src/domain/models/stream-chunk.js';

function fakeRouter(overrides: Partial<LLMRouter> = {}): LLMRouter {
  const base: Partial<LLMRouter> = {
    tokenizeAndCount: vi.fn().mockResolvedValue(100),
    getContextLimit: vi.fn().mockReturnValue(8_192),
    generate: vi.fn().mockResolvedValue({
      id: 'gen-1',
      provider: LLMProvider.OpenAI,
      model: 'gpt-4o-mini',
      content: 'hi',
      usage: { promptTokens: 1, completionTokens: 1, totalTokens: 2 },
      finishReason: FinishReason.Stop,
    } satisfies ModelResponse),
    generateStream: vi.fn().mockImplementation(async function* (): AsyncIterable<StreamChunk> {
      yield { id: 'gen-1', provider: LLMProvider.OpenAI, model: 'gpt-4o-mini', delta: 'hi', finishReason: FinishReason.Stop, usage: null };
    }),
  };
  return { ...base, ...overrides } as LLMRouter;
}

describe('LLMService', () => {
  it('delegates a well-formed request to the router', async () => {
    const router = fakeRouter();
    const service = new LLMService(router, new NullLogger());
    const result = await service.generate([userMessage('hi')]);
    expect(result.content).toBe('hi');
    expect(router.generate).toHaveBeenCalledTimes(1);
  });

  it('rejects a request whose prompt plus max_tokens would exceed the context limit', async () => {
    const router = fakeRouter({
      tokenizeAndCount: vi.fn().mockResolvedValue(8_000),
      getContextLimit: vi.fn().mockReturnValue(8_192),
    });
    const service = new LLMService(router, new NullLogger());

    await expect(service.generate([userMessage('hi')], { maxTokens: 500 })).rejects.toThrow(ContextLengthError);
    expect(router.generate).not.toHaveBeenCalled();
  });

  it('allows a request that fits comfortably within the context limit', async () => {
    const router = fakeRouter({
      tokenizeAndCount: vi.fn().mockResolvedValue(100),
      getContextLimit: vi.fn().mockReturnValue(8_192),
    });
    const service = new LLMService(router, new NullLogger());
    await expect(service.generate([userMessage('hi')], { maxTokens: 500 })).resolves.toBeDefined();
  });

  it('validates context length before streaming, not after', async () => {
    const router = fakeRouter({
      tokenizeAndCount: vi.fn().mockResolvedValue(8_000),
      getContextLimit: vi.fn().mockReturnValue(8_192),
    });
    const service = new LLMService(router, new NullLogger());

    const iterate = async () => {
      for await (const _chunk of service.generateStream([userMessage('hi')], { maxTokens: 500 })) {
        // no-op
      }
    };

    await expect(iterate()).rejects.toThrow(ContextLengthError);
    expect(router.generateStream).not.toHaveBeenCalled();
  });

  it('delegates countTokens to the router', async () => {
    const router = fakeRouter({ tokenizeAndCount: vi.fn().mockResolvedValue(7) });
    const service = new LLMService(router, new NullLogger());
    await expect(service.countTokens([userMessage('hi')])).resolves.toBe(7);
  });
});
