import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { AddressInfo } from 'node:net';
import { startHttpServer } from '../../src/api/http-server.js';
import { NullLogger } from '../../src/infrastructure/observability/logger.js';
import { AuthenticationError, RateLimitError } from '../../src/domain/errors/llm-error.js';
import { FinishReason } from '../../src/domain/enums/finish-reason.enum.js';
import { LLMProvider } from '../../src/domain/enums/provider.enum.js';
import type { ILLMService } from '../../src/application/interfaces/illm-service.interface.js';
import type { ModelResponse } from '../../src/domain/models/model-response.js';
import type { StreamChunk } from '../../src/domain/models/stream-chunk.js';

function fakeLLMService(overrides: Partial<ILLMService> = {}): ILLMService {
  const base: ILLMService = {
    generate: vi.fn().mockResolvedValue({
      id: 'gen-1',
      provider: LLMProvider.OpenAI,
      model: 'gpt-4o-mini',
      content: 'hi there',
      usage: { promptTokens: 1, completionTokens: 1, totalTokens: 2 },
      finishReason: FinishReason.Stop,
    } satisfies ModelResponse),
    generateStream: vi.fn().mockImplementation(async function* (): AsyncIterable<StreamChunk> {
      yield { id: 'gen-1', provider: LLMProvider.OpenAI, model: 'gpt-4o-mini', delta: 'hi', finishReason: null, usage: null };
      yield { id: 'gen-1', provider: LLMProvider.OpenAI, model: 'gpt-4o-mini', delta: '', finishReason: FinishReason.Stop, usage: { promptTokens: 1, completionTokens: 1, totalTokens: 2 } };
    }),
    countTokens: vi.fn().mockResolvedValue(42),
  };
  return { ...base, ...overrides };
}

describe('startHttpServer', () => {
  let server: ReturnType<typeof startHttpServer>;
  let baseUrl: string;
  let llmService: ILLMService;

  function boot(overrides: Partial<ILLMService> = {}): void {
    llmService = fakeLLMService(overrides);
    server = startHttpServer({ llmService, logger: new NullLogger() }, 0);
  }

  async function waitForListening(): Promise<void> {
    await new Promise<void>((resolve) => server.once('listening', resolve));
    const { port } = server.address() as AddressInfo;
    baseUrl = `http://127.0.0.1:${port}`;
  }

  afterEach(async () => {
    await new Promise<void>((resolve) => server.close(() => resolve()));
  });

  it('responds to GET /health without touching the LLM service', async () => {
    boot();
    await waitForListening();

    const res = await fetch(`${baseUrl}/health`);
    expect(res.status).toBe(200);
    await expect(res.json()).resolves.toEqual({ status: 'ok' });
    expect(llmService.generate).not.toHaveBeenCalled();
  });

  it('returns 404 for an unknown route', async () => {
    boot();
    await waitForListening();

    const res = await fetch(`${baseUrl}/nope`);
    expect(res.status).toBe(404);
  });

  it('calls generate and returns the ModelResponse as JSON', async () => {
    boot();
    await waitForListening();

    const res = await fetch(`${baseUrl}/v1/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: [{ role: 'user', content: 'hi' }] }),
    });

    expect(res.status).toBe(200);
    const body = (await res.json()) as ModelResponse;
    expect(body.content).toBe('hi there');
    expect(llmService.generate).toHaveBeenCalledWith([{ role: 'user', content: 'hi' }], undefined);
  });

  it('rejects a request missing "messages" with 400 and no LLM call', async () => {
    boot();
    await waitForListening();

    const res = await fetch(`${baseUrl}/v1/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    });

    expect(res.status).toBe(400);
    expect(llmService.generate).not.toHaveBeenCalled();
  });

  it('rejects a message with an invalid role with 400', async () => {
    boot();
    await waitForListening();

    const res = await fetch(`${baseUrl}/v1/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: [{ role: 'bogus', content: 'hi' }] }),
    });

    expect(res.status).toBe(400);
  });

  it('maps a thrown LLMError to its corresponding HTTP status', async () => {
    boot({ generate: vi.fn().mockRejectedValue(new RateLimitError('slow down')) });
    await waitForListening();

    const res = await fetch(`${baseUrl}/v1/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: [{ role: 'user', content: 'hi' }] }),
    });

    expect(res.status).toBe(429);
    const body = (await res.json()) as { error: { category: string } };
    expect(body.error.category).toBe('rate_limit');
  });

  it('maps AuthenticationError to 401', async () => {
    boot({ generate: vi.fn().mockRejectedValue(new AuthenticationError('bad key')) });
    await waitForListening();

    const res = await fetch(`${baseUrl}/v1/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: [{ role: 'user', content: 'hi' }] }),
    });

    expect(res.status).toBe(401);
  });

  it('streams generateStream chunks as SSE events', async () => {
    boot();
    await waitForListening();

    const res = await fetch(`${baseUrl}/v1/generate/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: [{ role: 'user', content: 'hi' }] }),
    });

    expect(res.status).toBe(200);
    expect(res.headers.get('content-type')).toBe('text/event-stream');
    const text = await res.text();
    expect(text).toContain('"delta":"hi"');
    expect(text.match(/^data: /gm)).toHaveLength(2);
  });

  it('returns the prompt token count from /v1/tokens', async () => {
    boot();
    await waitForListening();

    const res = await fetch(`${baseUrl}/v1/tokens`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: [{ role: 'user', content: 'hi' }] }),
    });

    expect(res.status).toBe(200);
    await expect(res.json()).resolves.toEqual({ promptTokens: 42 });
  });
});
