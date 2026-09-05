import { afterEach, describe, expect, it, vi } from 'vitest';
import { executeHttpRequest } from '../../src/infrastructure/llm/http/http-client.js';
import { LLMProvider } from '../../src/domain/enums/provider.enum.js';
import { CancelledError, NetworkError, TimeoutError } from '../../src/domain/errors/llm-error.js';

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('executeHttpRequest', () => {
  it('normalizes a low-level fetch failure (DNS/connection refused) into NetworkError', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockRejectedValue(new TypeError('fetch failed: ECONNREFUSED')),
    );

    await expect(
      executeHttpRequest(
        { url: 'http://localhost:11434/v1/chat/completions', method: 'POST', headers: {}, timeoutMs: 1_000 },
        LLMProvider.Local,
      ),
    ).rejects.toBeInstanceOf(NetworkError);
  });

  it('normalizes an internal timeout into TimeoutError', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation(
        (_url: unknown, init?: RequestInit) =>
          new Promise((_resolve, reject) => {
            const signal = init?.signal as AbortSignal | undefined;
            signal?.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')));
          }),
      ),
    );

    await expect(
      executeHttpRequest(
        { url: 'https://api.openai.com/v1/chat/completions', method: 'POST', headers: {}, timeoutMs: 10 },
        LLMProvider.OpenAI,
      ),
    ).rejects.toBeInstanceOf(TimeoutError);
  });

  it('normalizes an externally-cancelled request into CancelledError, not TimeoutError', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation(
        (_url: unknown, init?: RequestInit) =>
          new Promise((_resolve, reject) => {
            const signal = init?.signal as AbortSignal | undefined;
            signal?.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')));
          }),
      ),
    );

    const controller = new AbortController();
    const promise = executeHttpRequest(
      { url: 'https://api.openai.com/v1/chat/completions', method: 'POST', headers: {}, timeoutMs: 5_000, signal: controller.signal },
      LLMProvider.OpenAI,
    );
    controller.abort();

    await expect(promise).rejects.toBeInstanceOf(CancelledError);
  });

  it('returns the raw Response on success without transformation', async () => {
    const okResponse = new Response('{}', { status: 200 });
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(okResponse));

    const response = await executeHttpRequest(
      { url: 'https://api.openai.com/v1/chat/completions', method: 'POST', headers: {}, timeoutMs: 1_000 },
      LLMProvider.OpenAI,
    );
    expect(response).toBe(okResponse);
  });
});
