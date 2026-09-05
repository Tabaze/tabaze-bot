import { afterEach, describe, expect, it, vi } from 'vitest';
import { OpenAIAdapter } from '../../src/infrastructure/llm/adapters/openai.adapter.js';
import { ModelRegistry } from '../../src/infrastructure/llm/configuration/model-registry.js';
import { TokenizerService } from '../../src/infrastructure/llm/tokenization/tokenizer.service.js';
import { NullLogger } from '../../src/infrastructure/observability/logger.js';
import { createGenerationConfig } from '../../src/domain/models/generation-config.js';
import { userMessage } from '../../src/domain/models/message.js';
import { CancelledError } from '../../src/domain/errors/llm-error.js';
import { queueFetchResponses, sseDataEvent } from '../helpers/mock-fetch.js';

afterEach(() => {
  vi.unstubAllGlobals();
});

/** A `text/event-stream` Response that only enqueues its next event after a delay, so a test can abort mid-stream. */
function delayedSseResponse(rawEvents: readonly string[], delayMs: number): Response {
  const encoder = new TextEncoder();
  let index = 0;
  const stream = new ReadableStream<Uint8Array>({
    async pull(controller) {
      if (index >= rawEvents.length) {
        controller.close();
        return;
      }
      await new Promise((resolve) => setTimeout(resolve, delayMs));
      controller.enqueue(encoder.encode(rawEvents[index++]));
    },
  });
  return new Response(stream, { status: 200, headers: { 'Content-Type': 'text/event-stream' } });
}

describe('cancellation propagation during an active stream', () => {
  it('stops consuming the provider stream once the caller aborts, instead of reading further tokens', async () => {
    queueFetchResponses([
      delayedSseResponse(
        [
          sseDataEvent({ id: 'c1', model: 'gpt-4o-mini', choices: [{ delta: { content: 'Hel' }, finish_reason: null }] }),
          sseDataEvent({ id: 'c1', model: 'gpt-4o-mini', choices: [{ delta: { content: 'lo, this should never arrive' }, finish_reason: null }] }),
        ],
        20,
      ),
    ]);

    const adapter = new OpenAIAdapter(
      { apiKey: 'sk-test', baseUrl: 'https://api.openai.com/v1' },
      new ModelRegistry(),
      new TokenizerService(),
      new NullLogger(),
    );

    const controller = new AbortController();
    const iterator = adapter
      .generateStream([userMessage('hi')], 'gpt-4o-mini', createGenerationConfig({ stream: true }), { signal: controller.signal })
      [Symbol.asyncIterator]();

    const first = await iterator.next();
    expect(first.done).toBe(false);
    expect(first.value?.delta).toBe('Hel');

    controller.abort();

    await expect(iterator.next()).rejects.toBeInstanceOf(CancelledError);
  });

  it('rejects immediately with CancelledError when the signal is already aborted before the request starts', async () => {
    const adapter = new OpenAIAdapter(
      { apiKey: 'sk-test', baseUrl: 'https://api.openai.com/v1' },
      new ModelRegistry(),
      new TokenizerService(),
      new NullLogger(),
    );
    const controller = new AbortController();
    controller.abort();

    await expect(
      adapter.generate([userMessage('hi')], 'gpt-4o-mini', createGenerationConfig(), { signal: controller.signal }),
    ).rejects.toBeInstanceOf(CancelledError);
  });
});
