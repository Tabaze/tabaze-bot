import { afterEach, describe, expect, it, vi } from 'vitest';
import { OpenAICompatibleAdapter } from '../../src/infrastructure/llm/adapters/openai-compatible.adapter.js';
import { ModelRegistry } from '../../src/infrastructure/llm/configuration/model-registry.js';
import { TokenizerService } from '../../src/infrastructure/llm/tokenization/tokenizer.service.js';
import { NullLogger } from '../../src/infrastructure/observability/logger.js';
import { createGenerationConfig } from '../../src/domain/models/generation-config.js';
import { userMessage } from '../../src/domain/models/message.js';
import { jsonResponse, queueFetchResponses, requestBody } from '../helpers/mock-fetch.js';

afterEach(() => {
  vi.unstubAllGlobals();
});

function buildAdapter(overrides: Partial<{ baseUrl: string; apiKey: string; model: string }> = {}): OpenAICompatibleAdapter {
  return new OpenAICompatibleAdapter(
    { baseUrl: 'http://localhost:11434/v1', model: 'llama3', ...overrides },
    new ModelRegistry(),
    new TokenizerService(),
    new NullLogger(),
  );
}

describe('OpenAICompatibleAdapter', () => {
  it('targets the configured baseUrl regardless of which local server it is', async () => {
    const { calls } = queueFetchResponses([
      jsonResponse(200, { choices: [{ message: { content: 'ok' }, finish_reason: 'stop' }] }),
    ]);
    await buildAdapter({ baseUrl: 'http://localhost:8000/v1' }).generate([userMessage('hi')], 'llama3', createGenerationConfig());
    expect(calls[0]!.url).toBe('http://localhost:8000/v1/chat/completions');
  });

  it('omits the Authorization header when no API key is configured', async () => {
    const { calls } = queueFetchResponses([
      jsonResponse(200, { choices: [{ message: { content: 'ok' }, finish_reason: 'stop' }] }),
    ]);
    await buildAdapter().generate([userMessage('hi')], 'llama3', createGenerationConfig());
    const headers = calls[0]!.init.headers as Record<string, string>;
    expect(headers.Authorization).toBeUndefined();
  });

  it('includes a Bearer Authorization header when an API key is configured', async () => {
    const { calls } = queueFetchResponses([
      jsonResponse(200, { choices: [{ message: { content: 'ok' }, finish_reason: 'stop' }] }),
    ]);
    await buildAdapter({ apiKey: 'local-secret' }).generate([userMessage('hi')], 'llama3', createGenerationConfig());
    const headers = calls[0]!.init.headers as Record<string, string>;
    expect(headers.Authorization).toBe('Bearer local-secret');
  });

  it('falls back to the configured default model when an empty model string is passed', async () => {
    const { calls } = queueFetchResponses([
      jsonResponse(200, { choices: [{ message: { content: 'ok' }, finish_reason: 'stop' }] }),
    ]);
    await buildAdapter({ model: 'mistral' }).generate([userMessage('hi')], '', createGenerationConfig());
    expect(requestBody(calls[0]!).model).toBe('mistral');
  });

  it('does not send stream_options, since it is not reliably supported across local servers', async () => {
    const { calls } = queueFetchResponses([
      jsonResponse(200, { choices: [{ message: { content: 'ok' }, finish_reason: 'stop' }] }),
    ]);
    await buildAdapter().generate([userMessage('hi')], 'llama3', createGenerationConfig());
    expect('stream_options' in requestBody(calls[0]!)).toBe(false);
  });

  it('reports token counts as estimates, never as exact', async () => {
    await expect(buildAdapter().tokenizeAndCount([userMessage('hello world')])).resolves.toBeGreaterThan(0);
  });
});
