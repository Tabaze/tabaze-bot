import { afterEach, describe, expect, it, vi } from 'vitest';
import { GeminiAdapter } from '../../src/infrastructure/llm/adapters/gemini.adapter.js';
import { ModelRegistry } from '../../src/infrastructure/llm/configuration/model-registry.js';
import { TokenizerService } from '../../src/infrastructure/llm/tokenization/tokenizer.service.js';
import { NullLogger } from '../../src/infrastructure/observability/logger.js';
import { createGenerationConfig } from '../../src/domain/models/generation-config.js';
import { assistantMessage, systemMessage, userMessage } from '../../src/domain/models/message.js';
import { FinishReason } from '../../src/domain/enums/finish-reason.enum.js';
import { RateLimitError } from '../../src/domain/errors/llm-error.js';
import { jsonResponse, queueFetchResponses, requestBody, sseDataEvent, sseResponse } from '../helpers/mock-fetch.js';

const CONFIG = { apiKey: 'gemini-test-key', baseUrl: 'https://generativelanguage.googleapis.com/v1beta' };

function buildAdapter(): GeminiAdapter {
  return new GeminiAdapter(CONFIG, new ModelRegistry(), new TokenizerService(), new NullLogger());
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('GeminiAdapter role mapping', () => {
  it('maps user -> user and assistant -> model, and extracts system messages into systemInstruction', async () => {
    const { calls } = queueFetchResponses([
      jsonResponse(200, {
        candidates: [{ content: { role: 'model', parts: [{ text: 'hi' }] }, finishReason: 'STOP' }],
        usageMetadata: { promptTokenCount: 5, candidatesTokenCount: 2, totalTokenCount: 7 },
      }),
    ]);

    await buildAdapter().generate(
      [systemMessage('Be terse.'), userMessage('hello'), assistantMessage('hi there')],
      'gemini-1.5-flash',
      createGenerationConfig(),
    );

    const body = requestBody(calls[0]!);
    expect(body.systemInstruction).toEqual({ role: 'user', parts: [{ text: 'Be terse.' }] });
    expect(body.contents).toEqual([
      { role: 'user', parts: [{ text: 'hello' }] },
      { role: 'model', parts: [{ text: 'hi there' }] },
    ]);
  });

  it('always includes safety settings', async () => {
    const { calls } = queueFetchResponses([
      jsonResponse(200, { candidates: [{ content: { role: 'model', parts: [{ text: 'hi' }] }, finishReason: 'STOP' }] }),
    ]);
    await buildAdapter().generate([userMessage('hi')], 'gemini-1.5-flash', createGenerationConfig());
    const body = requestBody(calls[0]!);
    expect(Array.isArray(body.safetySettings)).toBe(true);
    expect((body.safetySettings as unknown[]).length).toBeGreaterThan(0);
  });

  it('sends the API key via the x-goog-api-key header', async () => {
    const { calls } = queueFetchResponses([
      jsonResponse(200, { candidates: [{ content: { role: 'model', parts: [{ text: 'hi' }] }, finishReason: 'STOP' }] }),
    ]);
    await buildAdapter().generate([userMessage('hi')], 'gemini-1.5-flash', createGenerationConfig());
    const headers = calls[0]!.init.headers as Record<string, string>;
    expect(headers['x-goog-api-key']).toBe('gemini-test-key');
  });
});

describe('GeminiAdapter.generate', () => {
  it('normalizes a response and maps SAFETY to ContentFilter', async () => {
    queueFetchResponses([
      jsonResponse(200, {
        candidates: [{ content: { role: 'model', parts: [{ text: 'blocked' }] }, finishReason: 'SAFETY' }],
        usageMetadata: { promptTokenCount: 3, candidatesTokenCount: 1, totalTokenCount: 4 },
      }),
    ]);
    const response = await buildAdapter().generate([userMessage('hi')], 'gemini-1.5-flash', createGenerationConfig());
    expect(response.finishReason).toBe(FinishReason.ContentFilter);
    expect(response.usage).toEqual({ promptTokens: 3, completionTokens: 1, totalTokens: 4 });
  });

  it('classifies a 429 as RateLimitError', async () => {
    queueFetchResponses([jsonResponse(429, { error: { code: 429, message: 'Resource exhausted', status: 'RESOURCE_EXHAUSTED' } })]);
    await expect(buildAdapter().generate([userMessage('hi')], 'gemini-1.5-flash', createGenerationConfig())).rejects.toBeInstanceOf(
      RateLimitError,
    );
  });
});

describe('GeminiAdapter.generateStream', () => {
  it('normalizes streamed candidates into StreamChunks', async () => {
    queueFetchResponses([
      sseResponse([
        sseDataEvent({ candidates: [{ content: { role: 'model', parts: [{ text: 'Hel' }] } }] }),
        sseDataEvent({ candidates: [{ content: { role: 'model', parts: [{ text: 'lo' }] }, finishReason: 'STOP' }], usageMetadata: { promptTokenCount: 2, candidatesTokenCount: 2, totalTokenCount: 4 } }),
      ]),
    ]);

    const chunks = [];
    for await (const chunk of buildAdapter().generateStream([userMessage('hi')], 'gemini-1.5-flash', createGenerationConfig({ stream: true }))) {
      chunks.push(chunk);
    }

    expect(chunks.map((c) => c.delta).join('')).toBe('Hello');
    expect(chunks.at(-1)?.finishReason).toBe(FinishReason.Stop);
  });
});
