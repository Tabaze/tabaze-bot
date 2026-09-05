import { afterEach, describe, expect, it, vi } from 'vitest';
import { OpenAIAdapter } from '../../src/infrastructure/llm/adapters/openai.adapter.js';
import { ModelRegistry } from '../../src/infrastructure/llm/configuration/model-registry.js';
import { TokenizerService } from '../../src/infrastructure/llm/tokenization/tokenizer.service.js';
import { NullLogger } from '../../src/infrastructure/observability/logger.js';
import { createGenerationConfig } from '../../src/domain/models/generation-config.js';
import { systemMessage, userMessage } from '../../src/domain/models/message.js';
import { FinishReason } from '../../src/domain/enums/finish-reason.enum.js';
import { AuthenticationError, ContextLengthError, RateLimitError, ServerError } from '../../src/domain/errors/llm-error.js';
import { jsonResponse, queueFetchResponses, requestBody, sseDataEvent, sseResponse, abortAwareFetch } from '../helpers/mock-fetch.js';

const CONFIG = { apiKey: 'sk-test', baseUrl: 'https://api.openai.com/v1' };

function buildAdapter(): OpenAIAdapter {
  return new OpenAIAdapter(CONFIG, new ModelRegistry(), new TokenizerService(), new NullLogger());
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('OpenAIAdapter.generate', () => {
  it('normalizes a successful completion into a ModelResponse', async () => {
    queueFetchResponses([
      jsonResponse(200, {
        id: 'chatcmpl-1',
        model: 'gpt-4o-mini',
        choices: [{ index: 0, message: { content: 'Hello there' }, finish_reason: 'stop' }],
        usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 },
      }),
    ]);

    const response = await buildAdapter().generate([userMessage('hi')], 'gpt-4o-mini', createGenerationConfig());

    expect(response).toEqual({
      id: 'chatcmpl-1',
      provider: 'openai',
      model: 'gpt-4o-mini',
      content: 'Hello there',
      usage: { promptTokens: 5, completionTokens: 3, totalTokens: 8 },
      finishReason: FinishReason.Stop,
    });
  });

  it('maps "length" to FinishReason.MaxTokens', async () => {
    queueFetchResponses([
      jsonResponse(200, {
        id: 'chatcmpl-2',
        model: 'gpt-4o-mini',
        choices: [{ index: 0, message: { content: 'truncated' }, finish_reason: 'length' }],
        usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 },
      }),
    ]);
    const response = await buildAdapter().generate([userMessage('hi')], 'gpt-4o-mini', createGenerationConfig());
    expect(response.finishReason).toBe(FinishReason.MaxTokens);
  });

  it('sends normalized parameters mapped to the OpenAI request shape', async () => {
    const { calls } = queueFetchResponses([
      jsonResponse(200, { id: 'x', model: 'gpt-4o-mini', choices: [{ index: 0, message: { content: 'ok' }, finish_reason: 'stop' }] }),
    ]);

    await buildAdapter().generate(
      [systemMessage('be helpful'), userMessage('hi')],
      'gpt-4o-mini',
      createGenerationConfig({ temperature: 0.5, topP: 0.9, maxTokens: 100, stopSequences: ['\n'] }),
    );

    const body = requestBody(calls[0]!);
    expect(body).toMatchObject({
      model: 'gpt-4o-mini',
      temperature: 0.5,
      top_p: 0.9,
      max_tokens: 100,
      stop: ['\n'],
      messages: [
        { role: 'system', content: 'be helpful' },
        { role: 'user', content: 'hi' },
      ],
    });
    expect(String(calls[0]!.init.headers && (calls[0]!.init.headers as Record<string, string>).Authorization)).toBe('Bearer sk-test');
  });

  it('classifies a 401 as AuthenticationError', async () => {
    queueFetchResponses([jsonResponse(401, { error: { message: 'Invalid API key' } })]);
    await expect(buildAdapter().generate([userMessage('hi')], 'gpt-4o-mini', createGenerationConfig())).rejects.toBeInstanceOf(
      AuthenticationError,
    );
  });

  it('classifies a 429 as RateLimitError and captures Retry-After', async () => {
    queueFetchResponses([jsonResponse(429, { error: { message: 'Rate limited' } }, { 'Retry-After': '2' })]);
    const error = await buildAdapter()
      .generate([userMessage('hi')], 'gpt-4o-mini', createGenerationConfig())
      .catch((e: unknown) => e);
    expect(error).toBeInstanceOf(RateLimitError);
    expect((error as RateLimitError).retryAfterMs).toBe(2000);
  });

  it('classifies a context-length 400 as ContextLengthError', async () => {
    queueFetchResponses([
      jsonResponse(400, { error: { message: "This model's maximum context length is 8192 tokens." } }),
    ]);
    await expect(buildAdapter().generate([userMessage('hi')], 'gpt-4o-mini', createGenerationConfig())).rejects.toBeInstanceOf(
      ContextLengthError,
    );
  });

  it('classifies a 500 as ServerError', async () => {
    queueFetchResponses([jsonResponse(500, { error: { message: 'Internal error' } })]);
    await expect(buildAdapter().generate([userMessage('hi')], 'gpt-4o-mini', createGenerationConfig())).rejects.toBeInstanceOf(
      ServerError,
    );
  });

  it('surfaces a TimeoutError when the request exceeds config.timeoutMs', async () => {
    vi.stubGlobal('fetch', abortAwareFetch(500, jsonResponse(200, {})));
    await expect(
      buildAdapter().generate([userMessage('hi')], 'gpt-4o-mini', createGenerationConfig({ timeoutMs: 10 })),
    ).rejects.toMatchObject({ category: 'timeout' });
  });
});

describe('OpenAIAdapter.generateStream', () => {
  it('normalizes SSE chunks and stops at [DONE]', async () => {
    queueFetchResponses([
      sseResponse([
        sseDataEvent({ id: 'c1', model: 'gpt-4o-mini', choices: [{ delta: { content: 'Hel' }, finish_reason: null }] }),
        sseDataEvent({ id: 'c1', model: 'gpt-4o-mini', choices: [{ delta: { content: 'lo' }, finish_reason: null }] }),
        sseDataEvent({
          id: 'c1',
          model: 'gpt-4o-mini',
          choices: [{ delta: {}, finish_reason: 'stop' }],
          usage: { prompt_tokens: 4, completion_tokens: 2, total_tokens: 6 },
        }),
        sseDataEvent('[DONE]'),
      ]),
    ]);

    const chunks = [];
    for await (const chunk of buildAdapter().generateStream([userMessage('hi')], 'gpt-4o-mini', createGenerationConfig({ stream: true }))) {
      chunks.push(chunk);
    }

    expect(chunks.map((c) => c.delta).join('')).toBe('Hello');
    expect(chunks.at(-1)?.finishReason).toBe(FinishReason.Stop);
    expect(chunks.at(-1)?.usage).toEqual({ promptTokens: 4, completionTokens: 2, totalTokens: 6 });
  });
});

describe('OpenAIAdapter capabilities and context limits', () => {
  it('reports a known context limit for gpt-4o-mini', () => {
    expect(buildAdapter().getContextLimit('gpt-4o-mini')).toBe(128_000);
  });

  it('falls back to a conservative context limit for an unregistered model', () => {
    expect(buildAdapter().getContextLimit('some-future-model')).toBeGreaterThan(0);
  });
});
