import { afterEach, describe, expect, it, vi } from 'vitest';
import { AnthropicAdapter } from '../../src/infrastructure/llm/adapters/anthropic.adapter.js';
import { ModelRegistry } from '../../src/infrastructure/llm/configuration/model-registry.js';
import { TokenizerService } from '../../src/infrastructure/llm/tokenization/tokenizer.service.js';
import { NullLogger } from '../../src/infrastructure/observability/logger.js';
import { createGenerationConfig } from '../../src/domain/models/generation-config.js';
import { assistantMessage, systemMessage, userMessage } from '../../src/domain/models/message.js';
import { FinishReason } from '../../src/domain/enums/finish-reason.enum.js';
import { AuthenticationError } from '../../src/domain/errors/llm-error.js';
import { jsonResponse, queueFetchResponses, requestBody, sseResponse, sseTypedEvent } from '../helpers/mock-fetch.js';

const CONFIG = { apiKey: 'anthropic-test-key', baseUrl: 'https://api.anthropic.com/v1' };

function buildAdapter(): AnthropicAdapter {
  return new AnthropicAdapter(CONFIG, new ModelRegistry(), new TokenizerService(), new NullLogger());
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('AnthropicAdapter system-message extraction', () => {
  it('extracts system-role messages into the top-level `system` field and excludes them from `messages`', async () => {
    const { calls } = queueFetchResponses([
      jsonResponse(200, {
        id: 'msg_1',
        model: 'claude-3-5-sonnet-20241022',
        content: [{ type: 'text', text: 'hi' }],
        stop_reason: 'end_turn',
        usage: { input_tokens: 10, output_tokens: 2 },
      }),
    ]);

    await buildAdapter().generate(
      [systemMessage('You are terse.'), userMessage('hello'), assistantMessage('hi there'), userMessage('again')],
      'claude-3-5-sonnet-20241022',
      createGenerationConfig(),
    );

    const body = requestBody(calls[0]!);
    expect(body.system).toBe('You are terse.');
    expect(body.messages).toEqual([
      { role: 'user', content: 'hello' },
      { role: 'assistant', content: 'hi there' },
      { role: 'user', content: 'again' },
    ]);
  });

  it('joins multiple system messages with a blank line', async () => {
    const { calls } = queueFetchResponses([
      jsonResponse(200, {
        id: 'msg_2',
        model: 'claude-3-5-sonnet-20241022',
        content: [{ type: 'text', text: 'hi' }],
        stop_reason: 'end_turn',
        usage: { input_tokens: 10, output_tokens: 2 },
      }),
    ]);

    await buildAdapter().generate(
      [systemMessage('First rule.'), systemMessage('Second rule.'), userMessage('hello')],
      'claude-3-5-sonnet-20241022',
      createGenerationConfig(),
    );

    expect(requestBody(calls[0]!).system).toBe('First rule.\n\nSecond rule.');
  });

  it('omits `system` entirely when there are no system messages', async () => {
    const { calls } = queueFetchResponses([
      jsonResponse(200, {
        id: 'msg_3',
        model: 'claude-3-5-sonnet-20241022',
        content: [{ type: 'text', text: 'hi' }],
        stop_reason: 'end_turn',
        usage: { input_tokens: 10, output_tokens: 2 },
      }),
    ]);

    await buildAdapter().generate([userMessage('hello')], 'claude-3-5-sonnet-20241022', createGenerationConfig());
    expect('system' in requestBody(calls[0]!)).toBe(false);
  });

  it('defaults max_tokens when the caller does not provide one, since Anthropic requires it', async () => {
    const { calls } = queueFetchResponses([
      jsonResponse(200, {
        id: 'msg_4',
        model: 'claude-3-5-sonnet-20241022',
        content: [{ type: 'text', text: 'hi' }],
        stop_reason: 'end_turn',
        usage: { input_tokens: 10, output_tokens: 2 },
      }),
    ]);
    await buildAdapter().generate([userMessage('hi')], 'claude-3-5-sonnet-20241022', createGenerationConfig());
    expect(requestBody(calls[0]!).max_tokens).toBe(4_096);
  });
});

describe('AnthropicAdapter.generate', () => {
  it('joins text content blocks and maps stop_reason to FinishReason', async () => {
    queueFetchResponses([
      jsonResponse(200, {
        id: 'msg_5',
        model: 'claude-3-5-sonnet-20241022',
        content: [{ type: 'text', text: 'Hello' }, { type: 'text', text: ', world' }],
        stop_reason: 'max_tokens',
        usage: { input_tokens: 12, output_tokens: 4 },
      }),
    ]);

    const response = await buildAdapter().generate([userMessage('hi')], 'claude-3-5-sonnet-20241022', createGenerationConfig());
    expect(response.content).toBe('Hello, world');
    expect(response.finishReason).toBe(FinishReason.MaxTokens);
    expect(response.usage).toEqual({ promptTokens: 12, completionTokens: 4, totalTokens: 16 });
  });

  it('sends the API key via x-api-key, not Authorization', async () => {
    const { calls } = queueFetchResponses([
      jsonResponse(200, { id: 'x', model: 'claude', content: [{ type: 'text', text: 'ok' }], stop_reason: 'end_turn', usage: {} }),
    ]);
    await buildAdapter().generate([userMessage('hi')], 'claude-3-5-sonnet-20241022', createGenerationConfig());
    const headers = calls[0]!.init.headers as Record<string, string>;
    expect(headers['x-api-key']).toBe('anthropic-test-key');
    expect(headers.Authorization).toBeUndefined();
  });

  it('classifies a 401 as AuthenticationError', async () => {
    queueFetchResponses([jsonResponse(401, { error: { type: 'authentication_error', message: 'invalid key' } })]);
    await expect(
      buildAdapter().generate([userMessage('hi')], 'claude-3-5-sonnet-20241022', createGenerationConfig()),
    ).rejects.toBeInstanceOf(AuthenticationError);
  });
});

describe('AnthropicAdapter.generateStream', () => {
  it('normalizes message_start/content_block_delta/message_delta events into StreamChunks', async () => {
    queueFetchResponses([
      sseResponse([
        sseTypedEvent('message_start', {
          type: 'message_start',
          message: { id: 'msg_1', model: 'claude-3-5-sonnet-20241022', usage: { input_tokens: 10 } },
        }),
        sseTypedEvent('content_block_delta', { type: 'content_block_delta', delta: { type: 'text_delta', text: 'Hel' } }),
        sseTypedEvent('content_block_delta', { type: 'content_block_delta', delta: { type: 'text_delta', text: 'lo' } }),
        sseTypedEvent('message_delta', { type: 'message_delta', delta: { stop_reason: 'end_turn' }, usage: { output_tokens: 5 } }),
        sseTypedEvent('message_stop', { type: 'message_stop' }),
      ]),
    ]);

    const chunks = [];
    for await (const chunk of buildAdapter().generateStream(
      [userMessage('hi')],
      'claude-3-5-sonnet-20241022',
      createGenerationConfig({ stream: true }),
    )) {
      chunks.push(chunk);
    }

    expect(chunks.map((c) => c.delta).join('')).toBe('Hello');
    expect(chunks.at(-1)?.finishReason).toBe(FinishReason.Stop);
    expect(chunks.at(-1)?.usage).toEqual({ promptTokens: 10, completionTokens: 5, totalTokens: 15 });
    expect(chunks.every((c) => c.id === 'msg_1')).toBe(true);
  });
});
