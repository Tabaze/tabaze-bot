import { LLMProvider } from '../../../domain/enums/provider.enum.js';
import { MessageRole, type MessageList } from '../../../domain/models/message.js';
import type { GenerationConfig } from '../../../domain/models/generation-config.js';
import type { ModelResponse, TokenUsage } from '../../../domain/models/model-response.js';
import type { StreamChunk } from '../../../domain/models/stream-chunk.js';
import { FinishReason } from '../../../domain/enums/finish-reason.enum.js';
import type { ProviderCapabilities } from '../../../domain/models/provider-capabilities.js';
import { NetworkError, UnknownLLMError } from '../../../domain/errors/llm-error.js';
import { BaseLLMAdapter, type GenerateOptions } from './base-llm.adapter.js';
import { executeHttpRequest } from '../http/http-client.js';
import { parseSseStream } from '../http/sse-parser.js';
import type { AnthropicProviderConfig } from '../configuration/llm.config.js';
import { ModelRegistry } from '../configuration/model-registry.js';
import { TokenizerService } from '../tokenization/tokenizer.service.js';
import type { ILogger } from '../../observability/logger.js';

const ANTHROPIC_VERSION = '2023-06-01';
const DEFAULT_MAX_TOKENS = 4_096;

interface AnthropicMessage {
  readonly role: 'user' | 'assistant';
  readonly content: string;
}

interface AnthropicContentBlock {
  readonly type: string;
  readonly text?: string;
}

interface AnthropicUsage {
  readonly input_tokens: number;
  readonly output_tokens: number;
}

interface AnthropicResponse {
  readonly id: string;
  readonly model: string;
  readonly content: readonly AnthropicContentBlock[];
  readonly stop_reason: string | null;
  readonly usage: AnthropicUsage;
}

interface AnthropicErrorBody {
  readonly error?: { readonly type?: string; readonly message?: string };
}

interface AnthropicStreamEvent {
  readonly type: string;
  readonly message?: { readonly id?: string; readonly model?: string; readonly usage?: AnthropicUsage };
  readonly delta?: { readonly type?: string; readonly text?: string; readonly stop_reason?: string | null };
  readonly usage?: AnthropicUsage;
  readonly error?: { readonly type?: string; readonly message?: string };
}

/** Extracts and concatenates system-role messages, since Anthropic takes `system` as a top-level field, not a message. */
function extractSystemAndMessages(messages: MessageList): { system: string | undefined; messages: AnthropicMessage[] } {
  const systemParts: string[] = [];
  const conversation: AnthropicMessage[] = [];

  for (const message of messages) {
    if (message.role === MessageRole.System) {
      systemParts.push(message.content);
    } else {
      conversation.push({ role: message.role === MessageRole.Assistant ? 'assistant' : 'user', content: message.content });
    }
  }

  return { system: systemParts.length > 0 ? systemParts.join('\n\n') : undefined, messages: conversation };
}

function mapFinishReason(reason: string | null | undefined): FinishReason {
  switch (reason) {
    case 'end_turn':
    case 'stop_sequence':
      return FinishReason.Stop;
    case 'max_tokens':
      return FinishReason.MaxTokens;
    case null:
    case undefined:
      return FinishReason.Unknown;
    default:
      return FinishReason.Unknown;
  }
}

function toUsage(usage: AnthropicUsage | undefined): TokenUsage {
  const promptTokens = usage?.input_tokens ?? 0;
  const completionTokens = usage?.output_tokens ?? 0;
  return { promptTokens, completionTokens, totalTokens: promptTokens + completionTokens };
}

/**
 * Adapter for the Anthropic Messages API. The defining difference from the
 * OpenAI-shaped providers is that system prompts are a top-level `system`
 * field rather than a message with role "system" -- extracted explicitly
 * in `extractSystemAndMessages` so that transformation never leaks past
 * this adapter.
 */
export class AnthropicAdapter extends BaseLLMAdapter {
  readonly provider = LLMProvider.Anthropic;

  constructor(
    private readonly config: AnthropicProviderConfig,
    private readonly modelRegistry: ModelRegistry,
    private readonly tokenizer: TokenizerService,
    private readonly logger: ILogger,
  ) {
    super();
  }

  private headers(): Record<string, string> {
    return {
      'Content-Type': 'application/json',
      'x-api-key': this.config.apiKey,
      'anthropic-version': ANTHROPIC_VERSION,
    };
  }

  private buildRequestBody(messages: MessageList, model: string, config: GenerationConfig, stream: boolean): Record<string, unknown> {
    const { system, messages: conversation } = extractSystemAndMessages(messages);
    const body: Record<string, unknown> = {
      model,
      messages: conversation,
      max_tokens: config.maxTokens ?? DEFAULT_MAX_TOKENS,
      stream,
    };
    if (system !== undefined) body.system = system;
    if (config.temperature !== undefined) body.temperature = config.temperature;
    if (config.topP !== undefined) body.top_p = config.topP;
    if (config.stopSequences !== undefined && config.stopSequences.length > 0) body.stop_sequences = config.stopSequences;
    return body;
  }

  private async raiseForStatus(response: Response): Promise<never> {
    let message = `Anthropic request failed with status ${response.status}`;
    let errorType: string | undefined;
    try {
      const body = (await response.json()) as AnthropicErrorBody;
      message = body.error?.message ?? message;
      errorType = body.error?.type;
    } catch {
      // Body was not JSON; keep the generic message.
    }

    const retryAfterMs = this.parseRetryAfterMs(response.headers);
    if (errorType === 'invalid_request_error' && /too long|maximum context|context window/i.test(message)) {
      throw this.contextLengthError(message, response.status);
    }
    throw this.classifyByStatus(response.status, message, retryAfterMs);
  }

  async generate(messages: MessageList, model: string, config: GenerationConfig, options?: GenerateOptions): Promise<ModelResponse> {
    const response = await executeHttpRequest(
      {
        url: `${this.config.baseUrl}/messages`,
        method: 'POST',
        headers: this.headers(),
        body: this.buildRequestBody(messages, model, config, false),
        timeoutMs: config.timeoutMs,
        signal: options?.signal,
      },
      this.provider,
    );

    if (!response.ok) {
      await this.raiseForStatus(response);
    }

    const completion = (await response.json()) as AnthropicResponse;
    const text = completion.content
      .filter((block) => block.type === 'text')
      .map((block) => block.text ?? '')
      .join('');

    return {
      id: completion.id,
      provider: this.provider,
      model: completion.model,
      content: text,
      usage: toUsage(completion.usage),
      finishReason: mapFinishReason(completion.stop_reason),
    };
  }

  async *generateStream(
    messages: MessageList,
    model: string,
    config: GenerationConfig,
    options?: GenerateOptions,
  ): AsyncIterable<StreamChunk> {
    const response = await executeHttpRequest(
      {
        url: `${this.config.baseUrl}/messages`,
        method: 'POST',
        headers: this.headers(),
        body: this.buildRequestBody(messages, model, config, true),
        timeoutMs: config.timeoutMs,
        signal: options?.signal,
      },
      this.provider,
    );

    if (!response.ok) {
      await this.raiseForStatus(response);
    }
    if (!response.body) {
      throw new NetworkError('Anthropic streaming response had no body.', { provider: this.provider });
    }

    let generationId = '';
    let resolvedModel = model;
    let inputTokens = 0;

    for await (const event of parseSseStream(response.body, this.provider, options?.signal)) {
      let parsed: AnthropicStreamEvent;
      try {
        parsed = JSON.parse(event.data) as AnthropicStreamEvent;
      } catch (error) {
        this.logger.warn('Failed to parse Anthropic stream chunk', { provider: this.provider, error: (error as Error).message });
        continue;
      }

      if (parsed.type === 'error') {
        throw this.classifyByStatus(500, parsed.error?.message ?? 'Anthropic streaming error');
      }

      if (parsed.type === 'message_start') {
        generationId = parsed.message?.id ?? generationId;
        resolvedModel = parsed.message?.model ?? resolvedModel;
        inputTokens = parsed.message?.usage?.input_tokens ?? inputTokens;
        continue;
      }

      if (parsed.type === 'content_block_delta' && parsed.delta?.type === 'text_delta') {
        yield {
          id: generationId,
          provider: this.provider,
          model: resolvedModel,
          delta: parsed.delta.text ?? '',
          finishReason: null,
          usage: null,
        };
        continue;
      }

      if (parsed.type === 'message_delta') {
        const outputTokens = parsed.usage?.output_tokens ?? 0;
        yield {
          id: generationId,
          provider: this.provider,
          model: resolvedModel,
          delta: '',
          finishReason: mapFinishReason(parsed.delta?.stop_reason),
          usage: { promptTokens: inputTokens, completionTokens: outputTokens, totalTokens: inputTokens + outputTokens },
        };
      }
    }
  }

  async tokenizeAndCount(messages: MessageList): Promise<number> {
    return this.tokenizer.count(messages, this.provider).count;
  }

  getCapabilities(model: string): ProviderCapabilities {
    return this.modelRegistry.getCapabilities(this.provider, model);
  }

  getContextLimit(model: string): number {
    return this.modelRegistry.getContextLimit(this.provider, model);
  }
}
