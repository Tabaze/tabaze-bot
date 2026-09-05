import { randomUUID } from 'node:crypto';
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
import type { OpenAIProviderConfig } from '../configuration/llm.config.js';
import { ModelRegistry } from '../configuration/model-registry.js';
import { TokenizerService } from '../tokenization/tokenizer.service.js';
import type { ILogger } from '../../observability/logger.js';

interface OpenAIChatMessage {
  readonly role: 'system' | 'user' | 'assistant';
  readonly content: string;
}

interface OpenAIChoice {
  readonly index: number;
  readonly message?: { readonly content: string | null };
  readonly delta?: { readonly content?: string | null };
  readonly finish_reason: string | null;
}

interface OpenAIUsage {
  readonly prompt_tokens: number;
  readonly completion_tokens: number;
  readonly total_tokens: number;
}

interface OpenAIChatCompletion {
  readonly id: string;
  readonly model: string;
  readonly choices: readonly OpenAIChoice[];
  readonly usage?: OpenAIUsage;
}

interface OpenAIErrorBody {
  readonly error?: { readonly message?: string; readonly code?: string; readonly type?: string };
}

function mapRole(role: MessageRole): OpenAIChatMessage['role'] {
  switch (role) {
    case MessageRole.System:
      return 'system';
    case MessageRole.User:
      return 'user';
    case MessageRole.Assistant:
      return 'assistant';
    default: {
      const exhaustiveCheck: never = role;
      throw new Error(`Unsupported message role: ${String(exhaustiveCheck)}`);
    }
  }
}

function mapFinishReason(reason: string | null | undefined): FinishReason {
  switch (reason) {
    case 'stop':
      return FinishReason.Stop;
    case 'length':
      return FinishReason.MaxTokens;
    case 'content_filter':
      return FinishReason.ContentFilter;
    case null:
    case undefined:
      return FinishReason.Unknown;
    default:
      return FinishReason.Unknown;
  }
}

function toUsage(usage: OpenAIUsage | undefined): TokenUsage {
  return {
    promptTokens: usage?.prompt_tokens ?? 0,
    completionTokens: usage?.completion_tokens ?? 0,
    totalTokens: usage?.total_tokens ?? 0,
  };
}

/**
 * Adapter for the OpenAI Chat Completions API. Also serves as the
 * reference implementation the other adapters mirror in structure.
 */
export class OpenAIAdapter extends BaseLLMAdapter {
  readonly provider = LLMProvider.OpenAI;

  constructor(
    private readonly config: OpenAIProviderConfig,
    private readonly modelRegistry: ModelRegistry,
    private readonly tokenizer: TokenizerService,
    private readonly logger: ILogger,
  ) {
    super();
  }

  private headers(): Record<string, string> {
    return {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${this.config.apiKey}`,
    };
  }

  private buildRequestBody(messages: MessageList, model: string, config: GenerationConfig, stream: boolean): Record<string, unknown> {
    const body: Record<string, unknown> = {
      model,
      messages: messages.map((m): OpenAIChatMessage => ({ role: mapRole(m.role), content: m.content })),
      stream,
    };
    if (config.temperature !== undefined) body.temperature = config.temperature;
    if (config.topP !== undefined) body.top_p = config.topP;
    if (config.maxTokens !== undefined) body.max_tokens = config.maxTokens;
    if (config.presencePenalty !== undefined) body.presence_penalty = config.presencePenalty;
    if (config.frequencyPenalty !== undefined) body.frequency_penalty = config.frequencyPenalty;
    if (config.stopSequences !== undefined && config.stopSequences.length > 0) body.stop = config.stopSequences;
    if (stream) body.stream_options = { include_usage: true };
    return body;
  }

  private async parseErrorBody(response: Response): Promise<string> {
    try {
      const body = (await response.json()) as OpenAIErrorBody;
      return body.error?.message ?? `OpenAI request failed with status ${response.status}`;
    } catch {
      return `OpenAI request failed with status ${response.status}`;
    }
  }

  private async raiseForStatus(response: Response): Promise<never> {
    const message = await this.parseErrorBody(response);
    const retryAfterMs = this.parseRetryAfterMs(response.headers);
    if (response.status === 400 && /context length|maximum context/i.test(message)) {
      throw this.contextLengthError(message, response.status);
    }
    throw this.classifyByStatus(response.status, message, retryAfterMs);
  }

  async generate(messages: MessageList, model: string, config: GenerationConfig, options?: GenerateOptions): Promise<ModelResponse> {
    const response = await executeHttpRequest(
      {
        url: `${this.config.baseUrl}/chat/completions`,
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

    const completion = (await response.json()) as OpenAIChatCompletion;
    const choice = completion.choices[0];
    if (!choice) {
      throw new UnknownLLMError('OpenAI response contained no choices.', { provider: this.provider });
    }

    return {
      id: completion.id,
      provider: this.provider,
      model: completion.model,
      content: choice.message?.content ?? '',
      usage: toUsage(completion.usage),
      finishReason: mapFinishReason(choice.finish_reason),
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
        url: `${this.config.baseUrl}/chat/completions`,
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
      throw new NetworkError('OpenAI streaming response had no body.', { provider: this.provider });
    }

    const generationId = randomUUID();
    let resolvedModel = model;

    for await (const event of parseSseStream(response.body, this.provider, options?.signal)) {
      if (event.data === '[DONE]') break;

      let parsed: OpenAIChatCompletion;
      try {
        parsed = JSON.parse(event.data) as OpenAIChatCompletion;
      } catch (error) {
        this.logger.warn('Failed to parse OpenAI stream chunk', { provider: this.provider, error: (error as Error).message });
        continue;
      }

      resolvedModel = parsed.model || resolvedModel;
      const choice = parsed.choices[0];
      const delta = choice?.delta?.content ?? '';
      const finishReason = choice?.finish_reason ? mapFinishReason(choice.finish_reason) : null;
      const usage = parsed.usage ? toUsage(parsed.usage) : null;

      if (delta.length === 0 && finishReason === null && usage === null) continue;

      yield {
        id: parsed.id || generationId,
        provider: this.provider,
        model: resolvedModel,
        delta,
        finishReason,
        usage,
      };
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
