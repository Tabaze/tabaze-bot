import { randomUUID } from 'node:crypto';
import { LLMProvider } from '../../../domain/enums/provider.enum.js';
import { MessageRole, type MessageList } from '../../../domain/models/message.js';
import type { GenerationConfig } from '../../../domain/models/generation-config.js';
import type { ModelResponse, TokenUsage } from '../../../domain/models/model-response.js';
import type { StreamChunk } from '../../../domain/models/stream-chunk.js';
import { FinishReason } from '../../../domain/enums/finish-reason.enum.js';
import { DEFAULT_CAPABILITIES, type ProviderCapabilities } from '../../../domain/models/provider-capabilities.js';
import { NetworkError, UnknownLLMError } from '../../../domain/errors/llm-error.js';
import { BaseLLMAdapter, type GenerateOptions } from './base-llm.adapter.js';
import { executeHttpRequest } from '../http/http-client.js';
import { parseSseStream } from '../http/sse-parser.js';
import type { LocalProviderConfig } from '../configuration/llm.config.js';
import { ModelRegistry } from '../configuration/model-registry.js';
import { TokenizerService } from '../tokenization/tokenizer.service.js';
import type { ILogger } from '../../observability/logger.js';

interface CompatibleChatMessage {
  readonly role: 'system' | 'user' | 'assistant';
  readonly content: string;
}

interface CompatibleChoice {
  readonly message?: { readonly content: string | null };
  readonly delta?: { readonly content?: string | null };
  readonly finish_reason: string | null;
}

interface CompatibleUsage {
  readonly prompt_tokens: number;
  readonly completion_tokens: number;
  readonly total_tokens: number;
}

interface CompatibleChatCompletion {
  readonly id?: string;
  readonly model?: string;
  readonly choices: readonly CompatibleChoice[];
  readonly usage?: CompatibleUsage;
}

interface CompatibleErrorBody {
  readonly error?: { readonly message?: string } | string;
  readonly message?: string;
}

function mapRole(role: MessageRole): CompatibleChatMessage['role'] {
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
    default:
      return FinishReason.Unknown;
  }
}

function toUsage(usage: CompatibleUsage | undefined): TokenUsage {
  return {
    promptTokens: usage?.prompt_tokens ?? 0,
    completionTokens: usage?.completion_tokens ?? 0,
    totalTokens: usage?.total_tokens ?? 0,
  };
}

const LOCAL_CAPABILITIES: ProviderCapabilities = {
  ...DEFAULT_CAPABILITIES,
  presencePenalty: true,
  frequencyPenalty: true,
};

/**
 * Adapter for any inference server that speaks the OpenAI-compatible HTTP
 * contract (Ollama's `/v1` shim, vLLM's OpenAI server, LM Studio, etc.).
 * It never assumes a specific one of these is running -- only that
 * `/chat/completions` follows the OpenAI request/response shape -- and it
 * never sends parameters (like `stream_options.include_usage`) that are
 * not reliably supported across that whole family of servers.
 */
export class OpenAICompatibleAdapter extends BaseLLMAdapter {
  readonly provider = LLMProvider.Local;

  constructor(
    private readonly config: LocalProviderConfig,
    private readonly modelRegistry: ModelRegistry,
    private readonly tokenizer: TokenizerService,
    private readonly logger: ILogger,
  ) {
    super();
  }

  private headers(): Record<string, string> {
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (this.config.apiKey) headers.Authorization = `Bearer ${this.config.apiKey}`;
    return headers;
  }

  private resolveModel(model: string): string {
    return model || this.config.model;
  }

  private buildRequestBody(messages: MessageList, model: string, config: GenerationConfig, stream: boolean): Record<string, unknown> {
    const body: Record<string, unknown> = {
      model: this.resolveModel(model),
      messages: messages.map((m): CompatibleChatMessage => ({ role: mapRole(m.role), content: m.content })),
      stream,
    };
    if (config.temperature !== undefined) body.temperature = config.temperature;
    if (config.topP !== undefined) body.top_p = config.topP;
    if (config.maxTokens !== undefined) body.max_tokens = config.maxTokens;
    if (config.presencePenalty !== undefined) body.presence_penalty = config.presencePenalty;
    if (config.frequencyPenalty !== undefined) body.frequency_penalty = config.frequencyPenalty;
    if (config.stopSequences !== undefined && config.stopSequences.length > 0) body.stop = config.stopSequences;
    return body;
  }

  private async parseErrorBody(response: Response): Promise<string> {
    try {
      const body = (await response.json()) as CompatibleErrorBody;
      if (typeof body.error === 'string') return body.error;
      return body.error?.message ?? body.message ?? `Local provider request failed with status ${response.status}`;
    } catch {
      return `Local provider request failed with status ${response.status}`;
    }
  }

  private async raiseForStatus(response: Response): Promise<never> {
    const message = await this.parseErrorBody(response);
    const retryAfterMs = this.parseRetryAfterMs(response.headers);
    if (response.status === 400 && /context length|too many tokens|context window/i.test(message)) {
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

    const completion = (await response.json()) as CompatibleChatCompletion;
    const choice = completion.choices[0];
    if (!choice) {
      throw new UnknownLLMError('Local provider response contained no choices.', { provider: this.provider });
    }

    return {
      id: completion.id ?? randomUUID(),
      provider: this.provider,
      model: completion.model ?? this.resolveModel(model),
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
      throw new NetworkError('Local provider streaming response had no body.', { provider: this.provider });
    }

    const generationId = randomUUID();
    let resolvedModel = this.resolveModel(model);

    for await (const event of parseSseStream(response.body, this.provider, options?.signal)) {
      if (event.data === '[DONE]') break;

      let parsed: CompatibleChatCompletion;
      try {
        parsed = JSON.parse(event.data) as CompatibleChatCompletion;
      } catch (error) {
        this.logger.warn('Failed to parse local provider stream chunk', { provider: this.provider, error: (error as Error).message });
        continue;
      }

      resolvedModel = parsed.model || resolvedModel;
      const choice = parsed.choices[0];
      const delta = choice?.delta?.content ?? '';
      const finishReason = choice?.finish_reason ? mapFinishReason(choice.finish_reason) : null;
      const usage = parsed.usage ? toUsage(parsed.usage) : null;

      if (delta.length === 0 && finishReason === null && usage === null) continue;

      yield {
        id: parsed.id ?? generationId,
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

  getCapabilities(): ProviderCapabilities {
    return LOCAL_CAPABILITIES;
  }

  getContextLimit(model: string): number {
    return this.modelRegistry.getContextLimit(this.provider, this.resolveModel(model));
  }
}
