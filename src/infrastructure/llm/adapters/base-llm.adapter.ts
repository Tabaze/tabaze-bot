import { LLMProvider } from '../../../domain/enums/provider.enum.js';
import type { MessageList } from '../../../domain/models/message.js';
import type { GenerationConfig } from '../../../domain/models/generation-config.js';
import type { ModelResponse } from '../../../domain/models/model-response.js';
import type { StreamChunk } from '../../../domain/models/stream-chunk.js';
import type { ProviderCapabilities } from '../../../domain/models/provider-capabilities.js';
import {
  AuthenticationError,
  AuthorizationError,
  ContextLengthError,
  InvalidRequestError,
  LLMError,
  RateLimitError,
  ServerError,
  UnknownLLMError,
} from '../../../domain/errors/llm-error.js';

export interface GenerateOptions {
  /** Propagated to the underlying HTTP request; aborting it cancels the in-flight provider call. */
  readonly signal?: AbortSignal;
}

/**
 * The contract every provider adapter implements. The application and
 * routing layers depend only on this abstraction -- never on a provider
 * SDK or its response types.
 */
export abstract class BaseLLMAdapter {
  abstract readonly provider: LLMProvider;

  abstract generate(
    messages: MessageList,
    model: string,
    config: GenerationConfig,
    options?: GenerateOptions,
  ): Promise<ModelResponse>;

  abstract generateStream(
    messages: MessageList,
    model: string,
    config: GenerationConfig,
    options?: GenerateOptions,
  ): AsyncIterable<StreamChunk>;

  abstract tokenizeAndCount(messages: MessageList, model: string): Promise<number>;

  abstract getCapabilities(model: string): ProviderCapabilities;

  abstract getContextLimit(model: string): number;

  /**
   * Best-effort classification of a provider's HTTP error status into a
   * normalized LLMError. Subclasses call this first and override the
   * result when the provider's error body carries more specific
   * information (e.g. a 400 that actually means "context length
   * exceeded" or "content filtered").
   */
  protected classifyByStatus(status: number, message: string, retryAfterMs?: number): LLMError {
    if (status === 401) {
      return new AuthenticationError(message, { provider: this.provider, statusCode: status });
    }
    if (status === 403) {
      return new AuthorizationError(message, { provider: this.provider, statusCode: status });
    }
    if (status === 429) {
      return new RateLimitError(message, { provider: this.provider, statusCode: status, retryAfterMs });
    }
    if (status === 400 || status === 404 || status === 422) {
      return new InvalidRequestError(message, { provider: this.provider, statusCode: status });
    }
    if (status >= 500) {
      return new ServerError(message, { provider: this.provider, statusCode: status });
    }
    return new UnknownLLMError(message, { provider: this.provider, statusCode: status });
  }

  protected contextLengthError(message: string, statusCode?: number): ContextLengthError {
    return new ContextLengthError(message, { provider: this.provider, statusCode });
  }

  protected parseRetryAfterMs(headers: Headers): number | undefined {
    const raw = headers.get('retry-after');
    if (!raw) return undefined;
    const seconds = Number(raw);
    if (Number.isFinite(seconds)) return seconds * 1000;
    const asDate = Date.parse(raw);
    if (!Number.isNaN(asDate)) return Math.max(0, asDate - Date.now());
    return undefined;
  }
}
