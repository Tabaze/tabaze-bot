import { ErrorCategory } from '../enums/error-category.enum.js';
import { LLMProvider } from '../enums/provider.enum.js';

export interface LLMErrorOptions {
  readonly provider?: LLMProvider;
  readonly statusCode?: number;
  readonly retryAfterMs?: number;
  readonly cause?: unknown;
}

/**
 * Base type for every normalized failure that can originate from a
 * provider request. `retryable` and `fallbackable` are decided once, here,
 * by the class that best knows the failure semantics -- the router never
 * has to re-derive routing decisions from raw HTTP status codes.
 */
export abstract class LLMError extends Error {
  abstract readonly category: ErrorCategory;
  abstract readonly retryable: boolean;
  abstract readonly fallbackable: boolean;

  readonly provider?: LLMProvider;
  readonly statusCode?: number;
  readonly retryAfterMs?: number;

  constructor(message: string, options: LLMErrorOptions = {}) {
    super(message, options.cause !== undefined ? { cause: options.cause } : undefined);
    this.name = new.target.name;
    this.provider = options.provider;
    this.statusCode = options.statusCode;
    this.retryAfterMs = options.retryAfterMs;
  }
}

export class AuthenticationError extends LLMError {
  readonly category = ErrorCategory.Authentication;
  readonly retryable = false;
  readonly fallbackable = true;
}

export class AuthorizationError extends LLMError {
  readonly category = ErrorCategory.Authorization;
  readonly retryable = false;
  readonly fallbackable = true;
}

export class RateLimitError extends LLMError {
  readonly category = ErrorCategory.RateLimit;
  readonly retryable = true;
  readonly fallbackable = true;
}

export class TimeoutError extends LLMError {
  readonly category = ErrorCategory.Timeout;
  readonly retryable = true;
  readonly fallbackable = true;
}

export class NetworkError extends LLMError {
  readonly category = ErrorCategory.Network;
  readonly retryable = true;
  readonly fallbackable = true;
}

export class ProviderUnavailableError extends LLMError {
  readonly category = ErrorCategory.ProviderUnavailable;
  readonly retryable = true;
  readonly fallbackable = true;
}

export class InvalidRequestError extends LLMError {
  readonly category = ErrorCategory.InvalidRequest;
  readonly retryable = false;
  readonly fallbackable = false;
}

export class ContextLengthError extends LLMError {
  readonly category = ErrorCategory.ContextLength;
  readonly retryable = false;
  readonly fallbackable = false;
}

export class ContentFilterError extends LLMError {
  readonly category = ErrorCategory.ContentFilter;
  readonly retryable = false;
  readonly fallbackable = false;
}

export class ServerError extends LLMError {
  readonly category = ErrorCategory.ServerError;
  readonly retryable = true;
  readonly fallbackable = true;
}

/** The request was aborted by the caller (client disconnect, explicit cancel). */
export class CancelledError extends LLMError {
  readonly category = ErrorCategory.Cancelled;
  readonly retryable = false;
  readonly fallbackable = false;
}

export class UnknownLLMError extends LLMError {
  readonly category = ErrorCategory.Unknown;
  readonly retryable = false;
  readonly fallbackable = true;
}
