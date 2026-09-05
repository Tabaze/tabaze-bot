import { CancelledError, LLMError } from '../../../domain/errors/llm-error.js';
import type { RetryConfig } from '../configuration/llm.config.js';

export interface RetryDecision {
  readonly retryable: boolean;
  readonly retryAfterMs?: number;
}

/** Central place that turns a thrown error into a retry decision, so every call site agrees on what is retryable. */
export function classifyForRetry(error: unknown): RetryDecision {
  if (error instanceof LLMError) {
    return { retryable: error.retryable, retryAfterMs: error.retryAfterMs };
  }
  return { retryable: false };
}

export function canFallback(error: unknown): boolean {
  if (error instanceof CancelledError) return false;
  if (error instanceof LLMError) return error.fallbackable;
  return false;
}

/** Sleeps for `ms`, rejecting early with CancelledError if `signal` aborts first. */
export function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  if (ms <= 0) return Promise.resolve();
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new CancelledError('Retry backoff was cancelled by the caller.'));
      return;
    }
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort);
      resolve();
    }, ms);
    const onAbort = (): void => {
      clearTimeout(timer);
      reject(new CancelledError('Retry backoff was cancelled by the caller.'));
    };
    signal?.addEventListener('abort', onAbort, { once: true });
  });
}

/**
 * Bounded exponential backoff with jitter. Never retries indefinitely and
 * never retries an error classified as non-retryable (invalid requests,
 * authentication failures, cancellations, ...).
 */
export class RetryPolicy {
  constructor(
    private readonly config: RetryConfig,
    private readonly random: () => number = Math.random,
  ) {}

  computeDelayMs(attempt: number, retryAfterMs?: number): number {
    if (retryAfterMs !== undefined) {
      return Math.max(0, Math.min(retryAfterMs, this.config.maxDelayMs));
    }
    const exponential = this.config.baseDelayMs * 2 ** attempt;
    const capped = Math.min(exponential, this.config.maxDelayMs);
    const jitterFactor = 0.5 + this.random() * 0.5;
    return Math.round(capped * jitterFactor);
  }

  get maxRetries(): number {
    return this.config.maxRetries;
  }

  /**
   * Runs `fn`, retrying with backoff while the thrown error is retryable
   * and the retry budget is not exhausted. `attempt` starts at 0 for the
   * first (non-retry) call.
   */
  async execute<T>(fn: (attempt: number) => Promise<T>, signal?: AbortSignal): Promise<T> {
    let lastError: unknown;
    for (let attempt = 0; attempt <= this.config.maxRetries; attempt++) {
      try {
        return await fn(attempt);
      } catch (error) {
        lastError = error;
        const { retryable, retryAfterMs } = classifyForRetry(error);
        if (!retryable || attempt === this.config.maxRetries) {
          throw error;
        }
        await sleep(this.computeDelayMs(attempt, retryAfterMs), signal);
      }
    }
    throw lastError;
  }
}
