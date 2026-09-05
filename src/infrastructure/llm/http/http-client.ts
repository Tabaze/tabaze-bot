import { LLMProvider } from '../../../domain/enums/provider.enum.js';
import { CancelledError, NetworkError, TimeoutError } from '../../../domain/errors/llm-error.js';

export interface HttpRequestOptions {
  readonly url: string;
  readonly method: 'GET' | 'POST';
  readonly headers: Record<string, string>;
  readonly body?: unknown;
  readonly timeoutMs: number;
  readonly signal?: AbortSignal;
}

/** Combines a caller-supplied abort signal with an internal timeout signal. */
function combineSignals(external: AbortSignal | undefined, timeoutMs: number): { signal: AbortSignal; cleanup: () => void } {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(new DOMException('Request timed out', 'TimeoutError')), timeoutMs);

  const onExternalAbort = (): void => controller.abort(external?.reason);
  if (external) {
    if (external.aborted) controller.abort(external.reason);
    else external.addEventListener('abort', onExternalAbort, { once: true });
  }

  const cleanup = (): void => {
    clearTimeout(timer);
    external?.removeEventListener('abort', onExternalAbort);
  };

  return { signal: controller.signal, cleanup };
}

function isTimeoutAbort(reason: unknown): boolean {
  return reason instanceof DOMException && reason.name === 'TimeoutError';
}

/**
 * Issues an HTTP request with a bounded timeout and cancellation support,
 * normalizing transport-level failures (as opposed to HTTP error status
 * codes, which each adapter classifies itself) into typed LLMErrors.
 */
export async function executeHttpRequest(options: HttpRequestOptions, provider: LLMProvider): Promise<Response> {
  const { signal, cleanup } = combineSignals(options.signal, options.timeoutMs);
  try {
    const response = await fetch(options.url, {
      method: options.method,
      headers: options.headers,
      body: options.body !== undefined ? JSON.stringify(options.body) : undefined,
      signal,
    });
    return response;
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      if (options.signal?.aborted) {
        throw new CancelledError('Request was cancelled by the caller.', { provider, cause: error });
      }
      throw new TimeoutError(`Request to ${provider} timed out after ${options.timeoutMs}ms.`, { provider, cause: error });
    }
    if (isTimeoutAbort(signal.reason)) {
      throw new TimeoutError(`Request to ${provider} timed out after ${options.timeoutMs}ms.`, { provider, cause: error });
    }
    throw new NetworkError(`Network error while contacting ${provider}: ${(error as Error).message}`, {
      provider,
      cause: error,
    });
  } finally {
    cleanup();
  }
}
