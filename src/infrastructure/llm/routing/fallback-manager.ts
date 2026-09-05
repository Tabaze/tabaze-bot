import type { MessageList } from '../../../domain/models/message.js';
import type { GenerationConfig } from '../../../domain/models/generation-config.js';
import type { ModelResponse } from '../../../domain/models/model-response.js';
import type { StreamChunk } from '../../../domain/models/stream-chunk.js';
import { ProviderUnavailableError } from '../../../domain/errors/llm-error.js';
import { StreamInterruptedError } from '../../../domain/errors/stream-interrupted-error.js';
import type { BaseLLMAdapter, GenerateOptions } from '../adapters/base-llm.adapter.js';
import { ProviderHealthRegistry } from './provider-health.js';
import { RetryPolicy, canFallback, classifyForRetry, sleep } from './retry-policy.js';
import type { ILogger } from '../../observability/logger.js';
import { IMetricsRecorder, METRIC_NAMES } from '../../observability/metrics.js';

export interface RouteTarget {
  readonly adapter: BaseLLMAdapter;
  readonly model: string;
}

/**
 * Executes a generate/generateStream call against a primary target with
 * bounded retries and circuit-breaker awareness, falling back to a
 * secondary target when the primary's failure is classified as
 * fallback-safe.
 *
 * Streaming fallback is the one place this class departs from "just retry
 * and fall back like the non-streaming path": once any chunk has reached
 * the caller, retrying or falling back would duplicate or contradict
 * output the caller already has, so the stream is terminated with a
 * StreamInterruptedError instead. See generateStream() for the exact rule.
 */
export class FallbackManager {
  constructor(
    private readonly health: ProviderHealthRegistry,
    private readonly retryPolicy: RetryPolicy,
    private readonly logger: ILogger,
    private readonly metrics: IMetricsRecorder,
  ) {}

  async generate(
    primary: RouteTarget,
    fallback: RouteTarget | undefined,
    messages: MessageList,
    config: GenerationConfig,
    options?: GenerateOptions,
  ): Promise<ModelResponse> {
    try {
      return await this.callWithBreakerAndRetry(primary, messages, config, options);
    } catch (error) {
      if (!fallback || !canFallback(error)) throw error;

      this.metrics.incrementCounter(METRIC_NAMES.FALLBACK_TOTAL, {
        from: primary.adapter.provider,
        to: fallback.adapter.provider,
      });
      this.logger.warn('Primary provider failed; falling back', {
        primaryProvider: primary.adapter.provider,
        primaryModel: primary.model,
        fallbackProvider: fallback.adapter.provider,
        fallbackModel: fallback.model,
        errorCategory: (error as { category?: string }).category,
      });

      return this.callWithBreakerAndRetry(fallback, messages, config, options);
    }
  }

  private async callWithBreakerAndRetry(
    target: RouteTarget,
    messages: MessageList,
    config: GenerationConfig,
    options?: GenerateOptions,
  ): Promise<ModelResponse> {
    const breaker = this.health.get(target.adapter.provider, target.model);

    return this.retryPolicy.execute(async () => {
      if (!breaker.canAttempt()) {
        throw new ProviderUnavailableError(`Circuit open for ${target.adapter.provider}/${target.model}.`, {
          provider: target.adapter.provider,
        });
      }
      breaker.recordAttemptStarted();

      const start = Date.now();
      const labels = { provider: target.adapter.provider, model: target.model };
      try {
        const result = await target.adapter.generate(messages, target.model, config, options);
        breaker.recordSuccess();
        this.metrics.recordHistogram(METRIC_NAMES.REQUEST_LATENCY, Date.now() - start, labels);
        this.metrics.incrementCounter(METRIC_NAMES.REQUESTS_TOTAL, { ...labels, outcome: 'success' });
        this.metrics.incrementCounter(METRIC_NAMES.TOKENS_PROMPT, labels, result.usage.promptTokens);
        this.metrics.incrementCounter(METRIC_NAMES.TOKENS_COMPLETION, labels, result.usage.completionTokens);
        return result;
      } catch (error) {
        breaker.recordFailure();
        this.metrics.incrementCounter(METRIC_NAMES.REQUESTS_FAILED, labels);
        throw error;
      }
    }, options?.signal);
  }

  async *generateStream(
    primary: RouteTarget,
    fallback: RouteTarget | undefined,
    messages: MessageList,
    config: GenerationConfig,
    options?: GenerateOptions,
  ): AsyncIterable<StreamChunk> {
    let accumulated = '';
    let emittedAny = false;

    try {
      for await (const chunk of this.streamWithBreakerAndRetry(primary, messages, config, options)) {
        emittedAny = true;
        accumulated += chunk.delta;
        yield chunk;
      }
      return;
    } catch (error) {
      if (emittedAny) {
        throw new StreamInterruptedError(
          `Streaming from ${primary.adapter.provider}/${primary.model} failed after content was already emitted; refusing to retry or fall back to avoid duplicated output.`,
          accumulated,
          { cause: error },
        );
      }

      if (!fallback || !canFallback(error)) throw error;

      this.metrics.incrementCounter(METRIC_NAMES.FALLBACK_TOTAL, {
        from: primary.adapter.provider,
        to: fallback.adapter.provider,
      });
      this.logger.warn('Primary provider stream failed before any output; falling back', {
        primaryProvider: primary.adapter.provider,
        primaryModel: primary.model,
        fallbackProvider: fallback.adapter.provider,
        fallbackModel: fallback.model,
        errorCategory: (error as { category?: string }).category,
      });
    }

    for await (const chunk of this.streamWithBreakerAndRetry(fallback, messages, config, options)) {
      yield chunk;
    }
  }

  /**
   * Streams from a single target, retrying (bounded, with backoff) only
   * while zero chunks have been emitted for the current attempt. Once a
   * chunk is emitted, a subsequent failure is thrown as-is so the caller
   * (generateStream above) can apply the no-duplicate-output rule instead
   * of this method silently retrying mid-stream.
   */
  private async *streamWithBreakerAndRetry(
    target: RouteTarget,
    messages: MessageList,
    config: GenerationConfig,
    options?: GenerateOptions,
  ): AsyncGenerator<StreamChunk> {
    const breaker = this.health.get(target.adapter.provider, target.model);
    const labels = { provider: target.adapter.provider, model: target.model };

    for (let attempt = 0; ; attempt++) {
      if (!breaker.canAttempt()) {
        throw new ProviderUnavailableError(`Circuit open for ${target.adapter.provider}/${target.model}.`, {
          provider: target.adapter.provider,
        });
      }
      breaker.recordAttemptStarted();

      const start = Date.now();
      let emittedInAttempt = false;
      try {
        for await (const chunk of target.adapter.generateStream(messages, target.model, config, options)) {
          emittedInAttempt = true;
          yield chunk;
        }
        breaker.recordSuccess();
        this.metrics.recordHistogram(METRIC_NAMES.REQUEST_LATENCY, Date.now() - start, labels);
        this.metrics.incrementCounter(METRIC_NAMES.REQUESTS_TOTAL, { ...labels, outcome: 'success' });
        return;
      } catch (error) {
        breaker.recordFailure();
        this.metrics.incrementCounter(METRIC_NAMES.REQUESTS_FAILED, labels);

        if (emittedInAttempt) throw error;

        const { retryable, retryAfterMs } = classifyForRetry(error);
        if (!retryable || attempt >= this.retryPolicy.maxRetries) throw error;

        await sleep(this.retryPolicy.computeDelayMs(attempt, retryAfterMs), options?.signal);
      }
    }
  }
}
