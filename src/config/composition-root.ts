import 'dotenv/config';
import { loadLLMConfig, type EnvSource } from '../infrastructure/llm/configuration/llm.config.js';
import { ModelRegistry } from '../infrastructure/llm/configuration/model-registry.js';
import { TokenizerService } from '../infrastructure/llm/tokenization/tokenizer.service.js';
import { LLMFactory } from '../infrastructure/llm/factory/llm.factory.js';
import { ProviderHealthRegistry } from '../infrastructure/llm/routing/provider-health.js';
import { RetryPolicy } from '../infrastructure/llm/routing/retry-policy.js';
import { FallbackManager } from '../infrastructure/llm/routing/fallback-manager.js';
import { LLMRouter } from '../infrastructure/llm/routing/llm-router.js';
import { LLMService } from '../application/services/llm.service.js';
import type { ILLMService } from '../application/interfaces/illm-service.interface.js';
import type { GenerateOptions } from '../infrastructure/llm/adapters/base-llm.adapter.js';
import type { MessageList } from '../domain/models/message.js';
import type { GenerationConfigInput } from '../domain/models/generation-config.js';
import type { ModelResponse } from '../domain/models/model-response.js';
import type { StreamChunk } from '../domain/models/stream-chunk.js';
import { ConsoleLogger, type ILogger } from '../infrastructure/observability/logger.js';
import { InMemoryMetricsRecorder, type IMetricsRecorder } from '../infrastructure/observability/metrics.js';

export interface CompositionRoot {
  readonly logger: ILogger;
  readonly metrics: IMetricsRecorder;
  readonly llmService: ILLMService;
}

/**
 * Builds the real LLMService (env config, adapters, routing) only on first
 * use, and re-attempts on every subsequent call if that build failed. This
 * lets the process (and its HTTP server) start and serve `/health` even
 * when the LLM provider configuration is missing or invalid -- the
 * resulting ConfigurationError only ever surfaces to the request that
 * actually needed the LLM, never at process startup.
 */
class LazyLLMService implements ILLMService {
  private cached?: LLMService;

  constructor(
    private readonly env: EnvSource,
    private readonly logger: ILogger,
    private readonly metrics: IMetricsRecorder,
  ) {}

  private resolve(): LLMService {
    if (!this.cached) {
      const config = loadLLMConfig(this.env);
      const modelRegistry = new ModelRegistry();
      const tokenizer = new TokenizerService();
      const factory = new LLMFactory(config, { modelRegistry, tokenizer, logger: this.logger });
      const health = new ProviderHealthRegistry(config.circuitBreaker);
      const retryPolicy = new RetryPolicy(config.retry);
      const fallbackManager = new FallbackManager(health, retryPolicy, this.logger, this.metrics);
      const router = new LLMRouter(config, factory, fallbackManager);
      this.cached = new LLMService(router, this.logger);
    }
    return this.cached;
  }

  generate(messages: MessageList, config?: GenerationConfigInput, options?: GenerateOptions): Promise<ModelResponse> {
    return this.resolve().generate(messages, config, options);
  }

  generateStream(messages: MessageList, config?: GenerationConfigInput, options?: GenerateOptions): AsyncIterable<StreamChunk> {
    return this.resolve().generateStream(messages, config, options);
  }

  countTokens(messages: MessageList): Promise<number> {
    return this.resolve().countTokens(messages);
  }
}

/**
 * The single place that wires concrete infrastructure (adapters, HTTP,
 * env-based configuration) into the abstractions the application layer
 * depends on. Nothing outside this file should ever call `new` on a
 * provider adapter directly.
 */
export function buildCompositionRoot(env: EnvSource = process.env): CompositionRoot {
  const logger: ILogger = new ConsoleLogger((env.LOG_LEVEL as 'debug' | 'info' | 'warn' | 'error' | undefined) ?? 'info');
  const metrics: IMetricsRecorder = new InMemoryMetricsRecorder();
  const llmService: ILLMService = new LazyLLMService(env, logger, metrics);

  return { logger, metrics, llmService };
}
