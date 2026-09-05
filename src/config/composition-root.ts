import 'dotenv/config';
import { loadLLMConfig, type EnvSource, type LLMConfig } from '../infrastructure/llm/configuration/llm.config.js';
import { ModelRegistry } from '../infrastructure/llm/configuration/model-registry.js';
import { TokenizerService } from '../infrastructure/llm/tokenization/tokenizer.service.js';
import { LLMFactory } from '../infrastructure/llm/factory/llm.factory.js';
import { ProviderHealthRegistry } from '../infrastructure/llm/routing/provider-health.js';
import { RetryPolicy } from '../infrastructure/llm/routing/retry-policy.js';
import { FallbackManager } from '../infrastructure/llm/routing/fallback-manager.js';
import { LLMRouter } from '../infrastructure/llm/routing/llm-router.js';
import { LLMService } from '../application/services/llm.service.js';
import type { ILLMService } from '../application/interfaces/illm-service.interface.js';
import { ConsoleLogger, type ILogger } from '../infrastructure/observability/logger.js';
import { InMemoryMetricsRecorder, type IMetricsRecorder } from '../infrastructure/observability/metrics.js';

export interface CompositionRoot {
  readonly config: LLMConfig;
  readonly logger: ILogger;
  readonly metrics: IMetricsRecorder;
  readonly llmService: ILLMService;
}

/**
 * The single place that wires concrete infrastructure (adapters, HTTP,
 * env-based configuration) into the abstractions the application layer
 * depends on. Nothing outside this file should ever call `new` on a
 * provider adapter directly.
 */
export function buildCompositionRoot(env: EnvSource = process.env): CompositionRoot {
  const config = loadLLMConfig(env);

  const logger: ILogger = new ConsoleLogger((env.LOG_LEVEL as 'debug' | 'info' | 'warn' | 'error' | undefined) ?? 'info');
  const metrics: IMetricsRecorder = new InMemoryMetricsRecorder();

  const modelRegistry = new ModelRegistry();
  const tokenizer = new TokenizerService();

  const factory = new LLMFactory(config, { modelRegistry, tokenizer, logger });
  const health = new ProviderHealthRegistry(config.circuitBreaker);
  const retryPolicy = new RetryPolicy(config.retry);
  const fallbackManager = new FallbackManager(health, retryPolicy, logger, metrics);
  const router = new LLMRouter(config, factory, fallbackManager);
  const llmService = new LLMService(router, logger);

  return { config, logger, metrics, llmService };
}
