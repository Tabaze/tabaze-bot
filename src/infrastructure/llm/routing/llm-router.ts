import type { MessageList } from '../../../domain/models/message.js';
import type { GenerationConfig } from '../../../domain/models/generation-config.js';
import type { ModelResponse } from '../../../domain/models/model-response.js';
import type { StreamChunk } from '../../../domain/models/stream-chunk.js';
import type { ProviderCapabilities } from '../../../domain/models/provider-capabilities.js';
import type { GenerateOptions } from '../adapters/base-llm.adapter.js';
import type { LLMConfig } from '../configuration/llm.config.js';
import { LLMFactory } from '../factory/llm.factory.js';
import { FallbackManager, type RouteTarget } from './fallback-manager.js';

/**
 * Strategy layer: decides which provider/model handles a request (primary
 * vs. fallback) based on runtime configuration, and delegates the actual
 * call-with-resilience to FallbackManager. Nothing here is coupled to a
 * specific provider -- swapping PRIMARY_PROVIDER in configuration is
 * enough to change routing.
 */
export class LLMRouter {
  private readonly primaryTarget: RouteTarget;
  private readonly fallbackTarget: RouteTarget | undefined;

  constructor(
    private readonly config: LLMConfig,
    factory: LLMFactory,
    private readonly fallbackManager: FallbackManager,
  ) {
    this.primaryTarget = {
      adapter: factory.create(config.routing.primaryProvider),
      model: config.routing.primaryModel,
    };

    this.fallbackTarget =
      config.routing.fallbackProvider && config.routing.fallbackModel
        ? { adapter: factory.create(config.routing.fallbackProvider), model: config.routing.fallbackModel }
        : undefined;
  }

  async generate(messages: MessageList, config: GenerationConfig, options?: GenerateOptions): Promise<ModelResponse> {
    return this.fallbackManager.generate(this.primaryTarget, this.fallbackTarget, messages, config, options);
  }

  generateStream(messages: MessageList, config: GenerationConfig, options?: GenerateOptions): AsyncIterable<StreamChunk> {
    return this.fallbackManager.generateStream(this.primaryTarget, this.fallbackTarget, messages, config, options);
  }

  async tokenizeAndCount(messages: MessageList): Promise<number> {
    return this.primaryTarget.adapter.tokenizeAndCount(messages, this.primaryTarget.model);
  }

  getContextLimit(): number {
    return this.primaryTarget.adapter.getContextLimit(this.primaryTarget.model);
  }

  getCapabilities(): ProviderCapabilities {
    return this.primaryTarget.adapter.getCapabilities(this.primaryTarget.model);
  }

  getActiveModel(): string {
    return this.primaryTarget.model;
  }
}
