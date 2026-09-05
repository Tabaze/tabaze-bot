import type { MessageList } from '../../domain/models/message.js';
import { createGenerationConfig, type GenerationConfigInput } from '../../domain/models/generation-config.js';
import type { ModelResponse } from '../../domain/models/model-response.js';
import type { StreamChunk } from '../../domain/models/stream-chunk.js';
import { ContextLengthError } from '../../domain/errors/llm-error.js';
import type { GenerateOptions } from '../../infrastructure/llm/adapters/base-llm.adapter.js';
import { LLMRouter } from '../../infrastructure/llm/routing/llm-router.js';
import type { ILogger } from '../../infrastructure/observability/logger.js';
import type { ILLMService } from '../interfaces/illm-service.interface.js';

/**
 * The application-facing entry point to the whole abstraction. It depends
 * only on LLMRouter (a provider-agnostic strategy) and normalized domain
 * types -- it has no knowledge of OpenAI, Anthropic, Gemini, or local
 * inference servers.
 */
export class LLMService implements ILLMService {
  constructor(
    private readonly router: LLMRouter,
    private readonly logger: ILogger,
  ) {}

  async generate(messages: MessageList, configInput?: GenerationConfigInput, options?: GenerateOptions): Promise<ModelResponse> {
    const config = createGenerationConfig({ ...configInput, stream: false });
    await this.validateContextLength(messages, config.maxTokens);
    return this.router.generate(messages, config, options);
  }

  async *generateStream(
    messages: MessageList,
    configInput?: GenerationConfigInput,
    options?: GenerateOptions,
  ): AsyncIterable<StreamChunk> {
    const config = createGenerationConfig({ ...configInput, stream: true });
    await this.validateContextLength(messages, config.maxTokens);
    yield* this.router.generateStream(messages, config, options);
  }

  async countTokens(messages: MessageList): Promise<number> {
    return this.router.tokenizeAndCount(messages);
  }

  private async validateContextLength(messages: MessageList, reservedCompletionTokens: number | undefined): Promise<void> {
    const promptTokens = await this.router.tokenizeAndCount(messages);
    const reserved = reservedCompletionTokens ?? 0;
    const limit = this.router.getContextLimit();

    if (promptTokens + reserved > limit) {
      this.logger.warn('Rejecting request that would exceed the model context window', {
        promptTokens,
        reservedCompletionTokens: reserved,
        contextLimit: limit,
      });
      throw new ContextLengthError(
        `Prompt (${promptTokens} tokens) plus requested max_tokens (${reserved}) exceeds the model's context limit of ${limit} tokens.`,
      );
    }
  }
}
