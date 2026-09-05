import type { MessageList } from '../../domain/models/message.js';
import type { GenerationConfigInput } from '../../domain/models/generation-config.js';
import type { ModelResponse } from '../../domain/models/model-response.js';
import type { StreamChunk } from '../../domain/models/stream-chunk.js';
import type { GenerateOptions } from '../../infrastructure/llm/adapters/base-llm.adapter.js';

/**
 * The only LLM-related type application/business code should depend on.
 * No provider identity, SDK type, or provider-specific parameter ever
 * appears in this contract.
 */
export interface ILLMService {
  generate(messages: MessageList, config?: GenerationConfigInput, options?: GenerateOptions): Promise<ModelResponse>;
  generateStream(messages: MessageList, config?: GenerationConfigInput, options?: GenerateOptions): AsyncIterable<StreamChunk>;
  countTokens(messages: MessageList): Promise<number>;
}
