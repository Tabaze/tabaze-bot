import { LLMProvider } from '../enums/provider.enum.js';
import { FinishReason } from '../enums/finish-reason.enum.js';

export interface TokenUsage {
  readonly promptTokens: number;
  readonly completionTokens: number;
  readonly totalTokens: number;
}

/**
 * The unified response shape returned by every adapter. Application code
 * depends only on this type, never on a provider SDK's response type.
 */
export interface ModelResponse {
  readonly id: string;
  readonly provider: LLMProvider;
  readonly model: string;
  readonly content: string;
  readonly usage: TokenUsage;
  readonly finishReason: FinishReason;
}
