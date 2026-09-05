import { LLMProvider } from '../enums/provider.enum.js';
import { FinishReason } from '../enums/finish-reason.enum.js';
import type { TokenUsage } from './model-response.js';

/**
 * The unified streaming chunk shape. Raw provider SSE/event payloads never
 * cross this boundary; adapters normalize every chunk into this shape.
 */
export interface StreamChunk {
  readonly id: string;
  readonly provider: LLMProvider;
  readonly model: string;
  readonly delta: string;
  readonly finishReason: FinishReason | null;
  readonly usage: TokenUsage | null;
}
