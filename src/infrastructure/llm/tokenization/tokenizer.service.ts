import { encode } from 'gpt-tokenizer';
import { LLMProvider } from '../../../domain/enums/provider.enum.js';
import type { MessageList } from '../../../domain/models/message.js';

export interface TokenCountResult {
  /** Normalized token count. */
  readonly count: number;
  /**
   * `true` only when the count comes from a real BPE tokenizer for the
   * provider's model family. `false` means it is a character-based
   * estimate -- callers must not present it to users as exact.
   */
  readonly exact: boolean;
}

/** Chat-completion framing overhead per message (role/name delimiters), per OpenAI's documented counting recipe. */
const PER_MESSAGE_OVERHEAD_TOKENS = 4;
const REPLY_PRIMING_TOKENS = 3;
const CHARS_PER_TOKEN_ESTIMATE = 4;

/**
 * Tokenization is inherently provider/model dependent. This service uses a
 * real tokenizer where one is reliably known to apply (OpenAI's cl100k
 * family, via `gpt-tokenizer`) and otherwise falls back to a clearly
 * labeled character-based estimate -- it never reports an estimate as
 * exact. This is the only place tokenization logic lives; adapters and the
 * application layer just call `count()`.
 */
export class TokenizerService {
  count(messages: MessageList, provider: LLMProvider): TokenCountResult {
    if (provider === LLMProvider.OpenAI) {
      return this.countWithBpeTokenizer(messages);
    }
    return this.estimate(messages);
  }

  private countWithBpeTokenizer(messages: MessageList): TokenCountResult {
    let total = REPLY_PRIMING_TOKENS;
    for (const message of messages) {
      total += PER_MESSAGE_OVERHEAD_TOKENS;
      total += encode(message.content).length;
    }
    return { count: total, exact: true };
  }

  private estimate(messages: MessageList): TokenCountResult {
    const totalChars = messages.reduce((sum, message) => sum + message.content.length, 0);
    const count = Math.ceil(totalChars / CHARS_PER_TOKEN_ESTIMATE) + messages.length * PER_MESSAGE_OVERHEAD_TOKENS;
    return { count, exact: false };
  }
}
