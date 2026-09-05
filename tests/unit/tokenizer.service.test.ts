import { describe, expect, it } from 'vitest';
import { TokenizerService } from '../../src/infrastructure/llm/tokenization/tokenizer.service.js';
import { LLMProvider } from '../../src/domain/enums/provider.enum.js';
import { userMessage } from '../../src/domain/models/message.js';

describe('TokenizerService', () => {
  const tokenizer = new TokenizerService();

  it('reports an exact BPE-based count for OpenAI', () => {
    const result = tokenizer.count([userMessage('Hello, world!')], LLMProvider.OpenAI);
    expect(result.exact).toBe(true);
    expect(result.count).toBeGreaterThan(0);
  });

  it('reports a labeled estimate for providers without a bundled tokenizer', () => {
    for (const provider of [LLMProvider.Anthropic, LLMProvider.Gemini, LLMProvider.Local]) {
      const result = tokenizer.count([userMessage('Hello, world!')], provider);
      expect(result.exact).toBe(false);
      expect(result.count).toBeGreaterThan(0);
    }
  });

  it('returns a normalized plain number regardless of provider', () => {
    const result = tokenizer.count([userMessage('test')], LLMProvider.Gemini);
    expect(typeof result.count).toBe('number');
    expect(Number.isInteger(result.count)).toBe(true);
  });

  it('scales roughly with message length', () => {
    const short = tokenizer.count([userMessage('hi')], LLMProvider.OpenAI);
    const long = tokenizer.count([userMessage('hi '.repeat(200))], LLMProvider.OpenAI);
    expect(long.count).toBeGreaterThan(short.count);
  });

  it('accounts for every message, not just the last one', () => {
    const single = tokenizer.count([userMessage('hello there')], LLMProvider.OpenAI);
    const multiple = tokenizer.count(
      [userMessage('hello there'), userMessage('hello there'), userMessage('hello there')],
      LLMProvider.OpenAI,
    );
    expect(multiple.count).toBeGreaterThan(single.count * 2);
  });
});
