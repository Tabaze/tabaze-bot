import { describe, expect, it } from 'vitest';
import { GeminiAdapter } from '../../src/infrastructure/llm/adapters/gemini.adapter.js';
import { ModelRegistry } from '../../src/infrastructure/llm/configuration/model-registry.js';
import { TokenizerService } from '../../src/infrastructure/llm/tokenization/tokenizer.service.js';
import { ConsoleLogger } from '../../src/infrastructure/observability/logger.js';
import { createGenerationConfig } from '../../src/domain/models/generation-config.js';
import { userMessage } from '../../src/domain/models/message.js';

const hasCredentials = Boolean(process.env.GEMINI_API_KEY);

describe.skipIf(!hasCredentials)('Gemini integration', () => {
  it('completes a minimal live generateContent request', async () => {
    const adapter = new GeminiAdapter(
      {
        apiKey: process.env.GEMINI_API_KEY!,
        baseUrl: process.env.GEMINI_BASE_URL || 'https://generativelanguage.googleapis.com/v1beta',
      },
      new ModelRegistry(),
      new TokenizerService(),
      new ConsoleLogger('warn'),
    );

    const response = await adapter.generate(
      [userMessage('Reply with exactly the word: pong')],
      process.env.GEMINI_TEST_MODEL || 'gemini-1.5-flash',
      createGenerationConfig({ maxTokens: 10, temperature: 0 }),
    );

    expect(response.content.length).toBeGreaterThan(0);
  });
});
