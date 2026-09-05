import { describe, expect, it } from 'vitest';
import { OpenAIAdapter } from '../../src/infrastructure/llm/adapters/openai.adapter.js';
import { ModelRegistry } from '../../src/infrastructure/llm/configuration/model-registry.js';
import { TokenizerService } from '../../src/infrastructure/llm/tokenization/tokenizer.service.js';
import { ConsoleLogger } from '../../src/infrastructure/observability/logger.js';
import { createGenerationConfig } from '../../src/domain/models/generation-config.js';
import { userMessage } from '../../src/domain/models/message.js';

/**
 * Runs only when OPENAI_API_KEY is present in the environment. Never
 * required for `npm test` -- only for `npm run test:integration` with
 * real credentials exported first.
 */
const hasCredentials = Boolean(process.env.OPENAI_API_KEY);

describe.skipIf(!hasCredentials)('OpenAI integration', () => {
  it('completes a minimal live chat request', async () => {
    const adapter = new OpenAIAdapter(
      { apiKey: process.env.OPENAI_API_KEY!, baseUrl: process.env.OPENAI_BASE_URL || 'https://api.openai.com/v1' },
      new ModelRegistry(),
      new TokenizerService(),
      new ConsoleLogger('warn'),
    );

    const response = await adapter.generate(
      [userMessage('Reply with exactly the word: pong')],
      process.env.ACTIVE_MODEL || 'gpt-4o-mini',
      createGenerationConfig({ maxTokens: 10, temperature: 0 }),
    );

    expect(response.content.length).toBeGreaterThan(0);
    expect(response.usage.totalTokens).toBeGreaterThan(0);
  });
});
