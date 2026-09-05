import { describe, expect, it } from 'vitest';
import { AnthropicAdapter } from '../../src/infrastructure/llm/adapters/anthropic.adapter.js';
import { ModelRegistry } from '../../src/infrastructure/llm/configuration/model-registry.js';
import { TokenizerService } from '../../src/infrastructure/llm/tokenization/tokenizer.service.js';
import { ConsoleLogger } from '../../src/infrastructure/observability/logger.js';
import { createGenerationConfig } from '../../src/domain/models/generation-config.js';
import { userMessage } from '../../src/domain/models/message.js';

const hasCredentials = Boolean(process.env.ANTHROPIC_API_KEY);

describe.skipIf(!hasCredentials)('Anthropic integration', () => {
  it('completes a minimal live message request', async () => {
    const adapter = new AnthropicAdapter(
      { apiKey: process.env.ANTHROPIC_API_KEY!, baseUrl: process.env.ANTHROPIC_BASE_URL || 'https://api.anthropic.com/v1' },
      new ModelRegistry(),
      new TokenizerService(),
      new ConsoleLogger('warn'),
    );

    const response = await adapter.generate(
      [userMessage('Reply with exactly the word: pong')],
      process.env.ANTHROPIC_TEST_MODEL || 'claude-3-5-sonnet-20241022',
      createGenerationConfig({ maxTokens: 10, temperature: 0 }),
    );

    expect(response.content.length).toBeGreaterThan(0);
    expect(response.usage.totalTokens).toBeGreaterThan(0);
  });
});
