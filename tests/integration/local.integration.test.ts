import { describe, expect, it } from 'vitest';
import { OpenAICompatibleAdapter } from '../../src/infrastructure/llm/adapters/openai-compatible.adapter.js';
import { ModelRegistry } from '../../src/infrastructure/llm/configuration/model-registry.js';
import { TokenizerService } from '../../src/infrastructure/llm/tokenization/tokenizer.service.js';
import { ConsoleLogger } from '../../src/infrastructure/observability/logger.js';
import { createGenerationConfig } from '../../src/domain/models/generation-config.js';
import { userMessage } from '../../src/domain/models/message.js';

/**
 * Runs only when LOCAL_LLM_INTEGRATION_TEST=1 is set, since a local
 * OpenAI-compatible server (Ollama/vLLM/LM Studio) has to actually be
 * running at LOCAL_LLM_BASE_URL for this to succeed.
 */
const shouldRun = process.env.LOCAL_LLM_INTEGRATION_TEST === '1';

describe.skipIf(!shouldRun)('Local OpenAI-compatible integration', () => {
  it('completes a minimal live chat request against a locally running server', async () => {
    const adapter = new OpenAICompatibleAdapter(
      {
        baseUrl: process.env.LOCAL_LLM_BASE_URL || 'http://localhost:11434/v1',
        apiKey: process.env.LOCAL_LLM_API_KEY,
        model: process.env.LOCAL_LLM_MODEL || 'llama3',
      },
      new ModelRegistry(),
      new TokenizerService(),
      new ConsoleLogger('warn'),
    );

    const response = await adapter.generate(
      [userMessage('Reply with exactly the word: pong')],
      process.env.LOCAL_LLM_MODEL || 'llama3',
      createGenerationConfig({ maxTokens: 10, temperature: 0 }),
    );

    expect(response.content.length).toBeGreaterThan(0);
  });
});
