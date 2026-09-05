import { buildCompositionRoot } from './config/composition-root.js';
import { systemMessage, userMessage, type MessageList } from './domain/models/message.js';
import { LLMError } from './domain/errors/llm-error.js';
import { StreamInterruptedError } from './domain/errors/stream-interrupted-error.js';

/**
 * Example application code. Note that nothing below this line imports
 * from `infrastructure/llm/adapters/*`, an OpenAI/Anthropic/Gemini SDK, or
 * `LLMProvider` -- it only knows about `ILLMService` and domain models.
 */
async function main(): Promise<void> {
  const { llmService, logger } = buildCompositionRoot();

  const messages: MessageList = [
    systemMessage('You are a concise, helpful assistant.'),
    userMessage('In one sentence, what is a multi-LLM abstraction layer?'),
  ];

  try {
    const promptTokens = await llmService.countTokens(messages);
    logger.info('Prompt token count', { promptTokens });

    const response = await llmService.generate(messages, { temperature: 0.7, maxTokens: 200 });
    logger.info('Generation complete', {
      provider: response.provider,
      model: response.model,
      finishReason: response.finishReason,
      usage: response.usage,
    });
    console.log(response.content);

    console.log('\n--- streaming the same request ---\n');
    for await (const chunk of llmService.generateStream(messages, { temperature: 0.7, maxTokens: 200 })) {
      process.stdout.write(chunk.delta);
      if (chunk.finishReason) {
        console.log(`\n[finished: ${chunk.finishReason}]`);
      }
    }
  } catch (error) {
    if (error instanceof StreamInterruptedError) {
      logger.error('Stream was interrupted after partial output', { emittedContent: error.emittedContent });
      return;
    }
    if (error instanceof LLMError) {
      logger.error('LLM request failed', { category: error.category, provider: error.provider, message: error.message });
      return;
    }
    throw error;
  }
}

main().catch((error: unknown) => {
  console.error('Fatal error:', error);
  process.exitCode = 1;
});
