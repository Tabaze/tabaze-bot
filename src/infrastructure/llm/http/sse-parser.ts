import { CancelledError } from '../../../domain/errors/llm-error.js';
import { LLMProvider } from '../../../domain/enums/provider.enum.js';

export interface SseEvent {
  readonly event?: string;
  readonly data: string;
}

function parseEventBlock(block: string): SseEvent | null {
  const lines = block.split('\n');
  let event: string | undefined;
  const dataLines: string[] = [];
  for (const line of lines) {
    if (line.startsWith('event:')) {
      event = line.slice('event:'.length).trim();
    } else if (line.startsWith('data:')) {
      dataLines.push(line.slice('data:'.length).trimStart());
    }
  }
  if (dataLines.length === 0) return null;
  return { event, data: dataLines.join('\n') };
}

/**
 * Parses a `text/event-stream` HTTP body into discrete SSE events. Shared
 * across every adapter's streaming path so buffering/decoding bugs are
 * fixed in one place rather than four.
 */
export async function* parseSseStream(
  body: ReadableStream<Uint8Array>,
  provider: LLMProvider,
  signal?: AbortSignal,
): AsyncGenerator<SseEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';

  const abort = async (): Promise<never> => {
    await reader.cancel().catch(() => undefined);
    throw new CancelledError('Stream was cancelled by the caller.', { provider });
  };

  try {
    while (true) {
      if (signal?.aborted) {
        await abort();
      }

      const { done, value } = await reader.read();
      if (done) {
        if (buffer.trim().length > 0) {
          const event = parseEventBlock(buffer);
          if (event) yield event;
        }
        break;
      }

      buffer += decoder.decode(value, { stream: true });
      buffer = buffer.replace(/\r\n/g, '\n');

      let separatorIndex: number;
      while ((separatorIndex = buffer.indexOf('\n\n')) !== -1) {
        const block = buffer.slice(0, separatorIndex);
        buffer = buffer.slice(separatorIndex + 2);
        const event = parseEventBlock(block);
        if (event) yield event;
      }
    }
  } finally {
    reader.releaseLock();
  }
}
