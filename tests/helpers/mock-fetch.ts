import { vi } from 'vitest';

export function jsonResponse(status: number, body: unknown, headers: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json', ...headers } });
}

/** Builds a `text/event-stream` Response from raw SSE-formatted event strings (each already ending in "\n\n"). */
export function sseResponse(rawEvents: readonly string[], status = 200): Response {
  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const event of rawEvents) {
        controller.enqueue(encoder.encode(event));
      }
      controller.close();
    },
  });
  return new Response(stream, { status, headers: { 'Content-Type': 'text/event-stream' } });
}

export function sseDataEvent(data: unknown): string {
  return `data: ${typeof data === 'string' ? data : JSON.stringify(data)}\n\n`;
}

export function sseTypedEvent(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

export type MockFetchCall = { url: string; init: RequestInit };

/** Stubs global fetch with a fixed queue of responses, in call order, and records each call for assertions. */
export function queueFetchResponses(responses: readonly Response[]): { fetchMock: ReturnType<typeof vi.fn>; calls: MockFetchCall[] } {
  const calls: MockFetchCall[] = [];
  let index = 0;
  const fetchMock = vi.fn(async (input: unknown, init?: RequestInit) => {
    calls.push({ url: String(input), init: init ?? {} });
    const response = responses[index++];
    if (!response) throw new Error('mock-fetch: no more responses queued');
    return response;
  });
  vi.stubGlobal('fetch', fetchMock);
  return { fetchMock, calls };
}

export function requestBody(call: MockFetchCall): Record<string, unknown> {
  return JSON.parse(String(call.init.body)) as Record<string, unknown>;
}

/** A fetch stub whose response only resolves after `delayMs`, and rejects immediately if the request's AbortSignal fires first -- for exercising timeout/cancellation behavior. */
export function abortAwareFetch(delayMs: number, response: Response): (input: unknown, init?: RequestInit) => Promise<Response> {
  return (_input, init) =>
    new Promise<Response>((resolve, reject) => {
      const signal = init?.signal as AbortSignal | undefined;
      const timer = setTimeout(() => resolve(response), delayMs);
      const onAbort = (): void => {
        clearTimeout(timer);
        reject(new DOMException('The operation was aborted.', 'AbortError'));
      };
      if (signal) {
        if (signal.aborted) {
          onAbort();
        } else {
          signal.addEventListener('abort', onAbort, { once: true });
        }
      }
    });
}
