/**
 * Raised when a streaming response fails *after* at least one delta has
 * already been emitted to the caller. At that point falling back and
 * concatenating a fresh response would duplicate or contradict what the
 * caller already received, so the stream is terminated instead of retried
 * or failed over. `emittedContent` lets the caller decide how to recover
 * (e.g. show a "response interrupted" notice) without guessing.
 */
export class StreamInterruptedError extends Error {
  readonly emittedContent: string;

  constructor(message: string, emittedContent: string, options?: { cause?: unknown }) {
    super(message, options?.cause !== undefined ? { cause: options.cause } : undefined);
    this.name = 'StreamInterruptedError';
    this.emittedContent = emittedContent;
  }
}
