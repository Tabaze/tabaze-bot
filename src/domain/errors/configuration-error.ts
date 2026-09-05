/**
 * Raised for invalid configuration discovered at startup or when building a
 * generation request. Distinct from LLMError because it never originates
 * from a provider response and must never be retried or fallen back on.
 */
export class ConfigurationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ConfigurationError';
  }
}
