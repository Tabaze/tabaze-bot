import { describe, expect, it } from 'vitest';
import { createGenerationConfig, DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT_MS } from '../../src/domain/models/generation-config.js';
import { ConfigurationError } from '../../src/domain/errors/configuration-error.js';

describe('createGenerationConfig', () => {
  it('applies sensible defaults when no input is given', () => {
    const config = createGenerationConfig();
    expect(config.timeoutMs).toBe(DEFAULT_TIMEOUT_MS);
    expect(config.maxRetries).toBe(DEFAULT_MAX_RETRIES);
    expect(config.stream).toBe(false);
    expect(config.temperature).toBeUndefined();
  });

  it('preserves explicitly provided values', () => {
    const config = createGenerationConfig({ temperature: 0.3, maxTokens: 512, stream: true });
    expect(config.temperature).toBe(0.3);
    expect(config.maxTokens).toBe(512);
    expect(config.stream).toBe(true);
  });

  it.each([
    ['temperature', { temperature: 3 }],
    ['temperature', { temperature: -1 }],
    ['topP', { topP: 1.5 }],
    ['presencePenalty', { presencePenalty: 3 }],
    ['frequencyPenalty', { frequencyPenalty: -3 }],
  ])('rejects out-of-range %s', (_name, input) => {
    expect(() => createGenerationConfig(input)).toThrow(ConfigurationError);
  });

  it('rejects a non-positive maxTokens', () => {
    expect(() => createGenerationConfig({ maxTokens: 0 })).toThrow(ConfigurationError);
    expect(() => createGenerationConfig({ maxTokens: -10 })).toThrow(ConfigurationError);
  });

  it('rejects a non-positive timeoutMs', () => {
    expect(() => createGenerationConfig({ timeoutMs: 0 })).toThrow(ConfigurationError);
  });

  it('rejects a negative maxRetries', () => {
    expect(() => createGenerationConfig({ maxRetries: -1 })).toThrow(ConfigurationError);
  });

  it('rejects more than 4 stop sequences', () => {
    expect(() => createGenerationConfig({ stopSequences: ['a', 'b', 'c', 'd', 'e'] })).toThrow(ConfigurationError);
  });

  it('rejects a non-string entry in stopSequences', () => {
    expect(() => createGenerationConfig({ stopSequences: ['a', 123 as unknown as string] })).toThrow(ConfigurationError);
  });
});
