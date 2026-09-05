import { ConfigurationError } from '../errors/configuration-error.js';

/**
 * Provider-agnostic generation parameters. Adapters translate this into the
 * provider-specific request shape; they never accept provider-specific
 * parameter objects from the application layer.
 */
export interface GenerationConfig {
  readonly temperature?: number;
  readonly maxTokens?: number;
  readonly topP?: number;
  readonly presencePenalty?: number;
  readonly frequencyPenalty?: number;
  readonly stopSequences?: readonly string[];
  readonly timeoutMs: number;
  readonly maxRetries: number;
  readonly stream: boolean;
}

export const DEFAULT_TIMEOUT_MS = 60_000;
export const DEFAULT_MAX_RETRIES = 2;
export const DEFAULT_STREAM = false;

export type GenerationConfigInput = Partial<GenerationConfig>;

function assertRange(name: string, value: number | undefined, min: number, max: number): void {
  if (value === undefined) return;
  if (Number.isNaN(value) || value < min || value > max) {
    throw new ConfigurationError(
      `Invalid GenerationConfig.${name}: ${value}. Expected a number between ${min} and ${max}.`,
    );
  }
}

/**
 * Builds a fully validated, defaulted GenerationConfig from partial user
 * input. Throws ConfigurationError on out-of-range or malformed values so
 * invalid requests never reach a provider adapter.
 */
export function createGenerationConfig(input: GenerationConfigInput = {}): GenerationConfig {
  assertRange('temperature', input.temperature, 0, 2);
  assertRange('topP', input.topP, 0, 1);
  assertRange('presencePenalty', input.presencePenalty, -2, 2);
  assertRange('frequencyPenalty', input.frequencyPenalty, -2, 2);

  if (input.maxTokens !== undefined && (!Number.isInteger(input.maxTokens) || input.maxTokens <= 0)) {
    throw new ConfigurationError(`Invalid GenerationConfig.maxTokens: ${input.maxTokens}. Expected a positive integer.`);
  }

  if (input.timeoutMs !== undefined && (!Number.isFinite(input.timeoutMs) || input.timeoutMs <= 0)) {
    throw new ConfigurationError(`Invalid GenerationConfig.timeoutMs: ${input.timeoutMs}. Expected a positive number.`);
  }

  if (input.maxRetries !== undefined && (!Number.isInteger(input.maxRetries) || input.maxRetries < 0)) {
    throw new ConfigurationError(`Invalid GenerationConfig.maxRetries: ${input.maxRetries}. Expected a non-negative integer.`);
  }

  if (input.stopSequences !== undefined) {
    if (!Array.isArray(input.stopSequences) || input.stopSequences.some((s) => typeof s !== 'string')) {
      throw new ConfigurationError('Invalid GenerationConfig.stopSequences: expected an array of strings.');
    }
    if (input.stopSequences.length > 4) {
      throw new ConfigurationError('Invalid GenerationConfig.stopSequences: providers commonly cap this at 4 sequences.');
    }
  }

  return {
    temperature: input.temperature,
    maxTokens: input.maxTokens,
    topP: input.topP,
    presencePenalty: input.presencePenalty,
    frequencyPenalty: input.frequencyPenalty,
    stopSequences: input.stopSequences,
    timeoutMs: input.timeoutMs ?? DEFAULT_TIMEOUT_MS,
    maxRetries: input.maxRetries ?? DEFAULT_MAX_RETRIES,
    stream: input.stream ?? DEFAULT_STREAM,
  };
}
