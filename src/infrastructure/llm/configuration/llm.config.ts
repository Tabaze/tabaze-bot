import { ConfigurationError } from '../../../domain/errors/configuration-error.js';
import { isLLMProvider, LLMProvider } from '../../../domain/enums/provider.enum.js';

export interface OpenAIProviderConfig {
  readonly apiKey: string;
  readonly baseUrl: string;
}

export interface AnthropicProviderConfig {
  readonly apiKey: string;
  readonly baseUrl: string;
}

export interface GeminiProviderConfig {
  readonly apiKey: string;
  readonly baseUrl: string;
}

export interface LocalProviderConfig {
  readonly baseUrl: string;
  readonly apiKey?: string;
  readonly model: string;
}

export interface RoutingConfig {
  readonly primaryProvider: LLMProvider;
  readonly primaryModel: string;
  readonly fallbackProvider?: LLMProvider;
  readonly fallbackModel?: string;
}

export interface RetryConfig {
  readonly maxRetries: number;
  readonly baseDelayMs: number;
  readonly maxDelayMs: number;
}

export interface CircuitBreakerConfig {
  readonly failureThreshold: number;
  readonly cooldownMs: number;
  readonly halfOpenMaxAttempts: number;
}

export interface LLMConfig {
  readonly provider: LLMProvider;
  readonly activeModel: string;
  readonly openai?: OpenAIProviderConfig;
  readonly anthropic?: AnthropicProviderConfig;
  readonly gemini?: GeminiProviderConfig;
  readonly local?: LocalProviderConfig;
  readonly routing: RoutingConfig;
  readonly retry: RetryConfig;
  readonly circuitBreaker: CircuitBreakerConfig;
  readonly defaultTimeoutMs: number;
}

export type EnvSource = Record<string, string | undefined>;

/** Reads a required variable, or records why it's missing and returns ''. */
function required(env: EnvSource, key: string, context: string, problems: string[]): string {
  const value = env[key];
  if (value === undefined || value.trim() === '') {
    problems.push(`"${key}" is missing (${context}).`);
    return '';
  }
  return value;
}

function optionalInt(env: EnvSource, key: string, fallback: number, problems: string[]): number {
  const raw = env[key];
  if (raw === undefined || raw.trim() === '') return fallback;
  const parsed = Number(raw);
  if (!Number.isFinite(parsed)) {
    problems.push(`"${key}" must be a number, got "${raw}".`);
    return fallback;
  }
  return parsed;
}

/** Validates a non-empty provider string, or records why it's invalid. Caller must not call this with an empty string. */
function parseProvider(raw: string, label: string, problems: string[]): LLMProvider {
  const normalized = raw.trim().toLowerCase();
  if (!isLLMProvider(normalized)) {
    problems.push(`"${label}" has an invalid value "${raw}". Expected one of: ${Object.values(LLMProvider).join(', ')}.`);
    return LLMProvider.OpenAI;
  }
  return normalized;
}

function buildOpenAIConfig(env: EnvSource, problems: string[]): OpenAIProviderConfig {
  return {
    apiKey: required(env, 'OPENAI_API_KEY', 'required because provider "openai" is in use', problems),
    baseUrl: env.OPENAI_BASE_URL?.trim() || 'https://api.openai.com/v1',
  };
}

function buildAnthropicConfig(env: EnvSource, problems: string[]): AnthropicProviderConfig {
  return {
    apiKey: required(env, 'ANTHROPIC_API_KEY', 'required because provider "anthropic" is in use', problems),
    baseUrl: env.ANTHROPIC_BASE_URL?.trim() || 'https://api.anthropic.com/v1',
  };
}

function buildGeminiConfig(env: EnvSource, problems: string[]): GeminiProviderConfig {
  return {
    apiKey: required(env, 'GEMINI_API_KEY', 'required because provider "gemini" is in use', problems),
    baseUrl: env.GEMINI_BASE_URL?.trim() || 'https://generativelanguage.googleapis.com/v1beta',
  };
}

function buildLocalConfig(env: EnvSource, problems: string[]): LocalProviderConfig {
  return {
    baseUrl: required(env, 'LOCAL_LLM_BASE_URL', 'required because provider "local" is in use', problems),
    apiKey: env.LOCAL_LLM_API_KEY?.trim() || undefined,
    model: required(env, 'LOCAL_LLM_MODEL', 'required because provider "local" is in use', problems),
  };
}

/** Builds the provider config for exactly the providers actually referenced by routing. */
function buildProviderConfigs(env: EnvSource, providersInUse: ReadonlySet<LLMProvider>, problems: string[]) {
  return {
    openai: providersInUse.has(LLMProvider.OpenAI) ? buildOpenAIConfig(env, problems) : undefined,
    anthropic: providersInUse.has(LLMProvider.Anthropic) ? buildAnthropicConfig(env, problems) : undefined,
    gemini: providersInUse.has(LLMProvider.Gemini) ? buildGeminiConfig(env, problems) : undefined,
    local: providersInUse.has(LLMProvider.Local) ? buildLocalConfig(env, problems) : undefined,
  };
}

function throwIfProblems(problems: readonly string[]): void {
  if (problems.length === 0) return;
  const list = problems.map((problem) => `  - ${problem}`).join('\n');
  throw new ConfigurationError(`Invalid LLM configuration, ${problems.length} problem(s):\n${list}`);
}

/**
 * Loads and validates the complete LLM configuration from environment
 * variables. Fails fast (ConfigurationError) at startup rather than on the
 * first user request, per the "no lazy config discovery" requirement --
 * and reports every missing or invalid setting at once rather than one at
 * a time, so a single run tells the caller everything they need to fix.
 */
export function loadLLMConfig(env: EnvSource = process.env): LLMConfig {
  const shapeProblems: string[] = [];

  const providerRaw = required(env, 'LLM_PROVIDER', 'selects the active provider', shapeProblems);
  const provider = providerRaw ? parseProvider(providerRaw, 'LLM_PROVIDER', shapeProblems) : LLMProvider.OpenAI;
  const activeModel = required(env, 'ACTIVE_MODEL', 'the model used with LLM_PROVIDER', shapeProblems);

  const fallbackProviderRaw = env.FALLBACK_PROVIDER?.trim();
  const fallbackModelRaw = env.FALLBACK_MODEL?.trim();
  if (fallbackProviderRaw && !fallbackModelRaw) {
    shapeProblems.push('"FALLBACK_MODEL" is required because "FALLBACK_PROVIDER" is set.');
  }
  if (!fallbackProviderRaw && fallbackModelRaw) {
    shapeProblems.push('"FALLBACK_PROVIDER" is required because "FALLBACK_MODEL" is set.');
  }

  const primaryProviderRaw = env.PRIMARY_PROVIDER?.trim();

  // Resolve provider/routing shape before checking provider-specific keys below --
  // otherwise a missing/invalid LLM_PROVIDER would cascade into misleading
  // "API key missing" noise for a provider the caller never actually chose.
  throwIfProblems(shapeProblems);

  const routing: RoutingConfig = {
    primaryProvider: primaryProviderRaw ? parseProvider(primaryProviderRaw, 'PRIMARY_PROVIDER', shapeProblems) : provider,
    primaryModel: env.PRIMARY_MODEL?.trim() || activeModel,
    fallbackProvider: fallbackProviderRaw ? parseProvider(fallbackProviderRaw, 'FALLBACK_PROVIDER', shapeProblems) : undefined,
    fallbackModel: fallbackModelRaw,
  };
  throwIfProblems(shapeProblems);

  const providersInUse = new Set<LLMProvider>([provider, routing.primaryProvider]);
  if (routing.fallbackProvider) providersInUse.add(routing.fallbackProvider);

  const problems: string[] = [];
  const providerConfigs = buildProviderConfigs(env, providersInUse, problems);

  const retry: RetryConfig = {
    maxRetries: optionalInt(env, 'LLM_RETRY_MAX_ATTEMPTS', 2, problems),
    baseDelayMs: optionalInt(env, 'LLM_RETRY_BASE_DELAY_MS', 250, problems),
    maxDelayMs: optionalInt(env, 'LLM_RETRY_MAX_DELAY_MS', 8_000, problems),
  };
  const circuitBreaker: CircuitBreakerConfig = {
    failureThreshold: optionalInt(env, 'LLM_CIRCUIT_BREAKER_FAILURE_THRESHOLD', 5, problems),
    cooldownMs: optionalInt(env, 'LLM_CIRCUIT_BREAKER_COOLDOWN_MS', 30_000, problems),
    halfOpenMaxAttempts: optionalInt(env, 'LLM_CIRCUIT_BREAKER_HALF_OPEN_MAX_ATTEMPTS', 1, problems),
  };
  const defaultTimeoutMs = optionalInt(env, 'LLM_DEFAULT_TIMEOUT_MS', 60_000, problems);

  throwIfProblems(problems);

  return {
    provider,
    activeModel,
    ...providerConfigs,
    routing,
    retry,
    circuitBreaker,
    defaultTimeoutMs,
  };
}
