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

function required(env: EnvSource, key: string, context: string): string {
  const value = env[key];
  if (value === undefined || value.trim() === '') {
    throw new ConfigurationError(`Missing required environment variable "${key}" (${context}).`);
  }
  return value;
}

function optionalInt(env: EnvSource, key: string, fallback: number): number {
  const raw = env[key];
  if (raw === undefined || raw.trim() === '') return fallback;
  const parsed = Number(raw);
  if (!Number.isFinite(parsed)) {
    throw new ConfigurationError(`Environment variable "${key}" must be a number, got "${raw}".`);
  }
  return parsed;
}

function parseProvider(raw: string, context: string): LLMProvider {
  const normalized = raw.trim().toLowerCase();
  if (!isLLMProvider(normalized)) {
    throw new ConfigurationError(
      `Invalid provider "${raw}" (${context}). Expected one of: ${Object.values(LLMProvider).join(', ')}.`,
    );
  }
  return normalized;
}

function buildOpenAIConfig(env: EnvSource): OpenAIProviderConfig {
  return {
    apiKey: required(env, 'OPENAI_API_KEY', 'required when LLM_PROVIDER or a routing slot is "openai"'),
    baseUrl: env.OPENAI_BASE_URL?.trim() || 'https://api.openai.com/v1',
  };
}

function buildAnthropicConfig(env: EnvSource): AnthropicProviderConfig {
  return {
    apiKey: required(env, 'ANTHROPIC_API_KEY', 'required when LLM_PROVIDER or a routing slot is "anthropic"'),
    baseUrl: env.ANTHROPIC_BASE_URL?.trim() || 'https://api.anthropic.com/v1',
  };
}

function buildGeminiConfig(env: EnvSource): GeminiProviderConfig {
  return {
    apiKey: required(env, 'GEMINI_API_KEY', 'required when LLM_PROVIDER or a routing slot is "gemini"'),
    baseUrl: env.GEMINI_BASE_URL?.trim() || 'https://generativelanguage.googleapis.com/v1beta',
  };
}

function buildLocalConfig(env: EnvSource): LocalProviderConfig {
  return {
    baseUrl: required(env, 'LOCAL_LLM_BASE_URL', 'required when LLM_PROVIDER or a routing slot is "local"'),
    apiKey: env.LOCAL_LLM_API_KEY?.trim() || undefined,
    model: required(env, 'LOCAL_LLM_MODEL', 'required when LLM_PROVIDER or a routing slot is "local"'),
  };
}

/** Builds the provider config for exactly the providers actually referenced by routing. */
function buildProviderConfigs(env: EnvSource, providersInUse: ReadonlySet<LLMProvider>) {
  return {
    openai: providersInUse.has(LLMProvider.OpenAI) ? buildOpenAIConfig(env) : undefined,
    anthropic: providersInUse.has(LLMProvider.Anthropic) ? buildAnthropicConfig(env) : undefined,
    gemini: providersInUse.has(LLMProvider.Gemini) ? buildGeminiConfig(env) : undefined,
    local: providersInUse.has(LLMProvider.Local) ? buildLocalConfig(env) : undefined,
  };
}

/**
 * Loads and validates the complete LLM configuration from environment
 * variables. Fails fast (ConfigurationError) at startup rather than on the
 * first user request, per the "no lazy config discovery" requirement.
 */
export function loadLLMConfig(env: EnvSource = process.env): LLMConfig {
  const provider = parseProvider(required(env, 'LLM_PROVIDER', 'selects the active provider'), 'LLM_PROVIDER');
  const activeModel = required(env, 'ACTIVE_MODEL', 'the model used with LLM_PROVIDER');

  const fallbackProviderRaw = env.FALLBACK_PROVIDER?.trim();
  const fallbackModelRaw = env.FALLBACK_MODEL?.trim();
  if ((fallbackProviderRaw && !fallbackModelRaw) || (!fallbackProviderRaw && fallbackModelRaw)) {
    throw new ConfigurationError('FALLBACK_PROVIDER and FALLBACK_MODEL must be set together, or not at all.');
  }

  const routing: RoutingConfig = {
    primaryProvider: parseProvider(env.PRIMARY_PROVIDER?.trim() || provider, 'PRIMARY_PROVIDER'),
    primaryModel: env.PRIMARY_MODEL?.trim() || activeModel,
    fallbackProvider: fallbackProviderRaw ? parseProvider(fallbackProviderRaw, 'FALLBACK_PROVIDER') : undefined,
    fallbackModel: fallbackModelRaw,
  };

  const providersInUse = new Set<LLMProvider>([provider, routing.primaryProvider]);
  if (routing.fallbackProvider) providersInUse.add(routing.fallbackProvider);

  const providerConfigs = buildProviderConfigs(env, providersInUse);

  return {
    provider,
    activeModel,
    ...providerConfigs,
    routing,
    retry: {
      maxRetries: optionalInt(env, 'LLM_RETRY_MAX_ATTEMPTS', 2),
      baseDelayMs: optionalInt(env, 'LLM_RETRY_BASE_DELAY_MS', 250),
      maxDelayMs: optionalInt(env, 'LLM_RETRY_MAX_DELAY_MS', 8_000),
    },
    circuitBreaker: {
      failureThreshold: optionalInt(env, 'LLM_CIRCUIT_BREAKER_FAILURE_THRESHOLD', 5),
      cooldownMs: optionalInt(env, 'LLM_CIRCUIT_BREAKER_COOLDOWN_MS', 30_000),
      halfOpenMaxAttempts: optionalInt(env, 'LLM_CIRCUIT_BREAKER_HALF_OPEN_MAX_ATTEMPTS', 1),
    },
    defaultTimeoutMs: optionalInt(env, 'LLM_DEFAULT_TIMEOUT_MS', 60_000),
  };
}
