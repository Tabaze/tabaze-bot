import { describe, expect, it } from 'vitest';
import { CircuitBreaker, CircuitState, ProviderHealthRegistry } from '../../src/infrastructure/llm/routing/provider-health.js';
import { LLMProvider } from '../../src/domain/enums/provider.enum.js';

const CONFIG = { failureThreshold: 3, cooldownMs: 100, halfOpenMaxAttempts: 1 };

describe('CircuitBreaker', () => {
  it('starts closed and allows attempts', () => {
    const breaker = new CircuitBreaker(CONFIG);
    expect(breaker.getState()).toBe(CircuitState.Closed);
    expect(breaker.canAttempt()).toBe(true);
  });

  it('stays closed if failures do not reach the threshold', () => {
    const breaker = new CircuitBreaker(CONFIG);
    breaker.recordFailure();
    breaker.recordFailure();
    expect(breaker.getState()).toBe(CircuitState.Closed);
    expect(breaker.canAttempt()).toBe(true);
  });

  it('trips open once consecutive failures reach the threshold', () => {
    const breaker = new CircuitBreaker(CONFIG);
    breaker.recordFailure();
    breaker.recordFailure();
    breaker.recordFailure();
    expect(breaker.getState()).toBe(CircuitState.Open);
    expect(breaker.canAttempt()).toBe(false);
  });

  it('a success resets the consecutive failure count', () => {
    const breaker = new CircuitBreaker(CONFIG);
    breaker.recordFailure();
    breaker.recordFailure();
    breaker.recordSuccess();
    breaker.recordFailure();
    breaker.recordFailure();
    expect(breaker.getState()).toBe(CircuitState.Closed);
  });

  it('moves to half-open after the cooldown and allows a probe attempt', () => {
    const breaker = new CircuitBreaker(CONFIG);
    const now = 1_000_000;
    breaker.recordFailure(now);
    breaker.recordFailure(now);
    breaker.recordFailure(now);
    expect(breaker.canAttempt(now + 50)).toBe(false);
    expect(breaker.canAttempt(now + CONFIG.cooldownMs + 1)).toBe(true);
    expect(breaker.getState()).toBe(CircuitState.HalfOpen);
  });

  it('closes again after a successful half-open probe', () => {
    const breaker = new CircuitBreaker(CONFIG);
    const now = 1_000_000;
    breaker.recordFailure(now);
    breaker.recordFailure(now);
    breaker.recordFailure(now);
    breaker.canAttempt(now + CONFIG.cooldownMs + 1);
    breaker.recordAttemptStarted();
    breaker.recordSuccess();
    expect(breaker.getState()).toBe(CircuitState.Closed);
    expect(breaker.canAttempt()).toBe(true);
  });

  it('re-opens if the half-open probe fails', () => {
    const breaker = new CircuitBreaker(CONFIG);
    const now = 1_000_000;
    breaker.recordFailure(now);
    breaker.recordFailure(now);
    breaker.recordFailure(now);
    breaker.canAttempt(now + CONFIG.cooldownMs + 1);
    breaker.recordAttemptStarted();
    breaker.recordFailure(now + CONFIG.cooldownMs + 1);
    expect(breaker.getState()).toBe(CircuitState.Open);
    expect(breaker.canAttempt(now + CONFIG.cooldownMs + 2)).toBe(false);
  });

  it('limits half-open probes to halfOpenMaxAttempts', () => {
    const breaker = new CircuitBreaker({ ...CONFIG, halfOpenMaxAttempts: 1 });
    const now = 1_000_000;
    breaker.recordFailure(now);
    breaker.recordFailure(now);
    breaker.recordFailure(now);
    const probeTime = now + CONFIG.cooldownMs + 1;
    expect(breaker.canAttempt(probeTime)).toBe(true);
    breaker.recordAttemptStarted();
    expect(breaker.canAttempt(probeTime)).toBe(false);
  });
});

describe('ProviderHealthRegistry', () => {
  it('tracks independent circuit state per provider/model pair', () => {
    const registry = new ProviderHealthRegistry(CONFIG);
    const gptBreaker = registry.get(LLMProvider.OpenAI, 'gpt-4o-mini');
    const claudeBreaker = registry.get(LLMProvider.Anthropic, 'claude-3-5-sonnet-20241022');

    gptBreaker.recordFailure();
    gptBreaker.recordFailure();
    gptBreaker.recordFailure();

    expect(gptBreaker.getState()).toBe(CircuitState.Open);
    expect(claudeBreaker.getState()).toBe(CircuitState.Closed);
  });

  it('returns the same breaker instance for repeated lookups of the same pair', () => {
    const registry = new ProviderHealthRegistry(CONFIG);
    expect(registry.get(LLMProvider.OpenAI, 'gpt-4o-mini')).toBe(registry.get(LLMProvider.OpenAI, 'gpt-4o-mini'));
  });
});
