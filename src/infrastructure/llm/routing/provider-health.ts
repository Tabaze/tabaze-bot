import { LLMProvider } from '../../../domain/enums/provider.enum.js';
import type { CircuitBreakerConfig } from '../configuration/llm.config.js';

export enum CircuitState {
  Closed = 'CLOSED',
  Open = 'OPEN',
  HalfOpen = 'HALF_OPEN',
}

/**
 * Per provider/model circuit breaker. CLOSED allows all traffic; after
 * `failureThreshold` consecutive failures it trips OPEN and rejects
 * attempts until `cooldownMs` elapses, then allows a bounded number of
 * HALF_OPEN probe attempts -- a success closes the circuit again, a
 * failure re-opens it.
 */
export class CircuitBreaker {
  private state: CircuitState = CircuitState.Closed;
  private consecutiveFailures = 0;
  private openedAt = 0;
  private halfOpenAttempts = 0;

  constructor(private readonly config: CircuitBreakerConfig) {}

  getState(): CircuitState {
    return this.state;
  }

  canAttempt(now: number = Date.now()): boolean {
    if (this.state === CircuitState.Closed) return true;

    if (this.state === CircuitState.Open) {
      if (now - this.openedAt < this.config.cooldownMs) return false;
      this.state = CircuitState.HalfOpen;
      this.halfOpenAttempts = 0;
    }

    return this.halfOpenAttempts < this.config.halfOpenMaxAttempts;
  }

  recordAttemptStarted(): void {
    if (this.state === CircuitState.HalfOpen) this.halfOpenAttempts += 1;
  }

  recordSuccess(): void {
    this.state = CircuitState.Closed;
    this.consecutiveFailures = 0;
    this.halfOpenAttempts = 0;
  }

  recordFailure(now: number = Date.now()): void {
    if (this.state === CircuitState.HalfOpen) {
      this.trip(now);
      return;
    }
    this.consecutiveFailures += 1;
    if (this.consecutiveFailures >= this.config.failureThreshold) {
      this.trip(now);
    }
  }

  private trip(now: number): void {
    this.state = CircuitState.Open;
    this.openedAt = now;
    this.consecutiveFailures = this.config.failureThreshold;
    this.halfOpenAttempts = 0;
  }
}

/** Owns one CircuitBreaker per provider/model pair so a failing model doesn't trip an unrelated one on the same provider. */
export class ProviderHealthRegistry {
  private readonly breakers = new Map<string, CircuitBreaker>();

  constructor(private readonly config: CircuitBreakerConfig) {}

  private key(provider: LLMProvider, model: string): string {
    return `${provider}::${model}`;
  }

  get(provider: LLMProvider, model: string): CircuitBreaker {
    const key = this.key(provider, model);
    let breaker = this.breakers.get(key);
    if (!breaker) {
      breaker = new CircuitBreaker(this.config);
      this.breakers.set(key, breaker);
    }
    return breaker;
  }
}
