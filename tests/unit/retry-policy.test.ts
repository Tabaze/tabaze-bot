import { describe, expect, it, vi } from 'vitest';
import { RetryPolicy, classifyForRetry, canFallback, sleep } from '../../src/infrastructure/llm/routing/retry-policy.js';
import { CancelledError, InvalidRequestError, RateLimitError, TimeoutError } from '../../src/domain/errors/llm-error.js';

const CONFIG = { maxRetries: 2, baseDelayMs: 1, maxDelayMs: 5 };

describe('classifyForRetry / canFallback', () => {
  it('marks rate limit and timeout errors as retryable and fallbackable', () => {
    expect(classifyForRetry(new RateLimitError('x')).retryable).toBe(true);
    expect(canFallback(new TimeoutError('x'))).toBe(true);
  });

  it('never retries or falls back on an invalid request', () => {
    const error = new InvalidRequestError('bad request');
    expect(classifyForRetry(error).retryable).toBe(false);
    expect(canFallback(error)).toBe(false);
  });

  it('never retries or falls back on cancellation', () => {
    const error = new CancelledError('cancelled');
    expect(classifyForRetry(error).retryable).toBe(false);
    expect(canFallback(error)).toBe(false);
  });

  it('treats non-LLMError throwables as non-retryable', () => {
    expect(classifyForRetry(new Error('boom')).retryable).toBe(false);
  });
});

describe('RetryPolicy', () => {
  it('returns the result immediately on first success without retrying', async () => {
    const policy = new RetryPolicy(CONFIG);
    const fn = vi.fn().mockResolvedValue('ok');
    await expect(policy.execute(fn)).resolves.toBe('ok');
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it('retries a retryable error up to maxRetries then succeeds', async () => {
    const policy = new RetryPolicy(CONFIG);
    const fn = vi
      .fn()
      .mockRejectedValueOnce(new TimeoutError('t1'))
      .mockRejectedValueOnce(new TimeoutError('t2'))
      .mockResolvedValue('ok');
    await expect(policy.execute(fn)).resolves.toBe('ok');
    expect(fn).toHaveBeenCalledTimes(3);
  });

  it('stops retrying once the retry budget is exhausted and throws the last error', async () => {
    const policy = new RetryPolicy(CONFIG);
    const error = new TimeoutError('always fails');
    const fn = vi.fn().mockRejectedValue(error);
    await expect(policy.execute(fn)).rejects.toBe(error);
    expect(fn).toHaveBeenCalledTimes(CONFIG.maxRetries + 1);
  });

  it('never retries a non-retryable error, even once', async () => {
    const policy = new RetryPolicy(CONFIG);
    const error = new InvalidRequestError('bad request');
    const fn = vi.fn().mockRejectedValue(error);
    await expect(policy.execute(fn)).rejects.toBe(error);
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it('honors a rate-limit Retry-After hint when computing backoff', () => {
    const policy = new RetryPolicy({ maxRetries: 1, baseDelayMs: 1000, maxDelayMs: 10_000 });
    expect(policy.computeDelayMs(0, 250)).toBe(250);
  });

  it('caps the computed delay at maxDelayMs', () => {
    const policy = new RetryPolicy({ maxRetries: 5, baseDelayMs: 1000, maxDelayMs: 2000 }, () => 1);
    expect(policy.computeDelayMs(10)).toBeLessThanOrEqual(2000);
  });
});

describe('sleep', () => {
  it('resolves after the given duration', async () => {
    await expect(sleep(1)).resolves.toBeUndefined();
  });

  it('rejects immediately with CancelledError when the signal is already aborted', async () => {
    const controller = new AbortController();
    controller.abort();
    await expect(sleep(1000, controller.signal)).rejects.toBeInstanceOf(CancelledError);
  });

  it('rejects with CancelledError if the signal aborts mid-sleep', async () => {
    const controller = new AbortController();
    const promise = sleep(1000, controller.signal);
    setTimeout(() => controller.abort(), 5);
    await expect(promise).rejects.toBeInstanceOf(CancelledError);
  });
});
