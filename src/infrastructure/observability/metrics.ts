/**
 * Vendor-neutral metrics port. The default implementation is an in-memory
 * recorder suitable for tests and small deployments; swap it for a
 * Prometheus/OpenTelemetry-backed implementation in production without
 * touching call sites.
 */
export interface MetricLabels {
  readonly [key: string]: string;
}

export interface IMetricsRecorder {
  incrementCounter(name: string, labels?: MetricLabels, value?: number): void;
  recordHistogram(name: string, valueMs: number, labels?: MetricLabels): void;
}

export const METRIC_NAMES = {
  REQUESTS_TOTAL: 'llm_requests_total',
  REQUESTS_FAILED: 'llm_requests_failed',
  REQUEST_LATENCY: 'llm_request_latency',
  FALLBACK_TOTAL: 'llm_fallback_total',
  TOKENS_PROMPT: 'llm_tokens_prompt',
  TOKENS_COMPLETION: 'llm_tokens_completion',
} as const;

interface HistogramStats {
  count: number;
  sum: number;
  min: number;
  max: number;
}

function labelKey(name: string, labels?: MetricLabels): string {
  if (!labels) return name;
  const sorted = Object.keys(labels).sort().map((k) => `${k}=${labels[k]}`);
  return `${name}{${sorted.join(',')}}`;
}

export class InMemoryMetricsRecorder implements IMetricsRecorder {
  private readonly counters = new Map<string, number>();
  private readonly histograms = new Map<string, HistogramStats>();

  incrementCounter(name: string, labels?: MetricLabels, value = 1): void {
    const key = labelKey(name, labels);
    this.counters.set(key, (this.counters.get(key) ?? 0) + value);
  }

  recordHistogram(name: string, valueMs: number, labels?: MetricLabels): void {
    const key = labelKey(name, labels);
    const existing = this.histograms.get(key);
    if (existing) {
      existing.count += 1;
      existing.sum += valueMs;
      existing.min = Math.min(existing.min, valueMs);
      existing.max = Math.max(existing.max, valueMs);
    } else {
      this.histograms.set(key, { count: 1, sum: valueMs, min: valueMs, max: valueMs });
    }
  }

  getCounter(name: string, labels?: MetricLabels): number {
    return this.counters.get(labelKey(name, labels)) ?? 0;
  }

  getHistogram(name: string, labels?: MetricLabels): HistogramStats | undefined {
    return this.histograms.get(labelKey(name, labels));
  }
}

export class NullMetricsRecorder implements IMetricsRecorder {
  incrementCounter(): void {}
  recordHistogram(): void {}
}
