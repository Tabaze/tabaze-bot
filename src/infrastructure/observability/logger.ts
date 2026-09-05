/**
 * Vendor-neutral structured logging port. Swap the console implementation
 * for a Pino/Winston/Datadog-backed one without touching call sites.
 */
export interface LogFields {
  readonly [key: string]: unknown;
}

export interface ILogger {
  debug(message: string, fields?: LogFields): void;
  info(message: string, fields?: LogFields): void;
  warn(message: string, fields?: LogFields): void;
  error(message: string, fields?: LogFields): void;
}

const REDACTED = '[REDACTED]';

/** Field name fragments that must never reach a log sink. */
const SENSITIVE_KEY_PATTERN = /api[-_]?key|authorization|secret|token|password/i;

function redact(fields: LogFields): LogFields {
  const safe: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(fields)) {
    safe[key] = SENSITIVE_KEY_PATTERN.test(key) ? REDACTED : value;
  }
  return safe;
}

export class ConsoleLogger implements ILogger {
  constructor(private readonly minLevel: 'debug' | 'info' | 'warn' | 'error' = 'info') {}

  private static readonly LEVEL_ORDER = ['debug', 'info', 'warn', 'error'] as const;

  private shouldLog(level: (typeof ConsoleLogger.LEVEL_ORDER)[number]): boolean {
    return ConsoleLogger.LEVEL_ORDER.indexOf(level) >= ConsoleLogger.LEVEL_ORDER.indexOf(this.minLevel);
  }

  private write(level: 'debug' | 'info' | 'warn' | 'error', message: string, fields?: LogFields): void {
    if (!this.shouldLog(level)) return;
    const payload = {
      timestamp: new Date().toISOString(),
      level,
      message,
      ...(fields ? redact(fields) : {}),
    };
    const line = JSON.stringify(payload);
    if (level === 'error') {
      console.error(line);
    } else if (level === 'warn') {
      console.warn(line);
    } else {
      console.log(line);
    }
  }

  debug(message: string, fields?: LogFields): void {
    this.write('debug', message, fields);
  }

  info(message: string, fields?: LogFields): void {
    this.write('info', message, fields);
  }

  warn(message: string, fields?: LogFields): void {
    this.write('warn', message, fields);
  }

  error(message: string, fields?: LogFields): void {
    this.write('error', message, fields);
  }
}

export class NullLogger implements ILogger {
  debug(): void {}
  info(): void {}
  warn(): void {}
  error(): void {}
}
