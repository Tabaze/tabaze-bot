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

type Level = 'debug' | 'info' | 'warn' | 'error';

const RESET = '\x1b[0m';
const GRAY = '\x1b[90m';
const LEVEL_COLOR: Record<Level, string> = {
  debug: '\x1b[90m',
  info: '\x1b[36m',
  warn: '\x1b[33m',
  error: '\x1b[31m',
};

function colorsEnabled(): boolean {
  return Boolean(process.stdout.isTTY) && !process.env.NO_COLOR;
}

function formatTime(date: Date): string {
  return date.toLocaleTimeString('en-US', { hour12: true });
}

/** Renders fields as `key=value key2=value2`, the way the tsx/vite dev CLIs print request details next to a log line. */
function formatFields(fields: LogFields): string {
  const entries = Object.entries(fields);
  if (entries.length === 0) return '';
  const rendered = entries.map(([key, value]) => `${key}=${typeof value === 'string' ? value : JSON.stringify(value)}`).join(' ');
  return ` ${rendered}`;
}

export class ConsoleLogger implements ILogger {
  constructor(private readonly minLevel: Level = 'info') {}

  private static readonly LEVEL_ORDER = ['debug', 'info', 'warn', 'error'] as const;

  private shouldLog(level: Level): boolean {
    return ConsoleLogger.LEVEL_ORDER.indexOf(level) >= ConsoleLogger.LEVEL_ORDER.indexOf(this.minLevel);
  }

  private write(level: Level, message: string, fields?: LogFields): void {
    if (!this.shouldLog(level)) return;

    const time = formatTime(new Date());
    const details = formatFields(fields ? redact(fields) : {});
    const line = colorsEnabled()
      ? `${GRAY}${time}${RESET} ${LEVEL_COLOR[level]}[${level}]${RESET} ${message}${GRAY}${details}${RESET}`
      : `${time} [${level}] ${message}${details}`;

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
