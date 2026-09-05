import { buildCompositionRoot } from './config/composition-root.js';
import { ConfigurationError } from './domain/errors/configuration-error.js';
import { startHttpServer } from './api/http-server.js';

const DEFAULT_PORT = 3000;

function resolvePort(env: NodeJS.ProcessEnv): number {
  const raw = env.PORT;
  if (!raw || raw.trim() === '') return DEFAULT_PORT;
  const parsed = Number(raw);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new ConfigurationError(`"PORT" must be a positive integer, got "${raw}".`);
  }
  return parsed;
}

function main(): void {
  const port = resolvePort(process.env);
  const { llmService, logger } = buildCompositionRoot();
  startHttpServer({ llmService, logger }, port);
}

try {
  main();
} catch (error) {
  if (error instanceof ConfigurationError) {
    console.warn(`[${new Date().toISOString()}] Server not started -- ${error.message}\nSet the missing variable(s) above in .env, then rerun.`);
    process.exitCode = 1;
  } else {
    console.error('Fatal error:', error);
    process.exitCode = 1;
  }
}
