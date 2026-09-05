import { createServer, type IncomingMessage, type ServerResponse, type Server } from 'node:http';
import type { ILLMService } from '../application/interfaces/illm-service.interface.js';
import type { ILogger } from '../infrastructure/observability/logger.js';
import { MessageRole, type Message, type MessageList } from '../domain/models/message.js';
import type { GenerationConfigInput } from '../domain/models/generation-config.js';
import { LLMError } from '../domain/errors/llm-error.js';
import { ErrorCategory } from '../domain/enums/error-category.enum.js';
import { ConfigurationError } from '../domain/errors/configuration-error.js';

export interface HttpServerDeps {
  readonly llmService: ILLMService;
  readonly logger: ILogger;
}

const MAX_BODY_BYTES = 1_000_000;

function readJsonBody(req: IncomingMessage): Promise<unknown> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on('data', (chunk: Buffer) => {
      size += chunk.length;
      if (size > MAX_BODY_BYTES) {
        reject(new ConfigurationError('Request body too large.'));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => {
      if (chunks.length === 0) {
        resolve(undefined);
        return;
      }
      try {
        resolve(JSON.parse(Buffer.concat(chunks).toString('utf8')));
      } catch {
        reject(new ConfigurationError('Request body must be valid JSON.'));
      }
    });
    req.on('error', reject);
  });
}

function isMessageRole(value: unknown): value is MessageRole {
  return typeof value === 'string' && (Object.values(MessageRole) as string[]).includes(value);
}

function parseMessages(body: unknown): MessageList {
  if (typeof body !== 'object' || body === null || !('messages' in body)) {
    throw new ConfigurationError('Request body must include a "messages" array.');
  }
  const { messages } = body as { messages: unknown };
  if (!Array.isArray(messages) || messages.length === 0) {
    throw new ConfigurationError('"messages" must be a non-empty array.');
  }
  return messages.map((entry, index): Message => {
    if (typeof entry !== 'object' || entry === null) {
      throw new ConfigurationError(`messages[${index}] must be an object with "role" and "content".`);
    }
    const { role, content } = entry as { role?: unknown; content?: unknown };
    if (!isMessageRole(role)) {
      throw new ConfigurationError(`messages[${index}].role must be one of: ${Object.values(MessageRole).join(', ')}.`);
    }
    if (typeof content !== 'string') {
      throw new ConfigurationError(`messages[${index}].content must be a string.`);
    }
    return { role, content };
  });
}

function parseGenerationConfig(body: unknown): GenerationConfigInput | undefined {
  if (typeof body !== 'object' || body === null || !('config' in body) || body.config === undefined) {
    return undefined;
  }
  const { config } = body as { config: unknown };
  if (typeof config !== 'object' || config === null) {
    throw new ConfigurationError('"config" must be an object.');
  }
  return config as GenerationConfigInput;
}

const ERROR_CATEGORY_STATUS: Record<ErrorCategory, number> = {
  [ErrorCategory.Authentication]: 401,
  [ErrorCategory.Authorization]: 403,
  [ErrorCategory.RateLimit]: 429,
  [ErrorCategory.Timeout]: 504,
  [ErrorCategory.Network]: 502,
  [ErrorCategory.ProviderUnavailable]: 503,
  [ErrorCategory.InvalidRequest]: 400,
  [ErrorCategory.ContextLength]: 400,
  [ErrorCategory.ContentFilter]: 422,
  [ErrorCategory.ServerError]: 502,
  [ErrorCategory.Cancelled]: 499,
  [ErrorCategory.Unknown]: 500,
};

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  if (res.headersSent) return;
  const payload = JSON.stringify(body);
  res.writeHead(status, { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(payload) });
  res.end(payload);
}

function sendError(res: ServerResponse, logger: ILogger, error: unknown): void {
  if (error instanceof ConfigurationError) {
    sendJson(res, 400, { error: { message: error.message } });
    return;
  }
  if (error instanceof LLMError) {
    logger.error('LLM request failed', { category: error.category, provider: error.provider, message: error.message });
    sendJson(res, ERROR_CATEGORY_STATUS[error.category], {
      error: { message: error.message, category: error.category, provider: error.provider },
    });
    return;
  }
  logger.error('Unhandled request error', { message: error instanceof Error ? error.message : String(error) });
  sendJson(res, 500, { error: { message: 'Internal server error.' } });
}

async function handleGenerate(req: IncomingMessage, res: ServerResponse, deps: HttpServerDeps): Promise<void> {
  const body = await readJsonBody(req);
  const messages = parseMessages(body);
  const config = parseGenerationConfig(body);
  const response = await deps.llmService.generate(messages, config);
  sendJson(res, 200, response);
}

async function handleGenerateStream(req: IncomingMessage, res: ServerResponse, deps: HttpServerDeps): Promise<void> {
  const body = await readJsonBody(req);
  const messages = parseMessages(body);
  const config = parseGenerationConfig(body);

  res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', Connection: 'keep-alive' });
  for await (const chunk of deps.llmService.generateStream(messages, config)) {
    res.write(`data: ${JSON.stringify(chunk)}\n\n`);
  }
  res.end();
}

async function handleCountTokens(req: IncomingMessage, res: ServerResponse, deps: HttpServerDeps): Promise<void> {
  const body = await readJsonBody(req);
  const messages = parseMessages(body);
  const promptTokens = await deps.llmService.countTokens(messages);
  sendJson(res, 200, { promptTokens });
}

/**
 * Minimal HTTP delivery layer over ILLMService: deliberately dependency-free
 * (node:http only). Swap for Express/Fastify if routing needs grow beyond
 * this handful of routes.
 */
export function startHttpServer(deps: HttpServerDeps, port: number): Server {
  const server = createServer((req, res) => {
    const method = req.method ?? 'GET';
    const url = req.url ?? '/';

    const respond = async (): Promise<void> => {
      if (method === 'GET' && url === '/health') {
        sendJson(res, 200, { status: 'ok' });
        return;
      }
      if (method === 'POST' && url === '/v1/generate') {
        await handleGenerate(req, res, deps);
        return;
      }
      if (method === 'POST' && url === '/v1/generate/stream') {
        await handleGenerateStream(req, res, deps);
        return;
      }
      if (method === 'POST' && url === '/v1/tokens') {
        await handleCountTokens(req, res, deps);
        return;
      }
      sendJson(res, 404, { error: { message: `No route for ${method} ${url}` } });
    };

    respond().catch((error: unknown) => {
      if (res.headersSent) {
        res.end();
        return;
      }
      sendError(res, deps.logger, error);
    });
  });

  server.listen(port, () => {
    deps.logger.info(`Server listening on port ${port}`, { port });
  });

  return server;
}
