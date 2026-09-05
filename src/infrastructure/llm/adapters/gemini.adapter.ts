import { randomUUID } from 'node:crypto';
import { LLMProvider } from '../../../domain/enums/provider.enum.js';
import { MessageRole, type MessageList } from '../../../domain/models/message.js';
import type { GenerationConfig } from '../../../domain/models/generation-config.js';
import type { ModelResponse, TokenUsage } from '../../../domain/models/model-response.js';
import type { StreamChunk } from '../../../domain/models/stream-chunk.js';
import { FinishReason } from '../../../domain/enums/finish-reason.enum.js';
import type { ProviderCapabilities } from '../../../domain/models/provider-capabilities.js';
import { NetworkError, UnknownLLMError } from '../../../domain/errors/llm-error.js';
import { BaseLLMAdapter, type GenerateOptions } from './base-llm.adapter.js';
import { executeHttpRequest } from '../http/http-client.js';
import { parseSseStream } from '../http/sse-parser.js';
import type { GeminiProviderConfig } from '../configuration/llm.config.js';
import { ModelRegistry } from '../configuration/model-registry.js';
import { TokenizerService } from '../tokenization/tokenizer.service.js';
import type { ILogger } from '../../observability/logger.js';

interface GeminiPart {
  readonly text?: string;
}

interface GeminiContent {
  readonly role: 'user' | 'model';
  readonly parts: readonly GeminiPart[];
}

interface GeminiCandidate {
  readonly content?: GeminiContent;
  readonly finishReason?: string;
}

interface GeminiUsageMetadata {
  readonly promptTokenCount?: number;
  readonly candidatesTokenCount?: number;
  readonly totalTokenCount?: number;
}

interface GeminiResponse {
  readonly candidates?: readonly GeminiCandidate[];
  readonly usageMetadata?: GeminiUsageMetadata;
  readonly modelVersion?: string;
}

interface GeminiErrorBody {
  readonly error?: { readonly code?: number; readonly message?: string; readonly status?: string };
}

/**
 * Fixed, moderate default safety configuration. Isolated here rather than
 * exposed through the unified GenerationConfig, since safety-category
 * thresholds are a Gemini-specific concept with no equivalent in the other
 * providers' APIs.
 */
const DEFAULT_SAFETY_SETTINGS = [
  { category: 'HARM_CATEGORY_HARASSMENT', threshold: 'BLOCK_ONLY_HIGH' },
  { category: 'HARM_CATEGORY_HATE_SPEECH', threshold: 'BLOCK_ONLY_HIGH' },
  { category: 'HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold: 'BLOCK_ONLY_HIGH' },
  { category: 'HARM_CATEGORY_DANGEROUS_CONTENT', threshold: 'BLOCK_ONLY_HIGH' },
];

function mapRole(role: MessageRole): 'user' | 'model' {
  return role === MessageRole.Assistant ? 'model' : 'user';
}

/** Gemini has no message-level system role; system content becomes a top-level systemInstruction. */
function extractSystemAndContents(messages: MessageList): { systemInstruction: GeminiContent | undefined; contents: GeminiContent[] } {
  const systemParts: string[] = [];
  const contents: GeminiContent[] = [];

  for (const message of messages) {
    if (message.role === MessageRole.System) {
      systemParts.push(message.content);
    } else {
      contents.push({ role: mapRole(message.role), parts: [{ text: message.content }] });
    }
  }

  return {
    systemInstruction: systemParts.length > 0 ? { role: 'user', parts: [{ text: systemParts.join('\n\n') }] } : undefined,
    contents,
  };
}

function mapFinishReason(reason: string | null | undefined): FinishReason {
  switch (reason) {
    case 'STOP':
      return FinishReason.Stop;
    case 'MAX_TOKENS':
      return FinishReason.MaxTokens;
    case 'SAFETY':
    case 'RECITATION':
      return FinishReason.ContentFilter;
    case undefined:
    case null:
      return FinishReason.Unknown;
    default:
      return FinishReason.Unknown;
  }
}

function toUsage(usage: GeminiUsageMetadata | undefined): TokenUsage {
  const promptTokens = usage?.promptTokenCount ?? 0;
  const completionTokens = usage?.candidatesTokenCount ?? 0;
  return { promptTokens, completionTokens, totalTokens: usage?.totalTokenCount ?? promptTokens + completionTokens };
}

function extractText(candidate: GeminiCandidate | undefined): string {
  return candidate?.content?.parts.map((part) => part.text ?? '').join('') ?? '';
}

/**
 * Adapter for the Google Gemini (Generative Language) API. Deliberately
 * not modeled as "OpenAI-compatible": Gemini's request/response shapes
 * (contents/parts, model-role assistant turns, systemInstruction,
 * usageMetadata) are mapped directly rather than through an
 * OpenAI-shaped intermediate.
 */
export class GeminiAdapter extends BaseLLMAdapter {
  readonly provider = LLMProvider.Gemini;

  constructor(
    private readonly config: GeminiProviderConfig,
    private readonly modelRegistry: ModelRegistry,
    private readonly tokenizer: TokenizerService,
    private readonly logger: ILogger,
  ) {
    super();
  }

  private headers(): Record<string, string> {
    return {
      'Content-Type': 'application/json',
      'x-goog-api-key': this.config.apiKey,
    };
  }

  private buildRequestBody(messages: MessageList, config: GenerationConfig): Record<string, unknown> {
    const { systemInstruction, contents } = extractSystemAndContents(messages);
    const generationConfig: Record<string, unknown> = {};
    if (config.temperature !== undefined) generationConfig.temperature = config.temperature;
    if (config.topP !== undefined) generationConfig.topP = config.topP;
    if (config.maxTokens !== undefined) generationConfig.maxOutputTokens = config.maxTokens;
    if (config.presencePenalty !== undefined) generationConfig.presencePenalty = config.presencePenalty;
    if (config.frequencyPenalty !== undefined) generationConfig.frequencyPenalty = config.frequencyPenalty;
    if (config.stopSequences !== undefined && config.stopSequences.length > 0) generationConfig.stopSequences = config.stopSequences;

    const body: Record<string, unknown> = {
      contents,
      generationConfig,
      safetySettings: DEFAULT_SAFETY_SETTINGS,
    };
    if (systemInstruction) body.systemInstruction = systemInstruction;
    return body;
  }

  private async raiseForStatus(response: Response): Promise<never> {
    let message = `Gemini request failed with status ${response.status}`;
    try {
      const body = (await response.json()) as GeminiErrorBody;
      message = body.error?.message ?? message;
    } catch {
      // Body was not JSON; keep the generic message.
    }

    const retryAfterMs = this.parseRetryAfterMs(response.headers);
    if (response.status === 400 && /token count|context (window|length)|exceeds the maximum/i.test(message)) {
      throw this.contextLengthError(message, response.status);
    }
    throw this.classifyByStatus(response.status, message, retryAfterMs);
  }

  async generate(messages: MessageList, model: string, config: GenerationConfig, options?: GenerateOptions): Promise<ModelResponse> {
    const response = await executeHttpRequest(
      {
        url: `${this.config.baseUrl}/models/${model}:generateContent`,
        method: 'POST',
        headers: this.headers(),
        body: this.buildRequestBody(messages, config),
        timeoutMs: config.timeoutMs,
        signal: options?.signal,
      },
      this.provider,
    );

    if (!response.ok) {
      await this.raiseForStatus(response);
    }

    const parsed = (await response.json()) as GeminiResponse;
    const candidate = parsed.candidates?.[0];
    if (!candidate) {
      throw new UnknownLLMError('Gemini response contained no candidates.', { provider: this.provider });
    }

    return {
      id: randomUUID(),
      provider: this.provider,
      model: parsed.modelVersion || model,
      content: extractText(candidate),
      usage: toUsage(parsed.usageMetadata),
      finishReason: mapFinishReason(candidate.finishReason),
    };
  }

  async *generateStream(
    messages: MessageList,
    model: string,
    config: GenerationConfig,
    options?: GenerateOptions,
  ): AsyncIterable<StreamChunk> {
    const response = await executeHttpRequest(
      {
        url: `${this.config.baseUrl}/models/${model}:streamGenerateContent?alt=sse`,
        method: 'POST',
        headers: this.headers(),
        body: this.buildRequestBody(messages, config),
        timeoutMs: config.timeoutMs,
        signal: options?.signal,
      },
      this.provider,
    );

    if (!response.ok) {
      await this.raiseForStatus(response);
    }
    if (!response.body) {
      throw new NetworkError('Gemini streaming response had no body.', { provider: this.provider });
    }

    const generationId = randomUUID();
    let resolvedModel = model;

    for await (const event of parseSseStream(response.body, this.provider, options?.signal)) {
      let parsed: GeminiResponse;
      try {
        parsed = JSON.parse(event.data) as GeminiResponse;
      } catch (error) {
        this.logger.warn('Failed to parse Gemini stream chunk', { provider: this.provider, error: (error as Error).message });
        continue;
      }

      resolvedModel = parsed.modelVersion || resolvedModel;
      const candidate = parsed.candidates?.[0];
      const delta = extractText(candidate);
      const finishReason = candidate?.finishReason ? mapFinishReason(candidate.finishReason) : null;
      const usage = parsed.usageMetadata ? toUsage(parsed.usageMetadata) : null;

      if (delta.length === 0 && finishReason === null && usage === null) continue;

      yield {
        id: generationId,
        provider: this.provider,
        model: resolvedModel,
        delta,
        finishReason,
        usage,
      };
    }
  }

  async tokenizeAndCount(messages: MessageList): Promise<number> {
    return this.tokenizer.count(messages, this.provider).count;
  }

  getCapabilities(model: string): ProviderCapabilities {
    return this.modelRegistry.getCapabilities(this.provider, model);
  }

  getContextLimit(model: string): number {
    return this.modelRegistry.getContextLimit(this.provider, model);
  }
}
