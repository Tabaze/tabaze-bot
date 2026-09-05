export enum LLMProvider {
  OpenAI = 'openai',
  Anthropic = 'anthropic',
  Gemini = 'gemini',
  Local = 'local',
}

export function isLLMProvider(value: string): value is LLMProvider {
  return (Object.values(LLMProvider) as string[]).includes(value);
}
