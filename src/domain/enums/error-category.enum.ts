export enum ErrorCategory {
  Authentication = 'authentication',
  Authorization = 'authorization',
  RateLimit = 'rate_limit',
  Timeout = 'timeout',
  Network = 'network',
  ProviderUnavailable = 'provider_unavailable',
  InvalidRequest = 'invalid_request',
  ContextLength = 'context_length',
  ContentFilter = 'content_filter',
  ServerError = 'server_error',
  Cancelled = 'cancelled',
  Unknown = 'unknown',
}
