export interface PlatformSettings {
  project_name: string;
  llm_provider: string;
  llm_model: string;
  llm_base_url: string;
  masked_api_key: string;
  default_temperature: number;
  default_top_p: number;
  default_max_tokens: number;
  debug_mode: boolean;
}

export interface StreamTokenEvent {
  type: 'token';
  token: string;
}

export interface StreamStartEvent {
  type: 'start';
  conversation_id: string;
  sources?: any[];
}

export interface StreamDoneEvent {
  type: 'done';
  message_id: string;
  content: string;
}

export interface StreamErrorEvent {
  type: 'error';
  error: string;
}

export type SSEEvent = StreamTokenEvent | StreamStartEvent | StreamDoneEvent | StreamErrorEvent;
