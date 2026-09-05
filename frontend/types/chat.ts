export type MessageRole = 'user' | 'assistant' | 'system';

export interface SourceReference {
  rank: number;
  source: string;
  page_number?: number;
  file_type: string;
  score: number;
  score_pct: string;
  text: string;
}

export interface ChatMessage {
  id: string;
  conversation_id: string;
  role: MessageRole;
  content: string;
  sources?: { items: SourceReference[] } | SourceReference[];
  feedback?: 'like' | 'dislike';
  created_at: number;
}


export interface ConversationSummary {
  id: string;
  title: string;
  message_count: number;
  created_at: number;
  updated_at: number;
}

export interface ConversationDetail {
  id: string;
  title: string;
  system_prompt?: string;
  created_at: number;
  updated_at: number;
  messages: ChatMessage[];
}
