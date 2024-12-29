const isLocal = process.env.NODE_ENV === "development"

export const ANKI_CONFIG = {
  API_URL: isLocal ? '/api/anki' : 'http://127.0.0.1:8765',
  VERSION: 6,
  DEFAULT_DECK: 'Model Deck',
  DEFAULT_MODEL: 'Basic',
  DEFAULT_TAGS: ['FluentAI']
} as const;

