const isLocal = process.env.NODE_ENV === "development"

export const ANKI_CONFIG = {
  API_URL: isLocal ? '/api/anki' : 'http://localhost:8000/api/anki',
  VERSION: 6,
  DEFAULT_DECK: 'Model Deck',
  DEFAULT_MODEL: 'Basic',
  DEFAULT_TAGS: ['mnemorai']
} as const;

