// The core card data structure used throughout the application
export interface Card {
  // Core fields
  word: string;
  translation: string;
  languageCode: string;

  // Visual and audio elements
  imageUrl: string;
  audioUrl: string;

  // Additional metadata
  ipa: string;
  verbalCue: string;

  // Optional fields for specific use cases
  tags?: string[];
  notes?: string;
  testSpelling?: boolean;
}

// Request type for card creation
export interface CreateCardRequest {
  word: string;
  languageCode: string;
  mnemonicKeyword?: string;
  keySentence?: string;
  imageModel?: string;
  llmModel?: string;
}

// Used when mapping a Card to an Anki note
export interface AnkiNote {
  deckName: string;
  modelName: string;
  fields: {
    Word: string;
    Picture: string;
    "Gender, Personal Connection, Extra Info (Back side)": string;
    "Pronunciation (Recording and/or IPA)": string;
    "Test Spelling? (y = yes, blank = no)": string;
  };
  options: {
    allowDuplicate: boolean;
  };
  tags: string[];
}

// Helper functions for converting between types
export const cardToAnkiNote = (
  card: Card,
  deckName: string,
  modelName: string = 'FluentAI Model'
): AnkiNote => {
  return {
    deckName,
    modelName,
    fields: {
      "Word": card.word,
      "Picture": `<img src="${card.imageUrl}" />`,
      "Gender, Personal Connection, Extra Info (Back side)": card.verbalCue + (card.notes ? `<div>${card.notes}</div>` : ""),
      "Pronunciation (Recording and/or IPA)": `${card.ipa}[sound:${card.audioUrl}]`,
      "Test Spelling? (y = yes, blank = no)": card.testSpelling ? "y" : "",
    },
    options: {
      allowDuplicate: false,
    },
    tags: card.tags || ['FluentAI'],
  };
};

// Helper function to create a Card from API response
export const createCardFromResponse = (
  request: CreateCardRequest,
  response: {
    imageUrl: string;
    ttsUrl: string;
    ipa: string;
    verbal_cue: string;
    translation: string;
  }
): Card => {
  return {
    word: request.word,
    languageCode: request.languageCode,
    translation: response.translation,
    imageUrl: response.imageUrl,
    audioUrl: response.ttsUrl,
    ipa: response.ipa,
    verbalCue: response.verbal_cue,
  };
};
