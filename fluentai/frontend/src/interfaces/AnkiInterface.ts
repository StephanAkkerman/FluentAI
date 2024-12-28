export interface AnkiConnectResponse {
  result: any;
  error: string | null;
}

export interface AnkiNote {
  deckName: string;
  modelName: string;
  fields: Record<string, string>;
  options?: {
    allowDuplicate: boolean;
  };
  tags?: string[];
}

export interface Card {
  img: string;
  word: string;
  keyPhrase: string;
  translation: string;
  ipa: string;
  recording: string;
}

