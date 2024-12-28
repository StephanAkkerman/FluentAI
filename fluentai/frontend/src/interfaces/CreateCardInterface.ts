export interface CreateCardInterface {
  word: string;
  language_code: string;
}

export interface CreateCardResponse {
  imageUrl: string;
  ttsUrl: string;
  ipa: string;
  verbalCue: string;
  translation: string;
}
