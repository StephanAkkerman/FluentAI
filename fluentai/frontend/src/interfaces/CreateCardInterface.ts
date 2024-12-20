export interface CreateCardInterface {
  word: string;
  language_code: string;
}

export interface CreateCardResponse {
  imageUrl: string;
  ttsUrl: string;
  IPA: string;
  recording: string;
  verbalCue: string;
  translation: string;
}
