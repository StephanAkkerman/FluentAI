export interface AnkiNote {
  deckName: string;
  modelName: string;
  fields: Record<string, string>;
  options?: {
    allowDuplicate: boolean;
  };
  tags?: string[];
}
