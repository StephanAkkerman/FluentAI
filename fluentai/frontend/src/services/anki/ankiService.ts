import axios from 'axios';
import { Card } from '@/interfaces/AnkiInterface';

export class AnkiService {
  private readonly API_URL = '/FluentAI/api/anki';

  private async getImageAsBase64(imageUrl: string): Promise<string> {
    try {
      // Fetch the image
      const response = await axios.get(imageUrl, {
        responseType: 'arraybuffer'
      });

      // Convert to base64
      const base64 = Buffer.from(response.data, 'binary').toString('base64');

      // Get file extension from URL or default to jpg
      const extension = imageUrl.split('.').pop()?.toLowerCase() || 'jpg';

      return `data:image/${extension};base64,${base64}`;
    } catch (error) {
      console.error('Error converting image to base64:', error);
      throw new Error('Failed to process image');
    }
  }

  private async storeMediaFile(filename: string, data: string): Promise<void> {
    try {
      await axios.post(this.API_URL, {
        action: 'storeMediaFile',
        version: 6,
        params: {
          filename,
          data: data.split(',')[1] // Remove the data:image/jpeg;base64, part
        }
      });
    } catch (error) {
      console.error('Error storing media file:', error);
      throw new Error('Failed to store media file in Anki');
    }
  }

  async saveCard(card: Card, deckName: string): Promise<void> {
    try {
      // First, convert the image to base64 and store it
      const base64Image = await this.getImageAsBase64(card.img);
      const filename = `fluentai-${card.word}-${Date.now()}.jpg`;
      await this.storeMediaFile(filename, base64Image);

      // Then create the note with the stored image
      await axios.post(this.API_URL, {
        action: 'addNote',
        version: 6,
        params: {
          note: {
            deckName: deckName,
            modelName: 'Basic',
            fields: {
              Front: `${card.word}<br><img src="${filename}" />`,
              Back: `${card.translation} - ${card.keyPhrase}`,
            },
            options: {
              allowDuplicate: false,
            },
            tags: ['FluentAI'],
          },
        },
      });
    } catch (error) {
      console.error('Error saving to Anki:', error);
      throw new Error('Failed to save card to Anki.');
    }
  }

  async getAvailableDecks(): Promise<string[]> {
    try {
      const response = await axios.post(this.API_URL, {
        action: 'deckNames',
        version: 6,
      });
      // Remove the default deck from possibilities
      const decks = response.data.result.filter((deck: string) =>
        deck !== "default"
      );
      return decks;
    } catch (error) {
      console.error('Error fetching decks:', error);
      throw new Error('Failed to fetch decks.');
    }
  }
}
