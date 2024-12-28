import axios from 'axios';
import { Card, cardToAnkiNote } from '@/interfaces/CardInterfaces';
import { ANKI_CONFIG } from '@/config/constants';

export class AnkiService {
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
      await axios.post(ANKI_CONFIG.API_URL, {
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

  async checkAndCreateModel(modelName: string = 'FluentAI Model'): Promise<void> {
    try {
      // Check if the model exists
      const response = await axios.post(ANKI_CONFIG.API_URL, {
        action: 'modelNames',
        version: 6,
      });

      const models: string[] = response.data.result;
      if (!models.includes(modelName)) {
        // Create the model if it doesn't exist
        await axios.post(ANKI_CONFIG.API_URL, {
          action: 'createModel',
          version: 6,
          params: {
            modelName,
            inOrderFields: [
              'Word',
              'Picture',
              'Gender, Personal Connection, Extra Info (Back side)',
              'Pronunciation (Recording and/or IPA)',
              'Test Spelling? (y = yes, blank = no)',
            ],
            css: `
              .card {
                font-family: arial;
                font-size: 30px;
                text-align: center;
                color: black;
                background-color: white;
              }
              .card1 { background-color: #FFFFFF; }
              .card2 { background-color: #FFFFFF; }
            `,
            isCloze: false,
            cardTemplates: [
              {
                Name: 'Word - Mnemonic',
                Front: '{{Word}}\n\n',
                Back: '{{FrontSide}}\n\n<hr id=answer>\n{{Picture}}\n\n{{#Pronunciation (Recording and/or IPA)}}\n<br>\n<font color=blue>{{Pronunciation (Recording and/or IPA)}}</font>{{/Pronunciation (Recording and/or IPA)}}<br>\n\n\n<span style="color:grey">\n{{Gender, Personal Connection, Extra Info (Back side)}}</span>\n<br><br>\n',
              },
              {
                Name: 'Mnemonic - Word',
                Front: '{{Picture}}<br><br>',
                Back: '{{FrontSide}}\n\n<hr id=answer>\n<br>\n<span style="font-size:1.5em;">{{Word}}</span><br>\n\n\n{{#Pronunciation (Recording and/or IPA)}}<br><font color=blue>{{Pronunciation (Recording and/or IPA)}}</font>{{/Pronunciation (Recording and/or IPA)}}\n\n{{#Gender, Personal Connection, Extra Info (Back side)}}<br><font color=grey>{{Gender, Personal Connection, Extra Info (Back side)}}</font>{{/Gender, Personal Connection, Extra Info (Back side)}}\n\n\n<span style="">',
              },
              {
                Name: 'Mnemonic - Spelling',
                Front: "{{#Test Spelling? (y = yes, blank = no)}}\nSpell this word: <br><br>\n{{Picture}}<br>\n\n{{#Pronunciation (Recording and/or IPA)}}<br><font color=blue>{{Pronunciation (Recording and/or IPA)}}</font>{{/Pronunciation (Recording and/or IPA)}}\n<br>\n\n{{/Test Spelling? (y = yes, blank = no)}}\n\n\n",
                Back: '<span style="font-size:1.5em;">{{Word}}</span><br><br>\n\n\n{{Picture}}<br>\n\n<span style="color:grey;">{{Gender, Personal Connection, Extra Info (Back side)}}</span>\n',
              },
            ],
          },
        });
      }
    } catch (error) {
      console.error('Error checking/creating model:', error);
      throw new Error('Failed to ensure the model exists in Anki.');
    }
  }

  async saveCard(card: Card, deckName: string, testSpelling: boolean): Promise<void> {
    try {
      // Ensure the model exists
      await this.checkAndCreateModel();

      // First, convert the image to base64 and store it
      const base64Image = await this.getImageAsBase64(card.imageUrl);
      const imageFilename = `fluentai-${card.word}-${Date.now()}.jpg`;
      await this.storeMediaFile(imageFilename, base64Image);

      // Convert the pronunciation to base64 and store it
      const base64Audio = await this.getImageAsBase64(card.audioUrl);
      const recordingFilename = `fluentai-${card.word}-${Date.now()}.wav`;
      await this.storeMediaFile(recordingFilename, base64Audio);

      // Update the card URLs to use the media files instead
      const cardWithMediaFiles = { ...card, testSpelling, imageUrl: imageFilename, audioUrl: recordingFilename }

      const note = cardToAnkiNote(cardWithMediaFiles, deckName);

      // Then create the note with the stored image
      await axios.post(ANKI_CONFIG.API_URL, {
        action: 'addNote',
        version: 6,
        params: { note },
      });
    } catch (error) {
      console.error('Error saving to Anki:', error);
      throw new Error('Failed to save card to Anki.');
    }
  }

  async getAvailableDecks(): Promise<string[]> {
    try {
      const response = await axios.post(ANKI_CONFIG.API_URL, {
        action: 'deckNames',
        version: 6,
      });
      // Remove the default deck from possibilities
      const decks = response.data.result.filter((deck: string) =>
        deck.toLowerCase() !== "default"
      );
      return decks;
    } catch (error) {
      console.error('Error fetching decks:', error);
      throw new Error('Failed to fetch decks.');
    }
  }
}
