import axios from 'axios';
import { ANKI_CONFIG } from '@/config/constants';
import { AnkiConnectResponse } from '@/interfaces/AnkiInterface';

export class AnkiConnect {
  private readonly URL = ANKI_CONFIG.API_URL;
  private readonly VERSION = ANKI_CONFIG.VERSION;

  async invoke(action: string, params?: any): Promise<any> {
    const payload = {
      action,
      version: this.VERSION,
      params,
    };

    try {
      const response = await axios.post<AnkiConnectResponse>(this.URL, payload);
      return this.handleResponse(response.data);
    } catch (error) {
      console.error('AnkiConnect error:', error);
      throw new Error('Failed to connect to Anki');
    }
  }

  private handleResponse(data: AnkiConnectResponse) {
    if (!('error' in data) || !('result' in data)) {
      throw new Error('Invalid response format from Anki');
    }

    if (data.error) {
      throw new Error(data.error);
    }

    return data.result;
  }
}
