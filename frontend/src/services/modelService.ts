import axios from "axios";
import { ModelOptions } from "@/interfaces/ModelInterface";

const BASE_URL = "http://localhost:8000/create_card";

export class ModelService {
  async getAvailableModels(): Promise<ModelOptions> {
    try {
      const [imageModelsResponse, llmModelsResponse] = await Promise.all([
        axios.get(`${BASE_URL}/image_models`),
        axios.get(`${BASE_URL}/llm_models`)
      ]);

      return {
        imageModels: imageModelsResponse.data.models,
        llmModels: llmModelsResponse.data.models
      };
    } catch (error) {
      console.error('Error fetching models:', error);
      throw new Error('Failed to fetch available models');
    }
  }
}

