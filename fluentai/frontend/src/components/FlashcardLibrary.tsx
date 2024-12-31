import React, { useState, useEffect, useCallback } from 'react';
import { ChevronLeft, ChevronRight, Library, AlertCircle, RefreshCw } from 'lucide-react';
import Button from '@/components/ui/Button';
import { AnkiService } from '@/services/anki/ankiService';
import Flashcard from '@/components/Flashcard';
import { Card as FlashCard } from '@/interfaces/CardInterfaces';
import { ANKI_CONFIG } from '@/config/constants';

const FlashcardLibrary = () => {
  const [cards, setCards] = useState<FlashCard[]>([]);
  const [selectedDeck, setSelectedDeck] = useState('');
  const [decks, setDecks] = useState<string[]>([]);
  const [currentPage, setCurrentPage] = useState(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [isRefreshing, setIsRefreshing] = useState(false);

  const cardsPerPage = 6;

  const totalPages = Math.ceil(cards.length / cardsPerPage);
  const startIndex = (currentPage - 1) * cardsPerPage;
  const endIndex = startIndex + cardsPerPage;
  const currentCards = cards.slice(startIndex, endIndex);

  const fetchDecks = useCallback(async () => {
    setIsRefreshing(true);
    setError('');
    try {
      const ankiService = new AnkiService();
      const availableDecks = await ankiService.getAvailableDecks();
      setDecks(availableDecks);
    } catch (err) {
      console.error("Error loading decks: ", err);
      setError('Failed to load decks');
    } finally {
      setIsRefreshing(false);
    }
  }, []);

  useEffect(() => {
    fetchDecks();
  }, [fetchDecks]);

  const getMediaFile = async (filename: string): Promise<string> => {
    try {
      const response = await fetch(ANKI_CONFIG.API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'retrieveMediaFile',
          version: 6,
          params: {
            filename
          }
        })
      });

      const data = await response.json();
      if (!data.result) {
        throw new Error('Media file not found');
      }

      const base64Data = data.result;
      const isImage = filename.match(/\.(jpg|jpeg|png|gif)$/i);
      const mimeType = isImage ? `image/${filename.split('.').pop()}` : 'audio/mpeg';

      const blob = await fetch(`data:${mimeType};base64,${base64Data}`).then(res => res.blob());
      return URL.createObjectURL(blob);
    } catch (err) {
      console.error(`Failed to load media file ${filename}:`, err);
      return '';
    }
  };

  const sanitizeHtml = (html: string): string => {
    // Create a temporary div to handle HTML content
    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = html;

    // Handle special Anki tags and formatting
    const processNode = (node: Node): string => {
      if (node.nodeType === Node.TEXT_NODE) {
        return node.textContent || '';
      }

      if (node.nodeType === Node.ELEMENT_NODE) {
        const element = node as Element;
        const tag = element.tagName.toLowerCase();

        // Convert common HTML formatting to text with appropriate spacing
        switch (tag) {
          case 'div':
          case 'p':
            return processChildren(element) + '\n';
          case 'br':
            return '\n';
          case 'b':
          case 'strong':
            return `${processChildren(element)}`;
          case 'i':
          case 'em':
            return `${processChildren(element)}`;
          case 'ul':
          case 'ol':
            return processChildren(element) + '\n';
          case 'li':
            return `• ${processChildren(element)}\n`;
          default:
            return processChildren(element);
        }
      }

      return '';
    };

    const processChildren = (element: Element): string => {
      return Array.from(element.childNodes)
        .map(node => processNode(node))
        .join('');
    };

    // Process the content and clean up extra whitespace
    return processNode(tempDiv)
      .replace(/\n{3,}/g, '\n\n')  // Replace multiple newlines with double newlines
      .trim();
  };

  const loadDeck = async (deckName: string) => {
    setLoading(true);
    setError('');
    try {
      const response = await fetch(ANKI_CONFIG.API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'findNotes',
          version: 6,
          params: {
            query: `deck:"${deckName}"`
          }
        })
      });

      const data = await response.json();
      const noteIds = data.result;

      const notesResponse = await fetch(ANKI_CONFIG.API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'notesInfo',
          version: 6,
          params: {
            notes: noteIds
          }
        })
      });

      const notesData = await notesResponse.json();
      let notesInfo = notesData.result;
      notesInfo = notesInfo.filter((note: any) =>
        note.modelName === "FluentAI Model"
      );

      const transformedCards = await Promise.all(notesInfo.map(async (note: any) => {
        const imageFilename = note.fields.Picture.value.match(/src="([^"]+)"/)?.[1] || '';
        const audioFilename = note.fields['Pronunciation (Recording and/or IPA)'].value.match(/\[sound:([^\]]+)\]/)?.[1] || '';

        const [imageUrl, audioUrl] = await Promise.all([
          imageFilename ? getMediaFile(imageFilename) : Promise.resolve(''),
          audioFilename ? getMediaFile(audioFilename) : Promise.resolve('')
        ]);

        // Sanitize the verbal cue HTML content
        const verbalCueHtml = note.fields['Gender, Personal Connection, Extra Info (Back side)'].value;
        const sanitizedVerbalCue = sanitizeHtml(verbalCueHtml);

        return {
          word: note.fields.Word.value,
          translation: '',
          imageUrl,
          audioUrl,
          ipa: note.fields['Pronunciation (Recording and/or IPA)'].value.replace(/\[sound:[^\]]+\]/, '').trim(),
          verbalCue: sanitizedVerbalCue,
          languageCode: 'en'
        };
      }));

      setCards(transformedCards);
      setSelectedDeck(deckName);
      setCurrentPage(1);
    } catch (err) {
      console.error('Failed to load deck: ', err);
      setError('Failed to load deck');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    return () => {
      cards.forEach(card => {
        if (card.imageUrl.startsWith('blob:')) {
          URL.revokeObjectURL(card.imageUrl);
        }
        if (card.audioUrl.startsWith('blob:')) {
          URL.revokeObjectURL(card.audioUrl);
        }
      });
    };
  }, [cards]);

  return (
    <div className="space-y-8">
      {/* Deck Selection Card */}
      <div className="bg-white dark:bg-gray-800 shadow-2xl border border-gray-200 dark:border-gray-700 rounded-2xl">
        <div className="p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Library className="w-6 h-6 text-blue-500" />
              <h2 className="text-2xl font-bold bg-gradient-to-r from-blue-500 to-teal-400 bg-clip-text text-transparent">
                Select Your Deck
              </h2>
            </div>
          </div>
          <div className="flex gap-2">
            <select
              className="w-full p-3 border rounded-lg bg-gray-50 dark:bg-gray-700 border-gray-300 dark:border-gray-600 
                         focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200"
              value={selectedDeck}
              onChange={(e) => loadDeck(e.target.value)}
            >
              <option value="" disabled>Choose a deck...</option>
              {decks.map((deck) => (
                <option key={deck} value={deck}>{deck}</option>
              ))}
            </select>
            <button
              onClick={fetchDecks}
              disabled={isRefreshing}
              className="p-2 border rounded hover:bg-gray-100 disabled:opacity-50"
            >
              <RefreshCw className={`h-5 w-5 ${isRefreshing ? 'animate-spin' : ''}`} />
            </button>
          </div>
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <div className="flex items-center gap-3 bg-red-50 dark:bg-red-900/30 text-red-700 dark:text-red-400 p-4 rounded-2xl border border-red-200 dark:border-red-800">
          <AlertCircle className="w-5 h-5" />
          <p>{error}</p>
        </div>
      )}

      {/* Loading State */}
      {loading && (
        <div className="flex flex-col items-center justify-center p-12 bg-white dark:bg-gray-800 rounded-2xl shadow-2xl border border-gray-200 dark:border-gray-700">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
          <p className="mt-4 text-gray-600 dark:text-gray-400">Loading your flashcards...</p>
        </div>
      )}

      {/* Cards Grid */}
      {!loading && cards.length > 0 && (
        <div className="space-y-8">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 justify-center items-center">
            {currentCards.map((card, index) => (
              <div key={index} className="transform transition-all duration-300 hover:scale-105 m-auto">
                <Flashcard
                  card={card}
                  isLoading={false}
                  showFront={true}
                />
              </div>
            ))}
          </div>

          {/* Pagination */}
          <div className="flex justify-center items-center gap-6 bg-white dark:bg-gray-800 p-4 rounded-2xl shadow-2xl border border-gray-200 dark:border-gray-700">
            <Button
              onClick={() => setCurrentPage(curr => Math.max(1, curr - 1))}
              disabled={currentPage === 1}
              variant="secondary"
              className="flex items-center gap-2"
            >
              <ChevronLeft className="w-5 h-5" />
              Previous
            </Button>

            <div className="flex items-center gap-3">
              <span className="text-lg font-medium px-4 py-2 rounded-lg bg-gray-100 dark:bg-gray-700">
                {currentPage}
              </span>
              <span className="text-gray-500 dark:text-gray-400">of</span>
              <span className="text-lg font-medium px-4 py-2 rounded-lg bg-gray-100 dark:bg-gray-700">
                {totalPages}
              </span>
            </div>

            <Button
              onClick={() => setCurrentPage(curr => Math.min(totalPages, curr + 1))}
              disabled={currentPage === totalPages}
              variant="secondary"
              className="flex items-center gap-2"
            >
              Next
              <ChevronRight className="w-5 h-5" />
            </Button>
          </div>
        </div>
      )}


      {/* Empty State */}
      {!loading && !error && cards.length === 0 && selectedDeck && (
        <div className="flex flex-col items-center justify-center p-12 bg-white dark:bg-gray-800 rounded-2xl shadow-2xl border border-gray-200 dark:border-gray-700">
          <Library className="w-16 h-16 text-gray-400 mb-4" />
          <h3 className="text-xl font-semibold mb-2">No Cards Found</h3>
          <p className="text-gray-600 dark:text-gray-400 text-center">
            This deck doesn&apos;t have any FluentAI cards yet. Create some cards first!
          </p>
        </div>
      )}
    </div>
  );
};

export default FlashcardLibrary;
