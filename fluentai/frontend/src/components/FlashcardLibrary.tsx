import React, { useState, useEffect } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import Button from '@/components/ui/Button';
import { AnkiService } from '@/services/anki/ankiService';
import Flashcard from '@/components/Flashcard';
import { Card } from '@/interfaces/CardInterfaces';

const FlashcardLibrary = () => {
  const [cards, setCards] = useState<Card[]>([]);
  const [selectedDeck, setSelectedDeck] = useState('');
  const [decks, setDecks] = useState<string[]>([]);
  const [currentPage, setCurrentPage] = useState(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const cardsPerPage = 6;
  const ankiService = new AnkiService();

  const totalPages = Math.ceil(cards.length / cardsPerPage);
  const startIndex = (currentPage - 1) * cardsPerPage;
  const endIndex = startIndex + cardsPerPage;
  const currentCards = cards.slice(startIndex, endIndex);

  useEffect(() => {
    const fetchDecks = async () => {
      try {
        const availableDecks = await ankiService.getAvailableDecks();
        setDecks(availableDecks);
      } catch (err) {
        setError('Failed to load decks');
      }
    };
    fetchDecks();
  }, []);

  const getMediaFile = async (filename: string): Promise<string> => {
    try {
      const response = await fetch('/api/anki', {
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
    let cleanText = processNode(tempDiv)
      .replace(/\n{3,}/g, '\n\n')  // Replace multiple newlines with double newlines
      .trim();

    return cleanText;
  };

  const loadDeck = async (deckName: string) => {
    setLoading(true);
    setError('');
    try {
      const response = await fetch('/api/anki', {
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

      const notesResponse = await fetch('/api/anki', {
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
      <div className='p-6'>
        <h2 className="text-2xl font-bold mb-4">Select Deck</h2>
        <select
          className="w-full p-2 border rounded"
          value={selectedDeck}
          onChange={(e) => loadDeck(e.target.value)}
        >
          <option value="">Choose a deck...</option>
          {decks.map((deck) => (
            <option key={deck} value={deck}>{deck}</option>
          ))}
        </select>
      </div>

      {error && (
        <div className="bg-red-100 text-red-700 p-4 rounded">
          {error}
        </div>
      )}

      {loading && (
        <div className="text-center p-8">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto"></div>
          <p className="mt-4">Loading cards...</p>
        </div>
      )}

      {!loading && cards.length > 0 && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {currentCards.map((card, index) => (
              <div key={index} className="transform transition-all duration-300 hover:scale-105">
                <Flashcard
                  card={card}
                  isLoading={false}
                  showFront={true}
                />
              </div>
            ))}
          </div>

          <div className="flex justify-center items-center space-x-4 mt-8">
            <Button
              text=""
              onClick={() => setCurrentPage(curr => Math.max(1, curr - 1))}
              disabled={currentPage === 1}
              className="p-2"
            >
              <ChevronLeft className="w-6 h-6" />
            </Button>

            <span className="text-lg font-medium">
              Page {currentPage} of {totalPages}
            </span>

            <Button
              text=""
              onClick={() => setCurrentPage(curr => Math.min(totalPages, curr + 1))}
              disabled={currentPage === totalPages}
              className="p-2"
            >
              <ChevronRight className="w-6 h-6" />
            </Button>
          </div>
        </>
      )}
    </div>
  );
};

export default FlashcardLibrary;
