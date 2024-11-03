"""
This script interfaces with Anki through AnkiConnect to add notes to Anki decks programmatically.
It demonstrates how to use AnkiConnect to add flashcards with images and audio.

Usage:
- Ensure Anki is running and AnkiConnect is installed.
- Run the script to add a sample note to your Anki collection.
"""

import base64
import html
import os

import requests

from fluentai.utils.logger import logger


class AnkiConnect:
    # URL and version for AnkiConnect
    URL = "http://localhost:8765/"
    VERSION = 6

    def invoke(self, action, params=None):
        """
        Invoke an AnkiConnect action with optional parameters.
        Returns the result or raises an exception if there's an error.
        """
        payload = {"action": action, "version": self.VERSION}
        if params:
            payload["params"] = params
        response = requests.post(self.URL, json=payload).json()
        if len(response) != 2:
            raise Exception("Unexpected number of fields in response")
        if "error" not in response or "result" not in response:
            raise Exception("Response is missing required fields")
        if response["error"] is not None:
            raise Exception(response["error"])
        return response["result"]

    def get_deck_names(self):
        """
        Retrieves a list of deck names from Anki.
        """
        try:
            return self.invoke("deckNames")
        except Exception:
            logger.error("Could not establish connection with Anki")
            logger.error(
                "Please make sure Anki is running and AnkiConnect is installed"
            )

    def store_media_file(self, src_file_path, word):
        """
        Stores a media file in Anki's collection.
        Returns the filename used in Anki.
        """
        action = "storeMediaFile"
        sanitized_word = "".join(
            [c for c in word if c.isalnum() or c in (" ", "-")]
        ).rstrip()
        ext = os.path.splitext(src_file_path)[1]
        dst = f"{sanitized_word}{ext}"

        with open(src_file_path, "rb") as f:
            b64_output = base64.b64encode(f.read()).decode("utf-8")
        params = {"filename": dst, "data": b64_output}

        self.invoke(action, params)
        return dst

    @staticmethod
    def format_notes(notes):
        """
        Formats notes by escaping HTML and converting line breaks.
        """
        html_notes = "<br>".join(html.escape(notes.strip()).split("\n"))
        return f"<div>{html_notes}</div>"

    def add_note(
        self,
        deck_name,
        word,
        answer,
        image_paths,
        word_usage,
        notes,
        recording_file_path,
        ipa_text,
        test_spelling,
    ):
        """
        Adds a new note to the specified Anki deck with provided fields.
        Returns the note ID.
        """
        stored_images = [
            self.store_media_file(image_path, f"{word}-{i}")
            for i, image_path in enumerate(image_paths)
        ]

        picture_field = "".join(f'<img src="{img}">' for img in stored_images)

        escaped_usage = html.escape(word_usage.replace("&", "&amp;"))
        formatted_notes = self.format_notes(notes)

        gender_notes_field = escaped_usage + formatted_notes + answer

        pronunciation_field = ipa_text

        if recording_file_path:
            stored_audio_filename = self.store_media_file(recording_file_path, word)
            pronunciation_field += f"[sound:{stored_audio_filename}]"

        test_spelling = "y" if test_spelling else ""

        params = {
            "note": {
                "deckName": deck_name,
                "modelName": "2. Picture Words",  # Could be moved to a config
                "fields": {
                    "Word": word,
                    "Answer": answer,
                    "Picture": picture_field,
                    "Gender, Personal Connection, Extra Info (Back side)": gender_notes_field,
                    "Pronunciation (Recording and/or IPA)": pronunciation_field,
                    "Test Spelling? (y = yes, blank = no)": test_spelling,
                },
                "tags": [],
            }
        }

        note_id = self.invoke("addNote", params)
        return note_id


def main():
    """
    Main function to test adding a note via AnkiConnect.
    """
    # Create an instance of AnkiConnect
    anki = AnkiConnect()

    # Test data
    deck_name = "Test Deck"
    word = "Example"
    answer = "Answer: Example"
    image_paths = []  # Update with actual paths if you have images
    word_usage = "This is an example of how the word 'example' is used in a sentence."
    notes = "These are sample notes for testing purposes."
    recording_file_path = None  # Update with actual path if you have an audio file
    ipa_text = ""  # Update with IPA pronunciation if available
    test_spelling = True

    # Add the note
    try:
        note_id = anki.add_note(
            deck_name,
            word,
            answer,
            image_paths,
            word_usage,
            notes,
            recording_file_path,
            ipa_text,
            test_spelling,
        )
        logger.info(f"Note added successfully with ID: {note_id}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
