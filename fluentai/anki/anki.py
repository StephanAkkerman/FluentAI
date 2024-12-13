import base64
import html
import os

import requests

from fluentai.services.card_gen.constants.config import config
from fluentai.services.card_gen.utils.logger import logger


class AnkiConnect:
    # URL and version for AnkiConnect
    URL = "http://localhost:8765/"
    VERSION = 6

    def invoke(self, action: str, params: dict = None):
        """Invoke an AnkiConnect action with optional parameters.

        Parameters
        ----------
        action : str
            The action to invoke. See the AnkiConnect API documentation for a list of actions.
        params : dict, optional
            The parameters associated with this action, by default None

        Returns
        -------
        dict
            The result of the action.

        Raises
        ------
        Exception
            If the response does not contain the expected fields.
        Exception
            If the response contains an error message.
        Exception
            If the response contains an unexpected number of fields.
        """
        payload = {"action": action, "version": self.VERSION}

        # Add parameters if they exist
        if params:
            payload["params"] = params

        try:
            response = requests.post(self.URL, json=payload).json()
        except requests.exceptions.ConnectionError:
            logger.error(
                """Could not establish connection with Anki. 
This can be caused by two things:
1. Anki is not running
2. Anki does not have the Anki-Connect plugin: https://foosoft.net/projects/anki-connect/."""
            )
            return

        if len(response) != 2:
            logger.error("Unexpected number of fields in response")
            logger.error(response)
            return

        if "error" not in response or "result" not in response:
            logger.error("Response is missing required fields")
            logger.error(response)
            return

        if response["error"] is not None:
            if "model was not found" in response["error"]:
                logger.error(
                    f"The {response['error']}. Please ensure the model exists in Anki by installing the template located at FluentAI/deck/FluentAI.apkg. You can do so by dragging and dropping it into Anki."
                )
            if "deck was not found" in response["error"]:
                logger.error(
                    f"The following {response['error']}. Please ensure a deck with that name exists in Anki."
                )
            else:
                logger.error(response["error"])
            return

        return response["result"]

    def get_deck_names(self) -> list[str]:
        """Retrieves a list of deck names from Anki.

        Returns
        -------
        list[str]
            List of deck names.
        """
        try:
            return self.invoke("deckNames")
        except Exception:
            logger.error("Could not establish connection with Anki")
            logger.error(
                "Please make sure Anki is running and AnkiConnect is installed"
            )

    def store_media_file(self, src_file_path: str, word: str) -> str:
        """Stores a media file in Anki's collection.

        Parameters
        ----------
        src_file_path : str
            The path to the file to store.
        word : str
            The word to use as the filename in Anki.

        Returns
        -------
        str
            Returns the filename used in Anki.
        """
        # Sanitize the word to remove special characters
        sanitized_word = "".join(
            [c for c in word if c.isalnum() or c in (" ", "-")]
        ).rstrip()

        # Get the file extension
        ext = os.path.splitext(src_file_path)[1]
        dst = f"{sanitized_word}{ext}"

        # Encode the file as base64
        with open(src_file_path, "rb") as f:
            b64_output = base64.b64encode(f.read()).decode("utf-8")

        self.invoke("storeMediaFile", {"filename": dst, "data": b64_output})

        return dst

    @staticmethod
    def format_notes(notes: str) -> str:
        """Formats notes by escaping HTML and converting line breaks.

        Parameters
        ----------
        notes : str
            The notes to format.

        Returns
        -------
        str
            The formatted notes.
        """
        html_notes = "<br>".join(html.escape(notes.strip()).split("\n"))
        return f"<div>{html_notes}</div>"

    def add_note(
        self,
        word: str,
        answer: str,
        image_paths: list[str],
        word_usage: str,
        notes: str,
        recording_file_path: str,
        ipa_text: str,
        test_spelling: bool,
        deck_name: str = config["DECK_NAME"],
    ) -> int:
        """Adds a new note to the specified Anki deck with provided fields.

        Parameters
        ----------
        deck_name : str
            The name of the deck to add the note to.
        word : str
            The word to add to the note (front side).
        answer : str
            The answer to the word (back side).
        image_paths : list[str]
            The paths to the images to add to the note (front side).
        word_usage : str
            The usage of the word in a sentence (back side).
        notes : str
            Additional notes for the word (back side).
        recording_file_path : str
            The path to the recording file.
        ipa_text : str
            The IPA pronunciation of the word (back side).
        test_spelling : bool
            Whether to test spelling for the word.

        Returns
        -------
        int
            Returns the note ID
        """
        # Store the images in Anki
        stored_images = [
            self.store_media_file(image_path, f"{word}-{i}")
            for i, image_path in enumerate(image_paths)
        ]

        # Create the picture field with the stored images
        picture_field = "".join(f'<img src="{img}">' for img in stored_images)

        # Format
        escaped_usage = html.escape(word_usage.replace("&", "&amp;"))
        formatted_notes = self.format_notes(notes)
        gender_notes_field = escaped_usage + formatted_notes + answer

        pronunciation_field = ipa_text

        # Store the recording in Anki
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

        return self.invoke("addNote", params)


def main():
    """
    Main function to test adding a note via AnkiConnect.
    """
    # Create an instance of AnkiConnect
    anki = AnkiConnect()

    # Add the note
    note_id = anki.add_note(
        word="Example",
        answer="Answer: Example",
        image_paths=[],  # Update with actual paths if you have images
        word_usage="This is an example of how the word 'example' is used in a sentence.",
        notes="These are sample notes for testing purposes.",
        recording_file_path=None,  # Update with actual path if you have an audio file
        ipa_text="",  # Update with IPA pronunciation if available
        test_spelling=True,
    )
    if note_id:
        logger.info(f"Note added successfully with ID: {note_id}")


def get_models():
    anki = AnkiConnect()
    action = "modelNames"
    response = anki.invoke(action)
    print(response)


def create_model(lang_code: str = "id_ID"):
    anki = AnkiConnect()
    action = "createModel"
    params = {
        "modelName": f"FluentAI Model ({lang_code})",
        "inOrderFields": [
            "Word",
            "Picture",
            "Gender, Personal Connection, Extra Info (Back side)",
            "Pronunciation (Recording and/or IPA)",
            "Test Spelling? (y = yes, blank = no)",
        ],
        "css": """
        .card {
        font-family: arial;
        font-size: 30px;
        text-align: center;
        color: black;
        background-color: white;
        }

        .card1 { background-color: #FFFFFF; }
        .card2 { background-color: #FFFFFF; }
        """,
        "isCloze": False,
        "cardTemplates": [
            {
                "Name": f"Word - Mnemonic ({lang_code})",
                "Front": "{{tts " + lang_code + ":Word}}\n<br>\n{{Word}}\n\n",
                "Back": '{{FrontSide}}\n\n<hr id=answer>\n{{Picture}}\n\n{{#Pronunciation (Recording and/or IPA)}}\n<br>\n<font color=blue>{{Pronunciation (Recording and/or IPA)}}</font>{{/Pronunciation (Recording and/or IPA)}}<br>\n\n\n<span style="color:grey">\n{{Gender, Personal Connection, Extra Info (Back side)}}</span>\n<br><br>\n',
            },
            {
                "Name": f"Mnemonic - Word ({lang_code})",
                "Front": "{{Picture}}<br><br>\n\n<font color=red></font><br><br>\n<font color=red></font><br><br>\n",
                "Back": "{{FrontSide}}\n\n<hr id=answer>\n\n{{tts "
                + lang_code
                + ':Word}}\n<br>\n<span style="font-size:1.5em;">{{Word}}</span><br>\n\n\n{{#Pronunciation (Recording and/or IPA)}}<br><font color=blue>{{Pronunciation (Recording and/or IPA)}}</font>{{/Pronunciation (Recording and/or IPA)}}\n\n{{#Gender, Personal Connection, Extra Info (Back side)}}<br><font color=grey>{{Gender, Personal Connection, Extra Info (Back side)}}</font>{{/Gender, Personal Connection, Extra Info (Back side)}}\n\n\n<span style="">',
            },
            {
                "Name": f"Mnemonic - Spelling ({lang_code})",
                "Front": "{{#Test Spelling? (y = yes, blank = no)}}\nSpell this word: <br><br>\n{{Picture}}<br>\n\n{{#Pronunciation (Recording and/or IPA)}}<br><font color=blue>{{Pronunciation (Recording and/or IPA)}}</font>{{/Pronunciation (Recording and/or IPA)}}\n<br>\n\n{{/Test Spelling? (y = yes, blank = no)}}\n\n\n",
                "Back": '<span style="font-size:1.5em;">{{Word}}</span><br><br>\n\n\n{{Picture}}<br>\n\n<span style="color:grey;">{{Gender, Personal Connection, Extra Info (Back side)}}</span>\n',
            },
        ],
    }
    anki.invoke(action, params)


if __name__ == "__main__":
    # main()
    # get_models()
    create_model()
    # anki = AnkiConnect()
    # action = "modelTemplates"
    # params = {"modelName": "2. Picture Words"}
    # response = anki.invoke(action, params)
    # print(response)
