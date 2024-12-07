# Supported Languages

This document lists the languages that are currently supported to learn and the languages that you can learn from (i.e. the languages that are used to teach the supported languages).

## Learnable Languages

This list is dependent on the [charsiu/g2p model](https://huggingface.co/charsiu/g2p_multilingual_byT5_small_100) you can find the original list [here](https://docs.google.com/spreadsheets/d/1y7kisk-UZT9LxpQB0xMIF4CkxJt0iYJlWAnyj6azSBE) and [google translate](https://translate.google.com/) for translation.

The codes are the ISO-639-3 codes with some modifications to distinguish local dialects/variants.

The languages in the table below are currently supported for learning. The languages with a check mark in the `g2p` column are supported for grapheme-to-phoneme conversion. The languages with a check mark in the `translation` column are supported for translation. If the language does not have a checkmark for translation, it is supported but the performance may not be as good as the languages with a checkmark.

The `vocab guide` column indicates whether a vocabulary guide is available for the language. The vocabulary guide is a list of the most common words in the language, which can be used to learn the language. You can find the original list [here](https://huggingface.co/datasets/StephanAkkerman/frequency-words-2018#supported-languages).

| Language                  | Code      | g2p | translation | vocab guide |
| ------------------------- | --------- | --- | ----------- | ----------- |
| Adyghe                    | ady       | ✅  |             |             |
| Afrikaans                 | afr       | ✅  | ✅          | ✅          |
| Albanian                  | sqi       | ✅  | ✅          | ✅          |
| Amharic                   | amh       | ✅  | ✅          |             |
| Arabic                    | ara       | ✅  | ✅          | ✅          |
| Aragonese                 | arg       | ✅  |             |             |
| Armenian (Eastern)        | arm-e     | ✅  | ✅          | ✅          |
| Armenian (Western)        | arm-w     | ✅  | ✅          | ✅          |
| Azerbaijani               | aze       | ✅  | ✅          |             |
| Bashkir                   | bak       | ✅  | ✅          |             |
| Basque                    | eus       | ✅  | ✅          | ✅          |
| Belarussian               | bel       | ✅  | ✅          |             |
| Bengali                   | ben       | ✅  | ✅          | ✅          |
| Bosnian                   | bos       | ✅  | ✅          | ✅          |
| Bulgarian                 | bul       | ✅  | ✅          | ✅          |
| Burmese                   | bur       | ✅  | ✅          | ✅          |
| Catalan                   | cat       | ✅  | ✅          |             |
| Classical Syriac          | syc       | ✅  |             |             |
| Chinese (Cantonese)       | yue       | ✅  | ✅          |             |
| Chinese (Traditional)     | zho-t     | ✅  | ✅          | ✅          |
| Chinese (Simplified)      | zho-s     | ✅  | ✅          | ✅          |
| Chinese (Min)             | min       | ✅  |             |             |
| Czech                     | cze       | ✅  | ✅          | ✅          |
| Danish                    | dan       | ✅  | ✅          | ✅          |
| Dutch                     | dut       | ✅  | ✅          | ✅          |
| Egyptian                  | egy       | ✅  |             |             |
| English (UK)              | eng-uk    | ✅  | ✅          | ✅          |
| English (US)              | eng-us    | ✅  | ✅          | ✅          |
| Esperanto                 | epo       | ✅  | ✅          |             |
| Estonian                  | est       | ✅  | ✅          | ✅          |
| Finnish                   | fin       | ✅  | ✅          | ✅          |
| French                    | fra       | ✅  | ✅          | ✅          |
| French (Quebec)           | fra-qu    | ✅  | ✅          | ✅          |
| Gaelic                    | gla       | ✅  | ✅          |             |
| Galician                  | glg       | ✅  | ✅          | ✅          |
| Georgian                  | geo       | ✅  | ✅          | ✅          |
| German                    | ger       | ✅  | ✅          | ✅          |
| Greek                     | gre       | ✅  | ✅          | ✅          |
| Greek (Ancient)           | grc       | ✅  |             |             |
| Guarani                   | grn       | ✅  | ✅          |             |
| Gujarati                  | guj       | ✅  | ✅          |             |
| Hindi                     | hin       | ✅  | ✅          | ✅          |
| Hungarian                 | hun       | ✅  | ✅          | ✅          |
| Icelandic                 | ice       | ✅  | ✅          | ✅          |
| Ido                       | ido       | ✅  |             |             |
| Indonesian                | ind       | ✅  | ✅          | ✅          |
| Interlingua               | ina       | ✅  |             | ✅          |
| Irish                     | gle       | ✅  | ✅          |             |
| Italian                   | ita       | ✅  | ✅          |             |
| Jamaican Creole           | jam       | ✅  | ✅          |             |
| Japanese                  | jpn       | ✅  | ✅          | ✅          |
| Kazakh                    | kaz       | ✅  | ✅          | ✅          |
| Khmer                     | khm       | ✅  | ✅          |             |
| Korean                    | kor       | ✅  | ✅          | ✅          |
| Kurdish                   | kur       | ✅  | ✅          |             |
| Latin (Classical)         | lat-clas  | ✅  | ✅          |             |
| Latin (Ecclesiastical)    | lat-eccl  | ✅  | ✅          |             |
| Lithuanian                | lit       | ✅  | ✅          | ✅          |
| Luxembourgish             | ltz       | ✅  | ✅          |             |
| Macedonian                | mac       | ✅  | ✅          | ✅          |
| Maltese                   | mlt       | ✅  | ✅          |             |
| Middle English            | enm       | ✅  |             |             |
| Northeastern Thai         | tts       | ✅  |             |             |
| Northern Sami             | sme       | ✅  | ✅          |             |
| Norwegian Bokmål          | nob       | ✅  | ✅          | ✅          |
| Oriya                     | ori       | ✅  | ✅          |             |
| Old English               | ang       | ✅  |             |             |
| Papiamento                | pap       | ✅  | ✅          |             |
| Persian                   | fas       | ✅  | ✅          | ✅          |
| Polish                    | pol       | ✅  | ✅          | ✅          |
| Portuguese (Portugal)     | por-po    | ✅  | ✅          | ✅          |
| Portuguese (Brazil)       | por-bz    | ✅  | ✅          | ✅          |
| Romanian                  | ron       | ✅  | ✅          | ✅          |
| Russian                   | rus       | ✅  | ✅          | ✅          |
| Sanskrit                  | san       | ✅  | ✅          |             |
| Serbian                   | srp       | ✅  | ✅          | ✅          |
| Serbo-Croatian (Latin)    | hbs-latn  | ✅  | ✅          | ✅          |
| Serbo-Croatian (Cyrillic) | hbs-cyrl  | ✅  | ✅          |             |
| Sindhi                    | snd       | ✅  | ✅          |             |
| Slovak                    | slo       | ✅  | ✅          | ✅          |
| Slovenian                 | slv       | ✅  | ✅          | ✅          |
| Spanish                   | spa       | ✅  | ✅          | ✅          |
| Spanish (Latin America)   | spa-latin | ✅  | ✅          |             |
| Spanish (Mexico)          | spa-me    | ✅  | ✅          |             |
| Swahili                   | swa       | ✅  | ✅          |             |
| Swedish                   | swe       | ✅  | ✅          | ✅          |
| Tagalog                   | tgl       | ✅  | ✅          | ✅          |
| Tamil                     | tam       | ✅  | ✅          | ✅          |
| Tatar                     | tat       | ✅  | ✅          |             |
| Thai                      | tha       | ✅  | ✅          | ✅          |
| Turkish                   | tur       | ✅  | ✅          | ✅          |
| Turkmen                   | tuk       | ✅  | ✅          |             |
| Ukrainian                 | ukr       | ✅  | ✅          | ✅          |
| Vietnamese (Northern)     | vie-n     | ✅  | ✅          | ✅          |
| Vietnamese (Central)      | vie-c     | ✅  | ✅          | ✅          |
| Vietnamese (Southern)     | vie-s     | ✅  | ✅          | ✅          |
| Welsh (North)             | wel-nw    | ✅  | ✅          |             |
| Welsh (South)             | wel-sw    | ✅  | ✅          |             |

## Native languages

We currently only support learning foreign languages in English.
