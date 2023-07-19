# Data for Transphoner #

*ipa.csv* - List of IPA symbols and basic characteristics

*ipaFeatures.csv* - Features for IPA symbols (generated from `ipa.csv` using `PhoneFeatures.makeFeaturesFile`)

*phoneInventory.csv* - List of valid phones for each language

## Phonetic Similarity ##
*dist.asjp-phonPattern.consonants.csv* - ASJP database phonetic patterns (see http://email.eva.mpg.de/~wichmann/languages.htm)

*dist.asjp-phonPattern.vowels.csv* - See above

*dist.ipa-mielke-acoustic.csv* - Phonetic distance by Mielke et al. (see http://aix1.uottawa.ca/~jmielke/)

## Dictionaries ###

### English ###
*twl.ipa.tsv.bz2* - English IPA pronunciations scraped from [dictionary.com](http://www.dictionary.com) for the official scrabble TWL06 wordlist (`data\wordlists\scrabble\TWL06.txt`)
- may also include some words from `en.freqList.50k.txt.bz2`
- scraping code at `fuzzybox\carmen\src\main\java\dataset\transphoner\DictionaryComScraper.java`
- scraped result at data share: `wordlists\scrabble\twl.all.ipa.tsv.bz2`
- final cleaning using `DictCleaners.englishCleaner`

*twl.glosses.tsv.bz2* - English glosses for `twl.ipa.tsv.bz2` scrapped from [dictionary.com](http://www.dictionary.com)

*mrc2.dct.bz2* - Machine Readable Corpus of psycholinguistic data from [http://www.psych.rl.ac.uk/]

*en.aoa.csv.bz2* - Age of acquisition data

*enwikitionary-ipa.tsv.bz2* - English IPA pronunciations from Wikitionary (quality is poor)

*en.syllables.yml* - English syllabification rules (from Ruby syllabifier)

*en.freqList.50k.txt.bz2* - Most frequent English words

### French ###
*fr-combined.ipa.tsv.bz2* - Combined French IP pronunciations from freedict and Wikitionary

*fr-freedict-ipa.tsv.bz2* - French IPA pronunciations from freedict

*frwikitionary-ipa.tsv.bz2* - French IPA pronunciations from Wikitionary (quality is poor)

*fr.freqList.50k.txt.bz2* - Most frequent French words

*fr.syllables.yml* - French syllabification rules (constructed from French prounciation rules)

*fr-imageability.tsv* - French words imageability downloaded from omninet

### German ###

*de-freedict-ipa.tsv.bz2* - German IPA pronunciations from freedict

*de.freqList.50k.txt.bz2* - Most frequent German words

### Chinese ###

*zh-cedict-full.tsv.bz2* - Chinese (Mandarin) IPA pronunciations from CEDICT

*zh.freqList.50k.txt.bz2* - Most frequent Chinese words

*zh.tones.csv* - Mandarin tones to IPA symbols

### Japanese ###

*ja-edict-full.tsv.bz2* - Japanese IPA pronunciations from EDICT

*ja.freqList.45k.txt.bz2* - Most frequent Japanese words

*katakana2ipa.csv* - Translating from Katakana to IPA

## Evaluation ##
*evaluation-ellis-keywords.csv* - List of 36 German to English words for evaluating the effectiveness of keywords for foreign language learning from Ellis and Beacon

*evaluation-slanguage-books.csv* - List of suggested keywords/keyphrase from [slanguage.com](http://www.slanguage.com/)

*evaluation-pressley-keywords.csv* List of 20 Spanish to English words from Pressley

*wordsim353.csv* Similarity of 353 English words from http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/


