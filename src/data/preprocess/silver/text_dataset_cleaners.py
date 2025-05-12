from __future__ import annotations

import re
import string

from abc import ABC, abstractmethod
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pkg_resources
import symspellpy

from symspellpy import SymSpell




class DatasetCleaner(ABC):
    def __call__(self, text: str | list[str]) -> str | list[str]:
        if isinstance(text, str):
            return self.clean_text(text)
        return self.clean_words(text)

    @abstractmethod
    def clean_text(self, text: str) -> str:
        """
        Cleans the given string
        """

    @abstractmethod
    def clean_words(self, words: list[str]) -> list[str]:
        """
        Cleans each word in a list of words
        """


class StopWordsDatasetCleaner(DatasetCleaner):
    def __init__(self) -> None:
        super().__init__()
        self.stopwords = set(stopwords.words("english"))

    def clean_text(self, text: str) -> str:
        cleaned_text = [word for word in word_tokenize(text) if word not in self.stopwords]
        return " ".join(cleaned_text)

    def clean_words(self, words: list[str]) -> list[str]:
        return [word for word in words if word not in self.stopwords]


class ToLowerCaseDatasetCleaner(DatasetCleaner):
    def clean_text(self, text: str) -> str:
        return text.lower()

    def clean_words(self, words: list[str]) -> list[str]:
        return [word.lower() for word in words]


class URLDatasetCleaner(DatasetCleaner):
    def clean_text(self, text: str) -> str:
        return re.sub(r"http\S+", "", text, flags=re.MULTILINE)

    def clean_words(self, words: list[str]) -> list[str]:
        return [self.clean_text(word) for word in words]


class PunctuationDatasetCleaner(DatasetCleaner):
    def __init__(self, punctuation: str = string.punctuation) -> None:
        super().__init__()
        self.table = str.maketrans("", "", punctuation)

    def clean_text(self, text: str) -> str:
        return " ".join(self.clean_words(text.split()))

    def clean_words(self, words: list[str]) -> list[str]:
        return [word.translate(self.table) for word in words if word.translate(self.table)]


class NonLettersDatasetCleaner(DatasetCleaner):
    def clean_text(self, text: str) -> str:
        return " ".join(self.clean_words(text.split()))

    def clean_words(self, words: list[str]) -> list[str]:
        return [word for word in words if word.isalpha()]


class NewLineCharacterDatasetCleaner(DatasetCleaner):
    def clean_text(self, text: str) -> str:
        return text.replace("\n", "")

    def clean_words(self, words: list[str]) -> list[str]:
        return [self.clean_text(word) for word in words]


class NonASCIIDatasetCleaner(DatasetCleaner):
    def clean_text(self, text: str) -> str:
        return " ".join(self.clean_words(text.split()))

    def clean_words(self, words: list[str]) -> list[str]:
        return [word for word in words if word.isascii()]


class SpellCorrectionDatasetCleaner(DatasetCleaner):
    def __init__(self, spell_correction_model: SpellCorrectionModel) -> None:
        super().__init__()
        self.spell_correction_model = spell_correction_model

    def clean_text(self, text: str) -> str:
        return self.spell_correction_model(text)

    def clean_words(self, words: list[str]) -> list[str]:
        text = " ".join(words)
        return self.clean_text(text).split()


class CharacterLimiterDatasetCleaner(DatasetCleaner):
    def __init__(self, character_limit: int = 300) -> None:
        super().__init__()
        self.character_limit = character_limit

    def clean_text(self, text: str) -> str:
        return text[: self.character_limit]

    def clean_words(self, words: list[str]) -> list[str]:
        text = " ".join(words)
        return self.clean_text(text).split()


class DatasetCleanerManager:
    def __init__(self, dataset_cleaners: dict[str, DatasetCleaner]) -> None:
        self.dataset_cleaners = dataset_cleaners

    def __call__(self, text: str | list[str]) -> str | list[str]:
        for dataset_cleaner in self.dataset_cleaners.values():
            text = dataset_cleaner(text)
        return text

class ExtraWhitespaceDatasetCleaner(DatasetCleaner):
    def clean_text(self, text: str) -> str:
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        return text.strip()

    def clean_words(self, words: list[str]) -> list[str]:
        return [word.strip() for word in words if word.strip()]


class SpellCorrectionModel:
    def __init__(
        self,
        max_dictionary_edit_distance: int = 2,
        prefix_length: int = 7,
        count_threshold: int = 1,
    ) -> None:
        self.max_dictionary_edit_distance = max_dictionary_edit_distance
        self.model = self._initialize_model(prefix_length, count_threshold)

    def _initialize_model(self, prefix_length: int, count_threshold: int) -> symspellpy.symspellpy.SymSpell:
        model = SymSpell(self.max_dictionary_edit_distance, prefix_length, count_threshold)

        dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
        bigram_dictionary_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_bigramdictionary_en_243_342.txt"
        )

        model.load_dictionary(dictionary_path, 0, 1)
        model.load_bigram_dictionary(bigram_dictionary_path, 0, 2)
        return model

    def __call__(self, text: str) -> str:
        # Split text into words while preserving apostrophes within words
        words = re.findall(r"[\w']+|[.,!?;]", text)
        corrected_words = []

        for word in words:
            if any(c.isalpha() for c in word):
                # Handle contractions and possessives separately
                if "'" in word:
                    corrected_words.append(word)
                else:
                    suggestions = self.model.lookup_compound(word, max_edit_distance=self.max_dictionary_edit_distance)
                    corrected_word = suggestions[0].term if suggestions else word
                    corrected_words.append(corrected_word)
            else:
                corrected_words.append(word)

        # Reconstruct text with proper spacing
        corrected_text = ' '.join(
            word if any(c.isalpha() for c in word) else word
            for word in corrected_words
        )
        return corrected_text

class HTMLTagsDatasetCleaner(DatasetCleaner):
    def clean_text(self, text: str) -> str:
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove excessive whitespace
        return ' '.join(text.split())

    def clean_words(self, words: list[str]) -> list[str]:
        return [self.clean_text(word) for word in words]


class EmojiDatasetCleaner(DatasetCleaner):
    def __init__(self) -> None:
        super().__init__()
        # Unicode ranges for emojis
        self.emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)

    def clean_text(self, text: str) -> str:
        return self.emoji_pattern.sub(r'', text)

    def clean_words(self, words: list[str]) -> list[str]:
        return [self.clean_text(word) for word in words]


class AbbreviationNormalizer(DatasetCleaner):
    def __init__(self, abbreviations: dict[str, str] = {
            "dr.": "doctor",
            "mr.": "mister",
            "mrs.": "missus",
            "st.": "street",
            "avg.": "average",
            # Add more abbreviations as needed
        } ) -> None:
        super().__init__()
        self.abbreviations = abbreviations

    def clean_text(self, text: str) -> str:
        text_lower = text.lower()
        for abbr, full in self.abbreviations.items():
            text_lower = text_lower.replace(abbr, full)
        return text_lower

    def clean_words(self, words: list[str]) -> list[str]:
        return self.clean_text(" ".join(words)).split()


class PunctuationNormalizer(DatasetCleaner):
    def clean_text(self, text: str) -> str:
        # Remove repeated punctuation
        text = re.sub(r'([!?,.])\1+', r'\1', text)
        # Ensure single space after punctuation
        text = re.sub(r'([!?,.])\s*', r'\1 ', text)
        return text.strip()

    def clean_words(self, words: list[str]) -> list[str]:
        return self.clean_text(" ".join(words)).split()


class DuplicateRemover(DatasetCleaner):
    def clean_text(self, text: str) -> str:
        # Remove duplicate consecutive sentences
        sentences = text.split('.')
        unique_sentences = []
        for s in sentences:
            s = s.strip()
            if s and s not in unique_sentences:
                unique_sentences.append(s)
        return '. '.join(unique_sentences)

    def clean_words(self, words: list[str]) -> list[str]:
        return list(dict.fromkeys(words))


class TokenLengthLimiter(DatasetCleaner):
    def __init__(self, max_tokens: int = 512) -> None:
        super().__init__()
        self.max_tokens = max_tokens

    def clean_text(self, text: str) -> str:
        words = text.split()
        if len(words) > self.max_tokens:
            return ' '.join(words[:self.max_tokens])
        return text

    def clean_words(self, words: list[str]) -> list[str]:
        return words[:self.max_tokens]


class SpecialTokensCleaner(DatasetCleaner):
    def __init__(self, special_tokens: set[str] = {'<context>', '<question>', '<answer>'} ) -> None:
        super().__init__()
        self.special_tokens = special_tokens

    def clean_text(self, text: str) -> str:
        # Ensure special tokens are properly formatted
        for token in self.special_tokens:
            text = text.replace(f" {token} ", f" {token}")
            text = text.replace(f"{token} ", f"{token}")
        return text

    def clean_words(self, words: list[str]) -> list[str]:
        return self.clean_text(" ".join(words)).split()


def create_default_cleaner_pipeline() -> DatasetCleanerManager:
    cleaners = {
        "html": HTMLTagsDatasetCleaner(),
        "emoji": EmojiDatasetCleaner(),
        "url": URLDatasetCleaner(),
        "newline": NewLineCharacterDatasetCleaner(),
        "lowercase": ToLowerCaseDatasetCleaner(),
        "abbreviation": AbbreviationNormalizer(),
        "punctuation_norm": PunctuationNormalizer(),
        "punctuation": PunctuationDatasetCleaner(),
        "non_ascii": NonASCIIDatasetCleaner(),
        "non_letters": NonLettersDatasetCleaner(),
        "stopwords": StopWordsDatasetCleaner(),
        "duplicates": DuplicateRemover(),
        "token_length": TokenLengthLimiter(),
        "special_tokens": SpecialTokensCleaner(),
    }
    return DatasetCleanerManager(cleaners)