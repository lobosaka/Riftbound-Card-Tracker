import re

class CardCodeParser:
    FULL_CODE_PATTERN = re.compile(
        r"(?P<set>[A-Z]{3})[-\s•.]*?(?P<number>\d{1,3})(?P<suffix>[A-Z*]?)[/\|](?P<total>\d{3})"
    )
    SUFFIXLESS_CODE_PATTERN = re.compile(
        r"(?P<number>\d{1,3})(?P<suffix>[A-Z*]?)[/\|](?P<total>\d{3})"
    )
    SHORT_CODE_PATTERN = re.compile(r"[A-Z]\d{2,3}")

    def __init__(self, repository):
        self.repository = repository

    def normalize_text(self, text):
        cleaned = text.upper().strip()
        cleaned = cleaned.replace(" ", "")
        cleaned = cleaned.replace("\n", "")
        cleaned = cleaned.replace("|", "/")
        cleaned = cleaned.replace("\\", "/")
        cleaned = cleaned.replace("•", "")
        cleaned = cleaned.replace("·", "")
        cleaned = cleaned.replace(".", "")
        cleaned = cleaned.replace(":", "")
        cleaned = cleaned.replace("I", "1")
        cleaned = cleaned.replace("L", "1")
        cleaned = cleaned.replace("O", "0")
        cleaned = cleaned.replace("Q", "0")
        cleaned = cleaned.replace("X", "*")
        return cleaned

    def normalize_candidate(self, text):
        cleaned = self.normalize_text(text)

        full_match = self.FULL_CODE_PATTERN.search(cleaned)
        if full_match:
            return self.repository.canonicalize_code(full_match.group(0))

        suffixless_match = self.SUFFIXLESS_CODE_PATTERN.search(cleaned)
        if suffixless_match:
            return self.repository.canonicalize_code(suffixless_match.group(0))

        short_match = self.SHORT_CODE_PATTERN.search(cleaned)
        return short_match.group(0) if short_match else cleaned

    def extract_candidates(self, lines):
        normalized_lines = [self.normalize_text(line) for line in lines if line]
        joined = " ".join(normalized_lines)
        candidates = []

        for text in [*normalized_lines, joined]:
            for match in self.FULL_CODE_PATTERN.finditer(text):
                if match.group("set") in self.repository.known_set_codes:
                    candidates.append(self.repository.canonicalize_code(match.group(0)))

            for match in self.SUFFIXLESS_CODE_PATTERN.finditer(text):
                candidates.append(self.repository.canonicalize_code(match.group(0)))

            if text in self.repository.known_set_codes:
                candidates.append(text)

        if not candidates:
            candidates.extend(self.normalize_candidate(line) for line in normalized_lines)

        return self._unique_preserving_order(candidates)

    def score_candidate(self, candidate):
        db_matches = len(self.repository.verify_code(candidate))
        if db_matches == 1:
            db_score = 3
        elif db_matches > 1:
            db_score = 1
        else:
            db_score = 0

        canonical_pattern = self.repository.CANONICAL_CODE_PATTERN
        if canonical_pattern.fullmatch(candidate) and "-" in candidate:
            match_score = 4
        elif canonical_pattern.fullmatch(candidate):
            match_score = 3
        elif re.fullmatch(r"[A-Z]\d{2}", candidate):
            match_score = 2
        elif self.SHORT_CODE_PATTERN.fullmatch(candidate):
            match_score = 1
        else:
            match_score = 0

        digits = sum(char.isdigit() for char in candidate)
        star_bonus = 1 if "*" in candidate else 0
        penalty = sum(char not in "0123456789/*ABCDEFGHIJKLMNOPQRSTUVWXYZ" for char in candidate)
        return (db_score, match_score, star_bonus, digits, -penalty)

    @staticmethod
    def _unique_preserving_order(values):
        seen = set()
        unique_values = []

        for value in values:
            if not value or value in seen:
                continue
            seen.add(value)
            unique_values.append(value)

        return unique_values
