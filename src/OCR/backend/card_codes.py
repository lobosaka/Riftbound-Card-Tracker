from dataclasses import dataclass
import re
import sys
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from services.card_data import load_cards_for_code_index


@dataclass(frozen=True)
class CardRecord:
    id: str
    name: str
    public_code: str


class CardRepository:
    SET_CODE_PATTERN = re.compile(r"^[A-Z]{3}$")
    CANONICAL_CODE_PATTERN = re.compile(
        r"^(?:(?P<set>[A-Z]{3})-?)?(?P<number>\d{1,3})(?P<suffix>[A-Z*]?)/(?P<total>\d{3})$"
    )

    def __init__(self):
        self.card_code_index = self._build_card_code_index()
        self.known_set_codes = {
            code for code in self.card_code_index if self.SET_CODE_PATTERN.fullmatch(code)
        }

    def canonicalize_code(self, code):
        match = self.CANONICAL_CODE_PATTERN.fullmatch(self._clean_code(code))
        if not match:
            return self._clean_code(code)

        set_code = match.group("set")
        number = int(match.group("number"))
        suffix = match.group("suffix")
        total = match.group("total")
        prefix = f"{set_code}-" if set_code else ""
        return f"{prefix}{number:03d}{suffix}/{total}"

    def verify_code(self, code):
        canonical = self.canonicalize_code(code)
        matches = self.card_code_index.get(canonical, [])
        return matches or self.card_code_index.get(self._code_suffix(canonical), [])

    def format_match_status(self, matches):
        if len(matches) == 1:
            match = matches[0]
            return f"VERIFIED -> {match.public_code} ({match.name})"
        if len(matches) > 1:
            joined = ", ".join(f"{match.public_code} ({match.name})" for match in matches)
            return f"AMBIGUOUS -> {joined}"
        return "NOT FOUND IN DB"

    def _build_card_code_index(self):
        index = {}
        for row in load_cards_for_code_index():
            public_code = row["publicCode"]
            if not public_code:
                continue

            record = CardRecord(
                id=row["id"],
                name=row["name"],
                public_code=public_code,
            )
            for variant in self._build_code_variants(public_code, row["collectorNumber"]):
                index.setdefault(variant, []).append(record)
        return index

    def _build_code_variants(self, public_code, collector_number):
        canonical = self.canonicalize_code(public_code)
        suffix = self._code_suffix(public_code)
        variants = {canonical, self.canonicalize_code(suffix)}

        if collector_number and "/" in suffix:
            total = suffix.split("/", 1)[1]
            variants.add(self.canonicalize_code(f"{collector_number}/{total}"))

        set_code = self._set_prefix(canonical)
        if set_code:
            variants.add(set_code)
        return variants

    @staticmethod
    def _clean_code(code):
        return (
            code.upper().strip().replace(" ", "").replace("•", "").replace("|", "/")
        )

    @staticmethod
    def _code_suffix(code):
        return code.split("-", 1)[1] if "-" in code else code

    @staticmethod
    def _set_prefix(code):
        return code.split("-", 1)[0] if "-" in code else None


class CardCodeParser:
    FULL_CODE_PATTERN = re.compile(
        r"(?P<set>[A-Z]{3})[-\s•.]*?(?P<number>\d{1,3})(?P<suffix>[A-Z*]?)[/\|](?P<total>\d{3})"
    )
    SUFFIXLESS_CODE_PATTERN = re.compile(
        r"(?P<number>\d{1,3})(?P<suffix>[A-Z*]?)[/\|](?P<total>\d{3})"
    )
    SHORT_CODE_PATTERN = re.compile(r"[A-Z]\d{2,3}")
    NORMALIZATION_TABLE = str.maketrans(
        {
            "|": "/",
            "\\": "/",
            "•": None,
            "·": None,
            ".": None,
            ":": None,
            "I": "1",
            "L": "1",
            "O": "0",
            "Q": "0",
            "X": "*",
        }
    )

    def __init__(self, repository):
        self.repository = repository

    def normalize_text(self, text):
        return text.upper().strip().replace(" ", "").replace("\n", "").translate(
            self.NORMALIZATION_TABLE
        )

    def normalize_candidate(self, text):
        cleaned = self.normalize_text(text)
        for pattern in (self.FULL_CODE_PATTERN, self.SUFFIXLESS_CODE_PATTERN):
            match = pattern.search(cleaned)
            if match:
                return self.repository.canonicalize_code(match.group(0))
        short_match = self.SHORT_CODE_PATTERN.search(cleaned)
        return short_match.group(0) if short_match else cleaned

    def extract_candidates(self, lines):
        normalized_lines = [self.normalize_text(line) for line in lines if line]
        candidates = []

        for text in [*normalized_lines, " ".join(normalized_lines)]:
            for match in self.FULL_CODE_PATTERN.finditer(text):
                code = match.group(0)
                if match.group("set") in self.repository.known_set_codes:
                    candidates.append(self.repository.canonicalize_code(code))
            for match in self.SUFFIXLESS_CODE_PATTERN.finditer(text):
                candidates.append(self.repository.canonicalize_code(match.group(0)))
            if text in self.repository.known_set_codes:
                candidates.append(text)

        if not candidates:
            candidates = [self.normalize_candidate(line) for line in normalized_lines]

        return self._unique_preserving_order(candidates)

    def score_candidate(self, candidate):
        matches = len(self.repository.verify_code(candidate))
        db_score = 3 if matches == 1 else 1 if matches > 1 else 0
        canonical = self.repository.CANONICAL_CODE_PATTERN.fullmatch(candidate)
        if canonical and "-" in candidate:
            match_score = 4
        elif canonical:
            match_score = 3
        elif re.fullmatch(r"[A-Z]\d{2}", candidate):
            match_score = 2
        elif self.SHORT_CODE_PATTERN.fullmatch(candidate):
            match_score = 1
        else:
            match_score = 0

        digits = sum(char.isdigit() for char in candidate)
        penalty = sum(char not in "0123456789/*ABCDEFGHIJKLMNOPQRSTUVWXYZ" for char in candidate)
        return (db_score, match_score, "*" in candidate, digits, -penalty)

    @staticmethod
    def _unique_preserving_order(values):
        seen = set()
        unique_values = []
        for value in values:
            if value and value not in seen:
                seen.add(value)
                unique_values.append(value)
        return unique_values
