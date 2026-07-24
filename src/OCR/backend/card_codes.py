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
    def __init__(self):
        """Lädt beim Erstellen alle Karten in einen Suchindex für schnelle Code-Lookups."""
        self.set_code_pattern = re.compile(r"^[A-Z]{3}$")
        self.canonical_code_pattern = re.compile(
            r"^(?:(?P<set>[A-Z]{3})-?)?(?P<number>\d{1,3})(?P<suffix>[A-Z*]?)/(?P<total>\d{3})$"
        )
        self.card_code_index = self._build_card_code_index()
        self.known_set_codes = {
            code for code in self.card_code_index if self.set_code_pattern.fullmatch(code)
        }

    def _build_card_code_index(self):
        """Erstellt einen Index, damit verschiedene Schreibweisen schnell zur Karte führen."""
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
            canonical = self.canonicalize_code(public_code)
            suffix = public_code.split("-", 1)[1] if "-" in public_code else public_code
            variants = {canonical, self.canonicalize_code(suffix)}

            collector_number = row["collectorNumber"]
            if collector_number and "/" in suffix:
                total = suffix.split("/", 1)[1]
                variants.add(self.canonicalize_code(f"{collector_number}/{total}"))

            set_code = canonical.split("-", 1)[0] if "-" in canonical else None
            if set_code:
                variants.add(set_code)

            for variant in variants:
                index.setdefault(variant, []).append(record)
        return index

    def canonicalize_code(self, code):
        """Bringt einen Karten-Code in ein einheitliches Standardformat."""
        match = self.canonical_code_pattern.fullmatch(self._clean_code(code))
        if not match:
            return self._clean_code(code)

        set_code = match.group("set")
        number = int(match.group("number"))
        suffix = match.group("suffix")
        total = match.group("total")
        prefix = f"{set_code}-" if set_code else ""
        return f"{prefix}{number:03d}{suffix}/{total}"

    @staticmethod
    def _clean_code(code):
        """Entfernt typische OCR-Störungen und vereinheitlicht die Grundschreibweise eines Codes."""
        return (
            code.upper().strip().replace(" ", "").replace("•", "").replace("|", "/")
        )

    def verify_code(self, code):
        """Prüft, ob der Code im Index existiert, und gibt passende Karten zurück."""
        canonical = self.canonicalize_code(code)
        matches = self.card_code_index.get(canonical, [])
        suffix = canonical.split("-", 1)[1] if "-" in canonical else canonical
        return matches or self.card_code_index.get(suffix, [])

    def format_match_status(self, matches):
        """Formatiert Suchergebnisse als lesbaren Status-Text für Debugging oder Ausgabe."""
        if len(matches) == 1:
            match = matches[0]
            return f"VERIFIED -> {match.public_code} ({match.name})"
        if len(matches) > 1:
            joined = ", ".join(f"{match.public_code} ({match.name})" for match in matches)
            return f"AMBIGUOUS -> {joined}"
        return "NOT FOUND IN DB"


class CardCodeParser:
    def __init__(self, repository):
        """Verknüpft den Parser mit dem Repository, um Codes direkt validieren zu können."""
        self.repository = repository
        self.full_code_pattern = re.compile(
            r"(?P<set>[A-Z]{3})[-\s•.]*?(?P<number>\d{1,3})(?P<suffix>[A-Z*]?)[/\|](?P<total>\d{3})"
        )
        self.suffixless_code_pattern = re.compile(
            r"(?P<number>\d{1,3})(?P<suffix>[A-Z*]?)[/\|](?P<total>\d{3})"
        )
        self.short_code_pattern = re.compile(r"[A-Z]\d{2,3}")
        self.normalization_table = str.maketrans(
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

    def main(self, lines):
        """
        Input:
            lines (list[str]): OCR-Textzeilen, aus denen Karten-Code-Kandidaten gelesen werden.

        Output:
            dict mit:
            - candidates (list[str]): extrahierte Kandidaten in Reihenfolge ihres Auftretens
            - scored_candidates (list[dict]): Kandidaten mit Score, DB-Matches und Status
            - best_candidate (str): bestbewerteter Kandidat oder leerer String
        """
        candidates = self.extract_candidates(lines)
        scored_candidates = []
        for candidate in candidates:
            matches = self.repository.verify_code(candidate)
            scored_candidates.append(
                {
                    "candidate": candidate,
                    "score": self.score_candidate(candidate),
                    "matches": matches,
                    "status": self.repository.format_match_status(matches),
                }
            )

        best_candidate = ""
        if candidates:
            best_candidate = max(candidates, key=self.score_candidate)

        return {
            "candidates": candidates,
            "scored_candidates": scored_candidates,
            "best_candidate": best_candidate,
        }

    def extract_candidates(self, lines):
        """Sammelt aus mehreren OCR-Zeilen alle plausiblen Karten-Code-Kandidaten."""
        normalized_lines = [self.normalize_text(line) for line in lines if line]
        candidates = []

        for text in [*normalized_lines, " ".join(normalized_lines)]:
            for match in self.full_code_pattern.finditer(text):
                code = match.group(0)
                if match.group("set") in self.repository.known_set_codes:
                    candidates.append(self.repository.canonicalize_code(code))
            for match in self.suffixless_code_pattern.finditer(text):
                candidates.append(self.repository.canonicalize_code(match.group(0)))
            if text in self.repository.known_set_codes:
                candidates.append(text)

        if not candidates:
            for text in normalized_lines:
                candidate = text
                for pattern in (self.full_code_pattern, self.suffixless_code_pattern):
                    match = pattern.search(text)
                    if match:
                        candidate = self.repository.canonicalize_code(match.group(0))
                        break
                else:
                    short_match = self.short_code_pattern.search(text)
                    if short_match:
                        candidate = short_match.group(0)
                candidates.append(candidate)

        unique_candidates = []
        seen = set()
        for candidate in candidates:
            if candidate and candidate not in seen:
                seen.add(candidate)
                unique_candidates.append(candidate)

        return unique_candidates

    def normalize_text(self, text):
        """Bereinigt OCR-Text und ersetzt typische Verwechslungen durch eine stabilere Schreibweise."""
        return text.upper().strip().replace(" ", "").replace("\n", "").translate(
            self.normalization_table
        )

    def score_candidate(self, candidate):
        """Bewertet einen Kandidaten danach, wie gut er wie ein echter Karten-Code aussieht."""
        matches = len(self.repository.verify_code(candidate))
        db_score = 3 if matches == 1 else 1 if matches > 1 else 0
        canonical = self.repository.canonical_code_pattern.fullmatch(candidate)
        if canonical and "-" in candidate:
            match_score = 4
        elif canonical:
            match_score = 3
        elif re.fullmatch(r"[A-Z]\d{2}", candidate):
            match_score = 2
        elif self.short_code_pattern.fullmatch(candidate):
            match_score = 1
        else:
            match_score = 0

        digits = sum(char.isdigit() for char in candidate)
        penalty = sum(char not in "0123456789/*ABCDEFGHIJKLMNOPQRSTUVWXYZ" for char in candidate)
        return (db_score, match_score, "*" in candidate, digits, -penalty)
