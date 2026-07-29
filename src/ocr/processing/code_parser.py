from collections import defaultdict
from dataclasses import dataclass
import logging
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
    def __init__(self, logger=None):
        """Lädt beim Erstellen alle Karten in einen Suchindex für schnelle Code-Lookups."""
        self.logger = logger or logging.getLogger(__name__)
        # Erkennt Set-Codes wie UNL, OGN oder SFD.
        self.set_code_pattern = re.compile(r"^[A-Z]{3}$")
        # Erkennt vollständige Kartencodes wie UNL-001/219 oder 001A/219.
        self.canonical_code_pattern = re.compile(
            r"^(?:(?P<set>[A-Z]{3})-?)?(?P<number>\d{1,3})(?P<suffix>[A-Z*]?)/(?P<total>\d{3})$"
        )
        # Baut beim Start einen Lookup-Index und merkt sich bekannte Set-Codes.
        self.known_set_codes = set()
        self.card_code_index = self._build_card_code_index()
        self.logger.info(
            "Card repository initialized with %d indexed code variants and %d known set codes.",
            len(self.card_code_index),
            len(self.known_set_codes),
        )

    def _build_card_code_index(self):
        """Erstellt einen Index, damit verschiedene Schreibweisen schnell zur Karte führen."""
        index = defaultdict(list)
        total_rows = 0

        # Geht jede Karte aus der DB einmal durch und erzeugt mehrere Suchvarianten.
        for row in load_cards_for_code_index():
            total_rows += 1
            public_code = row["publicCode"]
            if not public_code:
                continue

            record = CardRecord(
                id=row["id"],
                name=row["name"],
                public_code=public_code,
            )
            code_without_set = public_code.split("-", 1)[1] if "-" in public_code else public_code
            variants = {public_code, self.canonicalize_code(code_without_set)}

            # Baut zusätzlich eine Variante aus collectorNumber + Gesamtzahl.
            collector_number = row["collectorNumber"]
            if collector_number and "/" in code_without_set:
                _, total_cards = code_without_set.split("/", 1)
                variants.add(self.canonicalize_code(f"{collector_number}/{total_cards}"))

            # Der Set-Code wird getrennt gespeichert, aber nicht als eigener DB-Key indexiert.
            if "-" in public_code:
                set_code, _ = public_code.split("-", 1)
                if self.set_code_pattern.fullmatch(set_code):
                    self.known_set_codes.add(set_code)

            for variant in variants:
                index[variant].append(record)

        self.logger.debug(
            "Built card code index from %d rows.",
            total_rows,
        )
        return dict(index)

    def canonicalize_code(self, code):
        """Bringt einen Karten-Code in ein einheitliches Standardformat."""
        # Grundbereinigung: Großbuchstaben, Leerzeichen raus, | zu /.
        cleaned_code = self._clean_code(code)
        match = self.canonical_code_pattern.fullmatch(cleaned_code)
        if not match:
            return cleaned_code

        # Formt alles in eine stabile Standardschreibweise um.
        set_code = match.group("set")
        prefix = f"{set_code}-" if set_code else ""
        card_number = int(match.group("number"))
        card_suffix = match.group("suffix")
        total_cards = match.group("total")
        return f"{prefix}{card_number:03d}{card_suffix}/{total_cards}"

    @staticmethod
    def _clean_code(code):
        """Entfernt unerwünschte Zeichen und vereinheitlicht die Grundschreibweise eines Codes."""
        return (
            code.upper().strip().replace(" ", "").replace("•", "").replace("|", "/")
        )

    def verify_code(self, code):
        """Prüft, ob der Code im Index existiert, und gibt passende Karten zurück."""
        # Vereinheitlicht zuerst die Eingabe und versucht dann einen Fallback ohne Set-Code.
        canonical_code = self.canonicalize_code(code)
        matches = self.card_code_index.get(canonical_code, [])
        if matches:
            self.logger.debug(
                "Verified code '%s' as canonical '%s' with %d direct matches.",
                code,
                canonical_code,
                len(matches),
            )
            return matches

        code_without_set = canonical_code.split("-", 1)[1] if "-" in canonical_code else canonical_code
        fallback_matches = self.card_code_index.get(code_without_set, [])
        self.logger.debug(
            "Verified code '%s' as canonical '%s' with %d fallback matches using '%s'.",
            code,
            canonical_code,
            len(fallback_matches),
            code_without_set,
        )
        return fallback_matches


class CardCodeParser:
    def __init__(self, repository, logger=None):
        """Verknüpft den Parser mit dem Repository, um Codes direkt validieren zu können."""
        # Repository kapselt bekannte Set-Codes, Canonicalisierung und DB-Lookups.
        self.repository = repository
        self.logger = logger or logging.getLogger(__name__)
        # Codes mit Set wie UNL-001/219.
        self.full_code_pattern = re.compile(
            r"(?P<set>[A-Z]{3})[-\s•.]*?(?P<number>\d{1,3})(?P<suffix>[A-Z*]?)[/\|](?P<total>\d{3})"
        )
        # Codes ohne Set wie 001/219.
        self.partial_code_pattern = re.compile(
            r"(?P<number>\d{1,3})(?P<suffix>[A-Z*]?)[/\|](?P<total>\d{3})"
        )
        # Token-Codes folgen einem eigenen Schema wie UNL-T06 und haben keine Gesamtzahl.
        self.token_code_pattern = re.compile(
            r"(?P<set>[A-Z]{3})[-\s•·.]*(?P<token>T[0-9OIL]{2})"
        )
        # Übersetzungstabelle, um typische OCR-Fehler zu glätten.
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
        """Liefert extrahierte Kandidaten und den bestbewerteten Treffer."""
        # Extrahiert Kandidaten und nimmt dann den bestbewerteten Treffer.
        candidates = self.extract_candidates(lines)
        best_candidate = max(candidates, key=self.score_candidate, default="")
        if best_candidate:
            self.logger.info(
                "Selected best candidate '%s' from %d OCR candidates.",
                best_candidate,
                len(candidates),
            )
        else:
            self.logger.warning("No valid card code candidates found in OCR output.")
        return {
            "candidates": candidates,
            "best_candidate": best_candidate,
        }

    def extract_candidates(self, lines):
        """Sammelt aus mehreren OCR-Zeilen alle plausiblen Karten-Code-Kandidaten."""
        token_searchable_texts = [
            line.upper().strip().replace(" ", "").replace("\n", "")
            for line in lines
            if line
        ]
        # Bereinigt jede OCR-Zeile vor der eigentlichen Suche.
        normalized_lines = [
            line.upper().strip().replace(" ", "").replace("\n", "").translate(
                self.normalization_table
            )
            for line in lines
            if line
        ]
        searchable_texts = list(normalized_lines)
        if normalized_lines:
            # Der Join deckt Fälle ab, in denen OCR einen Code auf mehrere Zeilen verteilt.
            searchable_texts.append(" ".join(normalized_lines))
            token_searchable_texts.append(" ".join(token_searchable_texts))
        self.logger.debug("Searchable OCR texts: %s", searchable_texts)

        candidates = []
        for normalized_text in searchable_texts:
            # Vollcodes mit bekanntem Set-Code wie UNL-001/219.
            for match in self.full_code_pattern.finditer(normalized_text):
                set_code = match.group("set")
                self.logger.debug(
                    "Found full-code regex match %r with set code '%s'.",
                    match.group(0),
                    set_code,
                )
                if set_code in self.repository.known_set_codes:
                    candidate = self.repository.canonicalize_code(match.group(0))
                    if self.repository.verify_code(candidate):
                        self.logger.debug(
                            "Accepted full-code candidate '%s'.",
                            candidate,
                        )
                        candidates.append(candidate)
                    else:
                        self.logger.debug(
                            "Rejected full-code candidate '%s' after DB verification.",
                            candidate,
                        )
                else:
                    self.logger.debug(
                        "Rejected full-code match %r because set code '%s' is unknown.",
                        match.group(0),
                        set_code,
                    )

            # Codes ohne Set wie 001/219.
            for match in self.partial_code_pattern.finditer(normalized_text):
                candidate = self.repository.canonicalize_code(match.group(0))
                self.logger.debug(
                    "Found partial-code regex match %r normalized to '%s'.",
                    match.group(0),
                    candidate,
                )
                if self.repository.verify_code(candidate):
                    self.logger.debug("Accepted partial-code candidate '%s'.", candidate)
                    candidates.append(candidate)
                else:
                    self.logger.debug(
                        "Rejected partial-code candidate '%s' after DB verification.",
                        candidate,
                    )

        # Token-Karten verwenden z. B. UNL-T06 statt des normalen 001/219-Schemas.
        # Dieser Pfad bleibt ein Fallback, damit reguläre Kartencodes immer Vorrang haben.
        if not candidates:
            candidates.extend(
                self.extract_token_fallback_candidates(token_searchable_texts)
            )

        # Entfernt Duplikate, aber lässt die ursprüngliche Reihenfolge stehen.
        unique_candidates = list(dict.fromkeys(candidate for candidate in candidates if candidate))
        self.logger.info(
            "Extracted %d unique OCR candidates from %d searchable texts.",
            len(unique_candidates),
            len(searchable_texts),
        )
        return unique_candidates

    def extract_token_fallback_candidates(self, searchable_texts):
        """Extrahiert und validiert Token-Codes wie UNL-T06 als OCR-Fallback."""
        candidates = []
        for normalized_text in searchable_texts:
            for match in self.token_code_pattern.finditer(normalized_text):
                set_code = match.group("set")
                if set_code not in self.repository.known_set_codes:
                    self.logger.debug(
                        "Rejected token-code match %r because set code '%s' is unknown.",
                        match.group(0),
                        set_code,
                    )
                    continue

                token_code = match.group("token").translate(
                    str.maketrans({"O": "0", "I": "1", "L": "1"})
                )
                candidate = f"{set_code}-{token_code}"
                if self.repository.verify_code(candidate):
                    self.logger.debug(
                        "Accepted token-code fallback candidate '%s'.",
                        candidate,
                    )
                    candidates.append(candidate)
                else:
                    self.logger.debug(
                        "Rejected token-code fallback candidate '%s' after DB verification.",
                        candidate,
                    )

        return candidates

    def score_candidate(self, candidate):
        """Bewertet einen Kandidaten danach, wie gut er wie ein echter Karten-Code aussieht."""
        # Eindeutige DB-Treffer zählen stärker als reine Format-Heuristiken.
        match_count = len(self.repository.verify_code(candidate))
        db_score = 3 if match_count == 1 else 1 if match_count > 1 else 0

        # Saubere Kartencodes mit Set sind am stärksten, Kurzformen nur ein Fallback.
        canonical_match = self.repository.canonical_code_pattern.fullmatch(candidate)
        if canonical_match and "-" in candidate:
            match_score = 4
        elif canonical_match:
            match_score = 3
        elif re.fullmatch(r"(?:[A-Z]{3}-)?T\d{2}", candidate):
            match_score = 2
        else:
            match_score = 0

        digits = sum(char.isdigit() for char in candidate)
        penalty = sum(char not in "0123456789/*ABCDEFGHIJKLMNOPQRSTUVWXYZ" for char in candidate)
        score = (db_score, match_score, "*" in candidate, digits, -penalty)
        self.logger.debug(
            "Scored candidate '%s' with score=%s match_count=%d.",
            candidate,
            score,
            match_count,
        )
        return score
