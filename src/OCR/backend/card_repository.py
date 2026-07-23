import re
import sys
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from services.card_data import load_cards_for_code_index


class CardRecord:
    def __init__(self, id, name, public_code):
        self.id = id
        self.name = name
        self.public_code = public_code


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
        cleaned = self._clean_code(code)

        match = self.CANONICAL_CODE_PATTERN.fullmatch(cleaned)
        if not match:
            return cleaned

        set_code = match.group("set")
        number = int(match.group("number"))
        suffix = match.group("suffix")
        total = match.group("total")
        if set_code:
            return f"{set_code}-{number:03d}{suffix}/{total}"
        return f"{number:03d}{suffix}/{total}"

    def verify_code(self, code):
        canonical = self.canonicalize_code(code)
        matches = self.card_code_index.get(canonical, [])
        if matches:
            return matches

        suffix = self._code_suffix(canonical)
        if suffix != canonical:
            return self.card_code_index.get(suffix, [])

        return matches

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

        for record, collector_number in self._load_cards():
            for variant in self._build_code_variants(record.public_code, collector_number):
                index.setdefault(variant, []).append(record)

        return index

    def _build_code_variants(
        self,
        public_code,
        collector_number,
    ):
        canonical_public_code = self.canonicalize_code(public_code)
        variants = {canonical_public_code}
        suffix = self._code_suffix(public_code)

        variants.add(self.canonicalize_code(suffix))
        if collector_number and "/" in suffix:
            total = suffix.split("/", 1)[1]
            variants.add(self.canonicalize_code(f"{collector_number}/{total}"))
        set_code = self._set_prefix(canonical_public_code)
        if set_code:
            variants.add(set_code)

        return variants

    def _load_cards(self):
        cards = []
        for row in load_cards_for_code_index():
            card_id = row["id"]
            name = row["name"]
            public_code = row["publicCode"]
            collector_number = row["collectorNumber"]

            if not public_code:
                continue

            cards.append(
                (
                    CardRecord(id=card_id, name=name, public_code=public_code),
                    collector_number,
                )
            )

        return cards

    @staticmethod
    def _clean_code(code):
        cleaned = code.upper().strip().replace(" ", "")
        cleaned = cleaned.replace("•", "")
        cleaned = cleaned.replace("|", "/")
        return cleaned

    @staticmethod
    def _code_suffix(code):
        return code.split("-", 1)[1] if "-" in code else code

    @staticmethod
    def _set_prefix(code):
        return code.split("-", 1)[0] if "-" in code else None
