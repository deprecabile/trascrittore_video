"""Mappe di lingua condivise tra lang_detector.py e traduttore.py.

- ISO1_TO_ISO3: codice ISO 639-1 (2 lettere, output del detector papluca)
  -> codice ISO 639-3 (3 lettere, MAIUSCOLO) come richiesto in output.
- ISO3_TO_NAME_IT: codice ISO 639-3 -> nome della lingua in italiano,
  usato per riempire il placeholder {{lang}} nel prompt di traduzione
  (una parola come "inglese" rende un prompt migliore di un codice "ENG").

Le lingue coperte sono quelle del modello
papluca/xlm-roberta-base-language-detection.
"""

# codice del target: se il video e' gia' in questa lingua la pipeline si ferma
LINGUA_TARGET_ISO3 = "ITA"

ISO1_TO_ISO3 = {
    "ar": "ARA",
    "bg": "BUL",
    "de": "DEU",
    "el": "ELL",
    "en": "ENG",
    "es": "SPA",
    "fr": "FRA",
    "hi": "HIN",
    "it": "ITA",
    "ja": "JPN",
    "nl": "NLD",
    "pl": "POL",
    "pt": "POR",
    "ru": "RUS",
    "sw": "SWA",
    "th": "THA",
    "tr": "TUR",
    "ur": "URD",
    "vi": "VIE",
    "zh": "ZHO",
}

ISO3_TO_NAME_IT = {
    "ARA": "arabo",
    "BUL": "bulgaro",
    "DEU": "tedesco",
    "ELL": "greco",
    "ENG": "inglese",
    "SPA": "spagnolo",
    "FRA": "francese",
    "HIN": "hindi",
    "ITA": "italiano",
    "JPN": "giapponese",
    "NLD": "olandese",
    "POL": "polacco",
    "POR": "portoghese",
    "RUS": "russo",
    "SWA": "swahili",
    "THA": "thailandese",
    "TUR": "turco",
    "URD": "urdu",
    "VIE": "vietnamita",
    "ZHO": "cinese",
}


def iso1_to_iso3(codice_iso1: str) -> str:
    """Converte un codice a 2 lettere in 3 lettere MAIUSCOLE. Fallback: UND."""
    return ISO1_TO_ISO3.get(codice_iso1.lower(), "UND")


def nome_lingua_it(codice_iso3: str) -> str:
    """Nome italiano della lingua dato il codice ISO 639-3. Fallback: il codice."""
    return ISO3_TO_NAME_IT.get(codice_iso3.upper(), codice_iso3.lower())
