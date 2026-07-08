"""lang_detector.py — rileva la lingua di un file di trascrizione.

Usa il modello PyTorch papluca/xlm-roberta-base-language-detection, scaricato
(e riutilizzato) nella cache locale .cache/model.

Uso:
    python lang_detector.py <percorso_file_md>

Output:
    - su STDOUT: SOLO il codice ISO 639-3 a 3 lettere della lingua (es. "ITA"),
      cosi' l'orchestratore PowerShell puo' catturarlo pulito.
    - su STDERR: tutti i messaggi di avanzamento.
"""
import os
import re
import sys
import argparse

from dotenv import load_dotenv

from langs import iso1_to_iso3

# Root del progetto: gli script vivono in src/, la cache sta in root.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Carica .env dalla root (es. HF_TOKEN) prima di usare huggingface_hub / transformers.
load_dotenv(os.path.join(ROOT_DIR, ".env"))

# Il modello viene scaricato/riutilizzato nella cache condivisa in root.
CACHE_DIR = os.path.join(ROOT_DIR, ".cache", "model")
MODEL_ID = "papluca/xlm-roberta-base-language-detection"


def log(msg: str) -> None:
    """Messaggi informativi su stderr per non sporcare lo stdout."""
    print(msg, file=sys.stderr, flush=True)


def estrai_testo(percorso_file: str) -> str:
    """Legge il file .md e restituisce il solo corpo, senza header/Markdown."""
    with open(percorso_file, "r", encoding="utf-8") as f:
        contenuto = f.read()

    # Scarta l'header inserito da trascrittore_gemini.py (fino al primo '---').
    if "\n---\n" in contenuto:
        contenuto = contenuto.split("\n---\n", 1)[1]

    # Rimuove sintassi Markdown di base e URL per non confondere il classificatore.
    contenuto = re.sub(r"https?://\S+", " ", contenuto)
    contenuto = re.sub(r"[#*`>_\[\]()]", " ", contenuto)
    return contenuto.strip()


def rileva_lingua(testo: str) -> str:
    """Restituisce il codice ISO 639-3 (3 lettere, MAIUSCOLO) della lingua."""
    # Import pesanti solo qui: cosi' --help resta istantaneo.
    from transformers import pipeline

    os.makedirs(CACHE_DIR, exist_ok=True)
    log(f"Carico il modello di rilevamento lingua (cache: {CACHE_DIR})...")

    import torch

    device = 0 if torch.cuda.is_available() else -1
    log(f"Device: {'GPU (cuda)' if device == 0 else 'CPU'}")

    classificatore = pipeline(
        "text-classification",
        model=MODEL_ID,
        tokenizer=MODEL_ID,
        device=device,
        model_kwargs={"cache_dir": CACHE_DIR},
    )

    # Usiamo un campione ampio ma troncato dal tokenizer (max ~512 token).
    risultato = classificatore(testo[:4000], truncation=True, top_k=1)
    # top_k=1 puo' restituire una lista annidata a seconda della versione.
    if isinstance(risultato, list) and risultato and isinstance(risultato[0], list):
        risultato = risultato[0]
    predizione = risultato[0]

    iso1 = predizione["label"]
    iso3 = iso1_to_iso3(iso1)
    log(f"Lingua rilevata: {iso1} -> {iso3} (confidenza {predizione['score']:.2%})")
    return iso3


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rileva la lingua di un file di trascrizione (output: codice ISO 639-3)."
    )
    parser.add_argument("file", help="Percorso del file .md da analizzare")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        log(f"Errore: file non trovato: {args.file}")
        sys.exit(1)

    testo = estrai_testo(args.file)
    if not testo:
        log("Errore: il file non contiene testo analizzabile.")
        sys.exit(1)

    iso3 = rileva_lingua(testo)
    # UNICA riga su stdout: il codice della lingua.
    print(iso3)


if __name__ == "__main__":
    main()
