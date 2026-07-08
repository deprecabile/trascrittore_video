"""traduttore.py — traduce in italiano un file di trascrizione con Gemma 4.

Modello: google/gemma-4-E4B-it (LLM PyTorch / transformers).
- Su GPU con CUDA disponibile: caricamento in 4-bit (bitsandbytes), pensato
  per stare negli 8 GB di VRAM di una RTX 4060.
- Altrimenti: fallback su CPU in float32 (richiede molta RAM, ~32 GB, ma i
  64 GB della macchina bastano; sara' piu' lento).

Il modello viene riutilizzato dalla cache locale .cache/model. Alla PRIMA
esecuzione, se non presente, viene scaricato una volta li' dentro.

Uso:
    python traduttore.py <percorso_file_md> <codice_lingua_iso3>

Output:
    - su STDOUT: SOLO la riga "OUTPUT_FILE=<percorso>" del file tradotto.
    - su STDERR: tutti i messaggi di avanzamento.
"""
import os
import re
import sys
import shutil
import argparse

from dotenv import load_dotenv

from langs import nome_lingua_it, LINGUA_TARGET_ISO3

# Root del progetto: gli script vivono in src/, la cache/output stanno in root.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Carica .env dalla root: rende disponibile un eventuale HF_TOKEN a huggingface_hub
# (necessario solo per modelli gated; Gemma 4 e' Apache 2.0 e non lo richiede).
load_dotenv(os.path.join(ROOT_DIR, ".env"))

CACHE_DIR = os.path.join(ROOT_DIR, ".cache", "model")
# Usata solo dal fallback di offload se il modello non entra tutto in VRAM.
OFFLOAD_DIR = os.path.join(ROOT_DIR, ".cache", "offload")

# Sorgente del modello. Di default il repo pubblico ufficiale (Apache 2.0, non gated).
# Impostando GEMMA_MODEL_PATH a una CARTELLA LOCALE contenente i file del modello
# (config.json, model.safetensors, tokenizer.json, ...) si carica da lì offline.
MODEL_ID = os.environ.get("GEMMA_MODEL_PATH", "google/gemma-4-E4B-it")

# Il placeholder {{lang}} viene riempito con il nome italiano della lingua di partenza.
PROMPT_TEMPLATE = (
    "Traduci in italiano il seguente documento, scritto in lingua {{lang}}.\n"
    "Regole tassative:\n"
    "1. Traduci fedelmente in un italiano naturale e scorrevole.\n"
    "2. Preserva ESATTAMENTE la formattazione Markdown (titoli con #, grassetti "
    "con **, elenchi, paragrafi).\n"
    "3. Non tradurre e non alterare URL, nomi propri, sigle o codice.\n"
    "4. Rispondi con il SOLO documento tradotto, senza aggiungere commenti, "
    "premesse o note tue."
)

# Budget approssimativo di caratteri per blocco, per non saturare il contesto.
MAX_CHAR_BLOCCO = 1200

# --------------------------------------------------------------------------- #
# Speculative decoding (accelerazione della generazione)
# --------------------------------------------------------------------------- #
# Modalita' scelta via env SPEC_DECODING:
#   "off" (DEFAULT): generazione normale, token per token. E' il piu' veloce per
#            la traduzione CROSS-LINGUA: misurato ~2.5x piu' rapido di "lookup"
#            (EN->IT), perche' tra input e output tradotto non ci sono n-gram
#            comuni e il lookup finisce per speculare a vuoto (costo puro).
#   "lookup": prompt-lookup decoding. Nessun modello aggiuntivo, ZERO VRAM extra.
#            Conviene SOLO quando l'output ricalca molto l'input (stessa lingua,
#            o molto testo copiato verbatim), NON nella traduzione tra lingue.
#   "draft":  assisted generation con un draft model piu' piccolo della STESSA
#            famiglia (Gemma 4 E2B). E' l'approccio giusto per la traduzione (il
#            draft "parla italiano" e propone token plausibili), ma tiene DUE
#            modelli in VRAM: su una GPU da 8 GB e' al limite (rischio CUDA OOM).
SPEC_MODE = os.environ.get("SPEC_DECODING", "off").strip().lower()
# Draft model per la modalita' "draft": deve condividere il tokenizer del target
# (stessa famiglia). Override con GEMMA_DRAFT_MODEL_PATH per una copia locale.
DRAFT_MODEL_ID = os.environ.get("GEMMA_DRAFT_MODEL_PATH", "google/gemma-4-E2B-it")
# Quanti token "indovinare" dall'input a ogni passo, in modalita' lookup.
PROMPT_LOOKUP_TOKENS = 10


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# Lettura / scrittura file
# --------------------------------------------------------------------------- #
def leggi_documento(percorso_file: str):
    """Restituisce (url, corpo) separando l'header scritto da trascrittore_gemini."""
    with open(percorso_file, "r", encoding="utf-8") as f:
        contenuto = f.read()

    url = ""
    m = re.search(r"\*\*Fonte:\*\*\s*(\S+)", contenuto)
    if m:
        url = m.group(1)

    if "\n---\n" in contenuto:
        corpo = contenuto.split("\n---\n", 1)[1]
    else:
        corpo = contenuto
    return url, corpo.strip()


def percorso_output(percorso_input: str) -> str:
    """Costruisce out/translated/<cartella-data>/<nomefile>.md.

    Ricava la cartella-data (es. 2026_07_08) e il nome file dall'input, così
    l'output finisce sempre sotto out/translated/ con la stessa data e nome.
    """
    norm = percorso_input.replace("/", os.sep)
    nome_file = os.path.basename(norm)
    cartella_data = os.path.basename(os.path.dirname(norm))
    return os.path.join(ROOT_DIR, "out", "translated", cartella_data, nome_file)


def dividi_in_blocchi(corpo: str):
    """Raggruppa i paragrafi in blocchi sotto MAX_CHAR_BLOCCO caratteri."""
    paragrafi = re.split(r"\n\s*\n", corpo)
    blocchi, corrente = [], ""
    for p in paragrafi:
        p = p.strip()
        if not p:
            continue
        if corrente and len(corrente) + len(p) + 2 > MAX_CHAR_BLOCCO:
            blocchi.append(corrente)
            corrente = p
        else:
            corrente = f"{corrente}\n\n{p}" if corrente else p
    if corrente:
        blocchi.append(corrente)
    return blocchi


# --------------------------------------------------------------------------- #
# Modello
# --------------------------------------------------------------------------- #
def carica_modello():
    """Carica tokenizer e modello, preferendo la cache locale gia' presente."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(CACHE_DIR, exist_ok=True)
    cuda = torch.cuda.is_available()
    log(f"Device: {'GPU (cuda) 4-bit' if cuda else 'CPU float32'}")

    def _carica(local_only: bool):
        tok = AutoTokenizer.from_pretrained(
            MODEL_ID, cache_dir=CACHE_DIR, local_files_only=local_only
        )
        if cuda:
            from transformers import BitsAndBytesConfig

            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            comuni = dict(
                cache_dir=CACHE_DIR,
                local_files_only=local_only,
                quantization_config=quant,
                dtype=torch.bfloat16,
            )
            try:
                # Via veloce: tutto il modello in VRAM (Gemma E4B 4-bit ~4.5GB).
                mdl = AutoModelForCausalLM.from_pretrained(
                    MODEL_ID, device_map={"": 0}, **comuni
                )
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                # Fallback: non entra tutto in 8GB -> offload GPU+CPU fatto bene,
                # con cartella di offload per evitare tensori 'meta'.
                log("VRAM insufficiente per l'intero modello: attivo offload GPU+CPU...")
                torch.cuda.empty_cache()
                quant.llm_int8_enable_fp32_cpu_offload = True
                os.makedirs(OFFLOAD_DIR, exist_ok=True)
                mdl = AutoModelForCausalLM.from_pretrained(
                    MODEL_ID,
                    device_map="auto",
                    max_memory={0: "6GiB", "cpu": "48GiB"},
                    offload_folder=OFFLOAD_DIR,
                    **comuni,
                )
        else:
            mdl = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                cache_dir=CACHE_DIR,
                local_files_only=local_only,
                dtype=torch.float32,
            )
        return tok, mdl

    # Distinguo "file non in cache" (-> scarico) da un errore VERO di caricamento
    # (es. VRAM insufficiente): quest'ultimo NON va mascherato come "modello assente".
    try:
        from huggingface_hub.errors import LocalEntryNotFoundError
    except ImportError:  # versioni piu' vecchie di huggingface_hub
        from huggingface_hub.utils import LocalEntryNotFoundError

    # traduttore.py "si aspetta" il modello gia' in cache: prima tenta l'offline.
    def _carica_con_fallback():
        try:
            log(f"Cerco il modello nella cache locale ({CACHE_DIR})...")
            return _carica(local_only=True)
        except (LocalEntryNotFoundError, OSError):
            # Solo qui i file mancano davvero: scarico (una volta sola).
            log("Modello non presente in cache: lo scarico ora (solo alla prima esecuzione)...")
            return _carica(local_only=False)

    try:
        return _carica_con_fallback()
    except Exception as e:
        log(
            "Errore nel caricamento di Gemma 4 (google/gemma-4-E4B-it). "
            "Non e' un problema di cache: i file ci sono. Cause tipiche: VRAM "
            "insufficiente, oppure connessione di rete al primo download.\n"
            f"Dettaglio: {e}"
        )
        sys.exit(1)


def carica_draft_model():
    """Carica il draft model (Gemma 4 E2B, 4-bit su GPU) per l'assisted generation.

    Serve solo in modalita' SPEC_DECODING=draft e solo con CUDA. In caso di
    errore (modello assente, VRAM insufficiente) NON blocca la pipeline: ritorna
    None e la generazione prosegue senza draft.
    """
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    comuni = dict(cache_dir=CACHE_DIR, quantization_config=quant, dtype=torch.bfloat16)
    for local_only in (True, False):  # prima la cache, poi (una volta) il download
        try:
            log(f"Carico il draft model {DRAFT_MODEL_ID} (4-bit su GPU)...")
            return AutoModelForCausalLM.from_pretrained(
                DRAFT_MODEL_ID, device_map={"": 0}, local_files_only=local_only, **comuni
            )
        except Exception as e:  # noqa: BLE001 — degradiamo, non blocchiamo
            if local_only:
                continue  # non in cache: riprovo scaricando
            log(f"Draft model non caricato ({e}); proseguo senza assisted generation.")
            return None


def costruisci_gen_kwargs(draft_model) -> dict:
    """Parametri per generate() in base alla modalita' di speculative decoding."""
    kw = dict(max_new_tokens=2048, do_sample=False)
    if SPEC_MODE == "lookup":
        # Prompt-lookup decoding: zero VRAM extra, nessun modello aggiuntivo.
        kw["prompt_lookup_num_tokens"] = PROMPT_LOOKUP_TOKENS
        log(f"Speculative decoding: prompt-lookup ({PROMPT_LOOKUP_TOKENS} token).")
    elif SPEC_MODE == "draft" and draft_model is not None:
        kw["assistant_model"] = draft_model
        log("Speculative decoding: assisted generation con draft model.")
    else:
        log("Speculative decoding: disattivato (generazione standard).")
    return kw


def traduci_blocco(tokenizer, model, prompt: str, blocco: str, gen_kwargs: dict) -> str:
    import torch

    messaggi = [{"role": "user", "content": f"{prompt}\n\nDocumento:\n{blocco}"}]
    # return_dict=True: con Gemma 4 apply_chat_template restituisce un dizionario
    # (input_ids + attention_mask) da passare a generate con **inputs.
    inputs = tokenizer.apply_chat_template(
        messaggi,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output = model.generate(**inputs, **gen_kwargs)
    # Decodifica solo i token generati, escludendo il prompt in input.
    testo = tokenizer.decode(output[0][input_len:], skip_special_tokens=True)
    return testo.strip()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Traduce in italiano un file .md con Gemma 4.")
    parser.add_argument("file", help="Percorso del file .md da tradurre")
    parser.add_argument("lang", help="Codice ISO 639-3 della lingua di partenza (es. ENG)")
    args = parser.parse_args()

    iso3 = args.lang.strip().upper()

    if not os.path.exists(args.file):
        log(f"Errore: file non trovato: {args.file}")
        sys.exit(1)

    # Gia' in italiano: nessuna traduzione, copio la trascrizione in translated/.
    if iso3 == LINGUA_TARGET_ISO3:
        percorso_out = percorso_output(args.file)
        os.makedirs(os.path.dirname(percorso_out), exist_ok=True)
        shutil.copyfile(args.file, percorso_out)
        log(f"Il documento e' gia' in italiano: copiato in {percorso_out}")
        print(f"OUTPUT_FILE={percorso_out}")
        return

    nome_lingua = nome_lingua_it(iso3)
    prompt = PROMPT_TEMPLATE.replace("{{lang}}", nome_lingua)
    log(f"Traduco dalla lingua: {nome_lingua} ({iso3})")

    url, corpo = leggi_documento(args.file)
    blocchi = dividi_in_blocchi(corpo)
    log(f"Documento diviso in {len(blocchi)} blocco/i.")

    tokenizer, model = carica_modello()

    # Draft model solo se richiesto esplicitamente e con CUDA (su CPU non conviene).
    draft_model = None
    if SPEC_MODE == "draft":
        import torch

        if torch.cuda.is_available():
            draft_model = carica_draft_model()
        else:
            log("Modalita' draft ignorata: senza GPU non porta vantaggi.")
    gen_kwargs = costruisci_gen_kwargs(draft_model)

    tradotti = []
    for i, blocco in enumerate(blocchi, 1):
        log(f"Traduco blocco {i}/{len(blocchi)}...")
        tradotti.append(traduci_blocco(tokenizer, model, prompt, blocco, gen_kwargs))
    corpo_tradotto = "\n\n".join(tradotti)

    percorso_out = percorso_output(args.file)
    os.makedirs(os.path.dirname(percorso_out), exist_ok=True)
    with open(percorso_out, "w", encoding="utf-8") as f:
        f.write("# Traduzione in italiano\n\n")
        if url:
            f.write(f"**Fonte:** {url}\n\n")
        f.write(f"**Lingua originale:** {nome_lingua} ({iso3})\n\n---\n\n")
        f.write(corpo_tradotto)

    log(f"Traduzione salvata in: {percorso_out}")
    print(f"OUTPUT_FILE={percorso_out}")


if __name__ == "__main__":
    main()
