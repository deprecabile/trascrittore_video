import yt_dlp
from google import genai
from google.genai import types
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import argparse
from urllib.parse import urlparse, parse_qs
from datetime import datetime
from dotenv import load_dotenv

# Root del progetto: questo script vive in src/, .env e out/ stanno in root.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Carica variabili d'ambiente dal file .env in root
load_dotenv(os.path.join(ROOT_DIR, ".env"))

# --- CONFIGURAZIONE ---
API_KEY_GOOGLE = os.environ.get("GEMINI_API_KEY")
MODEL_ID = "gemini-3.5-transcribe"
MAX_SECONDI_BLOCCO = 30 * 60
TIMEOUT_ELABORAZIONE = 5 * 60

if not API_KEY_GOOGLE:
    raise ValueError("La chiave API GEMINI_API_KEY non è stata trovata nel file .env")

# Inizializza il client Google GenAI
client = genai.Client(
    api_key=API_KEY_GOOGLE,
    http_options=types.HttpOptions(
        api_version="v1beta",
        timeout=600_000,
        retry_options=types.HttpRetryOptions(
            attempts=4, initial_delay=2, max_delay=20, exp_base=2, jitter=1,
            http_status_codes=[408, 429, 500, 502, 503, 504],
        ),
    ),
)

def estrai_id_video(url):
    parsed = urlparse(url)
    if parsed.hostname in ('youtu.be', 'www.youtu.be'):
        return parsed.path[1:]
    if parsed.hostname in ('youtube.com', 'www.youtube.com'):
        if parsed.path == '/watch':
            return parse_qs(parsed.query).get('v', [''])[0]
        if parsed.path.startswith('/embed/'):
            return parsed.path.split('/')[2]
        if parsed.path.startswith('/v/'):
            return parsed.path.split('/')[2]
        if parsed.path.startswith('/shorts/'):
            return parsed.path.split('/')[2]
    return "trascrizione"

def scarica_audio(url):
    print("1. Scarico l'audio da YouTube...", file=sys.stderr, flush=True)
    opzioni = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '128',
        }],
        'outtmpl': 'audio_temp.%(ext)s',
        'quiet': True,
        'noprogress': True,
        'noplaylist': True,
        'logtostderr': True,
        'js_runtimes': {'node': {}}
    }
    for tentativo in range(2):
        try:
            # Una nuova istanza riestrae anche gli URL firmati del video.
            with yt_dlp.YoutubeDL(opzioni) as ydl:
                ydl.download([url])
            return "audio_temp.mp3"
        except yt_dlp.utils.DownloadError as e:
            if 'HTTP Error 403' not in str(e):
                raise
            if tentativo == 0:
                print(
                    "YouTube ha risposto HTTP 403: rigenero gli URL senza cache e riprovo...",
                    file=sys.stderr, flush=True,
                )
                opzioni['cachedir'] = False
                continue
            raise RuntimeError(
                "YouTube ha rifiutato il download dell'audio (HTTP 403) anche al secondo tentativo. "
                "Aggiorna yt-dlp e i componenti JavaScript dalla root del progetto con "
                "'uv sync --upgrade-package yt-dlp --upgrade-package yt-dlp-ejs', "
                "verifica che Node.js sia disponibile e riprova."
            ) from e

def prepara_blocchi_audio(file_audio, cartella_temp):
    """Divide gli MP3 lunghi: Transcribe puo' troncare audio oltre il limite API."""
    percorso = Path(file_audio).resolve()
    try:
        risultato = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(percorso)],
            check=True, capture_output=True, text=True, encoding="utf-8",
        )
        durata = float(risultato.stdout.strip())
        if not math.isfinite(durata) or durata <= 0:
            raise ValueError("La durata dell'audio non è valida.")
        if durata <= MAX_SECONDI_BLOCCO:
            return [percorso]

        print("   Divido l'audio in blocchi di circa 30 minuti...", file=sys.stderr, flush=True)
        subprocess.run(
            ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
             "-i", str(percorso), "-map", "0:a:0", "-c:a", "copy",
             "-f", "segment", "-segment_time", str(MAX_SECONDI_BLOCCO),
             "-reset_timestamps", "1", str(Path(cartella_temp) / "blocco_%04d.mp3")],
            check=True, capture_output=True, text=True, encoding="utf-8",
        )
    except FileNotFoundError as e:
        raise RuntimeError("ffmpeg e ffprobe devono essere disponibili nel PATH.") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Preparazione dell'audio fallita: {e.stderr.strip()}") from e

    blocchi = sorted(Path(cartella_temp).glob("blocco_*.mp3"))
    if not blocchi or any(blocco.stat().st_size == 0 for blocco in blocchi):
        raise RuntimeError("ffmpeg non ha prodotto blocchi audio validi.")
    return blocchi


def trascrivi_blocco(file_audio):
    audio_file = client.files.upload(file=str(file_audio))
    nome_remoto = audio_file.name
    try:
        scadenza = time.monotonic() + TIMEOUT_ELABORAZIONE
        while audio_file.state and audio_file.state.name == "PROCESSING":
            if time.monotonic() >= scadenza:
                raise TimeoutError("Elaborazione del file audio su Google scaduta.")
            print("   Elaborazione file in corso...", file=sys.stderr, flush=True)
            time.sleep(2)
            audio_file = client.files.get(name=nome_remoto)

        if not audio_file.state or audio_file.state.name != "ACTIVE":
            raise RuntimeError("Il file audio non è pronto per la trascrizione su Google.")

        response = client.interactions.create(
            model=MODEL_ID,
            input=[{"type": "audio", "uri": audio_file.uri, "mime_type": audio_file.mime_type}],
            generation_config={"transcription_config": {"mode": "smart"}},
            store=False,
        )
        if response.status != "completed":
            raise RuntimeError(f"Trascrizione incompleta: stato {response.status}.")
        testo = (response.output_text or "").strip()
        if not testo:
            raise RuntimeError("Gemini non ha restituito alcuna trascrizione.")
        return testo
    finally:
        # Anche in caso di 503; un errore di pulizia non nasconde l'errore originale.
        try:
            client.files.delete(name=nome_remoto)
        except Exception as e:
            print(f"Avviso: impossibile eliminare il file remoto {nome_remoto}: {e}",
                  file=sys.stderr, flush=True)


def trascrivi_con_gemini(file_audio):
    print(f"2. Preparo l'audio per {MODEL_ID}...", file=sys.stderr, flush=True)
    cartella_cache = Path(ROOT_DIR) / ".cache"
    cartella_cache.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="trascrizione-", dir=cartella_cache) as cartella_temp:
        blocchi = prepara_blocchi_audio(file_audio, cartella_temp)
        testi = []
        for indice, blocco in enumerate(blocchi, start=1):
            print(f"3. Trascrivo il blocco {indice}/{len(blocchi)} con {MODEL_ID} (smart)...",
                  file=sys.stderr, flush=True)
            testi.append(trascrivi_blocco(blocco))
        return "\n\n".join(testi)

def sanitizza_nome_file(nome):
    # Rimuove un'eventuale estensione .md e i caratteri non validi nei nomi file Windows.
    nome = nome.strip()
    if nome.lower().endswith(".md"):
        nome = nome[:-3]
    nome = re.sub(r'[<>:"/\\|?*]', "_", nome).strip(" .")
    return nome

def ottieni_percorso_file(url, nome=None):
    data_odierna = datetime.now().strftime("%Y_%m_%d")
    cartella_output = os.path.join(ROOT_DIR, "out", "original", data_odierna)
    if nome:
        nome = sanitizza_nome_file(nome)
    nome_file = nome or estrai_id_video(url) or "video_sconosciuto"
    return os.path.join(cartella_output, f"{nome_file}.md")

def salva_in_md(testo, percorso_file, url):
    print("4. Salvo il file Markdown...", file=sys.stderr, flush=True)
    os.makedirs(os.path.dirname(percorso_file), exist_ok=True)
    with open(percorso_file, "w", encoding="utf-8") as f:
        f.write(f"# Trascrizione con Gemini\n\n**Fonte:** {url}\n\n---\n\n")
        f.write(testo)

# --- ESECUZIONE ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trascrive un video di YouTube usando Gemini.")
    parser.add_argument("url", help="L'URL del video di YouTube da trascrivere")
    parser.add_argument("--name", default=None, help="Nome del file di output (senza estensione); default: l'ID del video")
    parser.add_argument("--f", action="store_true", help="Forza la sovrascrittura se il file esiste già")
    args = parser.parse_args()

    percorso_file = ottieni_percorso_file(args.url, args.name)

    if not args.f and os.path.exists(percorso_file) and os.path.getsize(percorso_file) > 0:
        print(f"Il video è già stato trascritto: {percorso_file}\nUsa il parametro --f per forzare la sovrascrittura.", file=sys.stderr, flush=True)
        # Riga machine-readable per l'orchestratore della pipeline.
        print(f"OUTPUT_FILE={percorso_file}")
        exit(0)

    audio_local = None
    try:
        audio_local = scarica_audio(args.url)
        testo_finale = trascrivi_con_gemini(audio_local)
        salva_in_md(testo_finale, percorso_file, args.url)
        print(f"\nFatto! Controlla il file '{percorso_file}'.", file=sys.stderr, flush=True)
        # Riga machine-readable per l'orchestratore della pipeline.
        print(f"OUTPUT_FILE={percorso_file}")
    except Exception as e:
        print(f"\nSi è verificato un errore: {e}", file=sys.stderr, flush=True)
        sys.exit(1)
    finally:
        # Pulizia del file audio locale temporaneo
        if audio_local and os.path.exists(audio_local):
            os.remove(audio_local)
