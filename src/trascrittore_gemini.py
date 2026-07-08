import yt_dlp
from google import genai
import os
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

if not API_KEY_GOOGLE:
    raise ValueError("La chiave API GEMINI_API_KEY non è stata trovata nel file .env")

# Inizializza il client Google GenAI
client = genai.Client(api_key=API_KEY_GOOGLE)

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
    print("1. Scarico l'audio da YouTube...")
    opzioni = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '128',
        }],
        'outtmpl': 'audio_temp.%(ext)s',
        'quiet': True,
        'js_runtimes': {'node': {}}
    }
    with yt_dlp.YoutubeDL(opzioni) as ydl:
        ydl.download([url])
    return "audio_temp.mp3"

def trascrivi_con_gemini(file_audio):
    print("2. Carico il file audio sui server di Google...")
    # Carichiamo il file usando la Files API (necessaria per file grandi come gli audio)
    audio_file = client.files.upload(file=file_audio)
    
    # Aspettiamo che Google elabori il file (di solito pochi secondi)
    while audio_file.state.name == "PROCESSING":
        print("   Elaborazione file in corso...")
        time.sleep(2)
        audio_file = client.files.get(name=audio_file.name)

    if audio_file.state.name == "FAILED":
        raise Exception("Caricamento del file audio fallito su Google.")

    print("3. Gemini sta trascrivendo l'audio...")
    
    prompt = """
    Trascrivi questo file audio in italiano. 
    Segui queste regole tassative:
    1. Mantieni il testo fedele ma correggi eventuali piccoli "ehm" o ripetizioni parassite.
    2. Dividi il testo in paragrafi logici per renderlo leggibile.
    3. Se ci sono concetti chiave, usa il grassetto per evidenziarli.
    """

    # Chiediamo la trascrizione a Gemini 2.5 Flash
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[audio_file, prompt]
    )
    
    # Pulizia: cancelliamo il file dai server di Google
    client.files.delete(name=audio_file.name)
    
    return response.text

def ottieni_percorso_file(url):
    data_odierna = datetime.now().strftime("%Y_%m_%d")
    cartella_output = os.path.join(ROOT_DIR, "out", "original", data_odierna)
    video_id = estrai_id_video(url)
    if not video_id:
        video_id = "video_sconosciuto"
    return os.path.join(cartella_output, f"{video_id}.md")

def salva_in_md(testo, percorso_file, url):
    print("4. Salvo il file Markdown...")
    os.makedirs(os.path.dirname(percorso_file), exist_ok=True)
    with open(percorso_file, "w", encoding="utf-8") as f:
        f.write(f"# Trascrizione con Gemini\n\n**Fonte:** {url}\n\n---\n\n")
        f.write(testo)

# --- ESECUZIONE ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trascrive un video di YouTube usando Gemini.")
    parser.add_argument("url", help="L'URL del video di YouTube da trascrivere")
    parser.add_argument("--f", action="store_true", help="Forza la sovrascrittura se il file esiste già")
    args = parser.parse_args()

    percorso_file = ottieni_percorso_file(args.url)

    if not args.f and os.path.exists(percorso_file) and os.path.getsize(percorso_file) > 0:
        print(f"Il video è già stato trascritto: {percorso_file}\nUsa il parametro --f per forzare la sovrascrittura.")
        # Riga machine-readable per l'orchestratore della pipeline.
        print(f"OUTPUT_FILE={percorso_file}")
        exit(0)

    audio_local = None
    try:
        audio_local = scarica_audio(args.url)
        testo_finale = trascrivi_con_gemini(audio_local)
        salva_in_md(testo_finale, percorso_file, args.url)
        print(f"\nFatto! Controlla il file '{percorso_file}'.")
        # Riga machine-readable per l'orchestratore della pipeline.
        print(f"OUTPUT_FILE={percorso_file}")
    except Exception as e:
        print(f"\nSi è verificato un errore: {e}")
    finally:
        # Pulizia del file audio locale temporaneo
        if audio_local and os.path.exists(audio_local):
            os.remove(audio_local)