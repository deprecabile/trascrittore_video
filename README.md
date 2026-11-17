# Trascrittore Video

Pipeline che, dato l'URL di un video di YouTube, ne produce la **trascrizione in Markdown** e — se il video non è in italiano — la **traduzione automatica in italiano**.

---

## Cos'è e a cosa serve

Il progetto automatizza in tre passi il percorso "video → testo leggibile in italiano":

1. **Trascrizione** — scarica l'audio del video e lo trascrive nella lingua originale con **Gemini 3.5 Transcribe** (`gemini-3.5-transcribe`), in modalità `smart`: pulizia delle esitazioni e formattazione del testo, salvato in Markdown.
2. **Rilevamento lingua** — riconosce la lingua della trascrizione con un modello **locale** (nessuna chiamata esterna).
3. **Traduzione** — se la lingua non è l'italiano, traduce il documento con un modello **locale** (**Gemma 4** di Google), preservando la formattazione Markdown.

Il rilevamento lingua e la traduzione girano **in locale sul tuo PC** (via PyTorch), sfruttando la GPU se disponibile: solo la trascrizione iniziale usa un servizio cloud (Gemini).

I risultati vengono salvati in:

```
out/original/<AAAA_MM_GG>/<id-video>.md      # trascrizione (lingua originale)
out/translated/<AAAA_MM_GG>/<id-video>.md    # versione italiana (tradotta, o copia se già in italiano)
```

---

## Come si usa

Un solo comando, che orchestra tutta la pipeline:

```powershell
.\run_transcript.ps1 -Url "https://www.youtube.com/watch?v=XXXXXXXXXXX"
```

Cosa succede:

- Il video viene trascritto in `out/original/<data>/<id>.md`.
- Se è **già in italiano**, la trascrizione viene semplicemente copiata in `out/translated/<data>/<id>.md`.
- Altrimenti viene tradotto in `out/translated/<data>/<id>.md`.

In entrambi i casi trovi quindi sempre la versione italiana in `out/translated/`.

Al termine lo script stampa i percorsi dei file prodotti.

**Opzioni:**

| Comando | Effetto |
|---|---|
| `.\run_transcript.ps1 -Url "<url>"` | Esegue la pipeline; salta la trascrizione se il video è già stato trascritto. |
| `.\run_transcript.ps1 -Url "<url>" -f` | Forza la ri-trascrizione anche se il file esiste già. |

> **Nota sulla prima esecuzione:** al primo avvio i modelli locali vengono scaricati in `.cache/model/` (il traduttore Gemma pesa ~16 GB). È un'operazione una tantum: dalle volte successive vengono riutilizzati dalla cache.

Sono accettati tutti i formati di URL YouTube (`watch?v=`, `youtu.be/`, `/embed/`, `/shorts/`, ...).

---

## Preparazione dell'ambiente (dopo il clone)

### Prerequisiti di sistema

- **[uv](https://docs.astral.sh/uv/)** — gestore di ambiente e pacchetti Python (scarica anche la versione di Python giusta).
- **[ffmpeg e ffprobe](https://ffmpeg.org/)** — usati per estrarre l'audio, verificarne la durata e dividerlo in blocchi (devono essere nel `PATH`).
- **[Node.js](https://nodejs.org/)** (22 o successivo, nel `PATH`) — richiesto da yt-dlp per risolvere i controlli JavaScript di YouTube. Gli script complementari `yt-dlp-ejs` vengono installati da `uv sync` tramite la dipendenza `yt-dlp[default]`.
- **GPU NVIDIA** (consigliata) — per far girare la traduzione in modo veloce. Senza GPU funziona comunque su CPU, ma è molto più lento (serve parecchia RAM).

### 1. Installa le dipendenze Python

Dalla cartella del progetto:

```powershell
uv sync
```

`uv` crea l'ambiente virtuale `.venv/` e installa tutto quanto dichiarato in `pyproject.toml`. La build CUDA di PyTorch (indice `cu128`) è già configurata in modo dichiarativo: **non serve installare torch a mano**.

L'SDK Google richiede Python 3.11 o successivo; il venv del progetto usa Python 3.14.

Verifica che la GPU sia vista:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Deve stampare `... True`. Se stampa `False`, la traduzione userà la CPU (funziona, ma lenta).

### 2. Configura la chiave di Gemini

Crea un file `.env` nella cartella del progetto con la tua chiave API di Google Gemini:

```
GEMINI_API_KEY=la_tua_chiave_qui
```

La chiave si ottiene da [Google AI Studio](https://aistudio.google.com/apikey). Serve **solo** per la trascrizione; rilevamento lingua e traduzione sono locali e non richiedono credenziali.

### 3. Pronto

```powershell
.\run_transcript.ps1 -Url "https://www.youtube.com/watch?v=XXXXXXXXXXX"
```

### Se YouTube restituisce HTTP 403

L'errore avviene durante il download, prima della chiamata a Gemini. Lo script
rigenera gli URL senza cache e riprova una sola volta in caso di 403.
Se fallisce ancora, aggiorna yt-dlp e gli script JavaScript insieme (con le loro
dipendenze), senza richiedere l'aggiornamento dei modelli locali:

```powershell
uv sync --upgrade-package yt-dlp --upgrade-package yt-dlp-ejs
node --version
```

Poi ripeti il comando di trascrizione. Installare solo `yt-dlp` senza l'extra
`[default]` non installa gli script EJS, anche se Node è già presente.
Vedi la [guida ufficiale EJS](https://github.com/yt-dlp/yt-dlp/wiki/EJS).

Se il 403 persiste, puoi verificare il solo download di un piccolo campione,
senza chiamare Gemini:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m yt_dlp --ignore-config --js-runtimes node --no-playlist --test -f "bestaudio/best" -o ".cache/download-check/%(id)s.%(ext)s" "<url>"
```

### Trascrizione con Gemini 3.5 Transcribe

La trascrizione usa la **Interactions API** e la modalità `smart`, con rilevamento
automatico della lingua. La traduzione in italiano rimane affidata allo stadio
locale Gemma. Non viene inviato un prompt di traduzione o di riassunto a Transcribe;
la formattazione è quella della modalità `smart`, senza l'istruzione aggiuntiva
del vecchio prompt per evidenziare i concetti chiave in grassetto.

Gli audio oltre 30 minuti vengono divisi con ffmpeg e trascritti in ordine, poi
riuniti in un unico Markdown. Questo evita il troncamento silenzioso previsto
dall'API per audio oltre il suo limite. I tagli avvengono ai confini dei frame MP3,
quindi una frase può attraversare due blocchi. File temporanei e upload vengono
rimossi anche in caso di errore; le interazioni usano `store=False`.
Vedi la [guida ufficiale Transcribe](https://aistudio.google.com/docs/transcribe).

Un errore **503 UNAVAILABLE** indica indisponibilità temporanea, non dimostra
che il modello sia stato disabilitato. L'SDK è configurato per un massimo di
4 tentativi con attese crescenti per errori transitori (408, 429, 500, 502, 503,
504). Gli altri errori, ad esempio una chiave non valida, interrompono il processo.
Una risposta incompleta o vuota non viene salvata come trascrizione riuscita.

Verifica locale della trascrizione e della gestione dei blocchi (senza chiamate API):

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

---

## Struttura dei modelli usati

| Passo | Modello | Dove gira |
|---|---|---|
| Trascrizione | `gemini-3.5-transcribe`, modalità `smart` (Google) | Cloud (Interactions API) |
| Rilevamento lingua | `papluca/xlm-roberta-base-language-detection` | Locale (PyTorch) |
| Traduzione | `google/gemma-4-E4B-it` (Apache 2.0) | Locale (PyTorch, 4-bit su GPU) |

I modelli locali sono scaricati in `.cache/model/` senza bisogno di token o login (Gemma 4 è pubblico, Apache 2.0). Per usare una copia offline del traduttore si può impostare la variabile d'ambiente `GEMMA_MODEL_PATH` sulla cartella del modello.

---

## Struttura del progetto

```
trascrittore_video/
├── run_transcript.ps1        # orchestratore della pipeline (entry point)
├── src/                      # sorgenti Python
│   ├── trascrittore_gemini.py  # 1. trascrizione (Gemini)
│   ├── lang_detector.py        # 2. rilevamento lingua (locale)
│   ├── traduttore.py           # 3. traduzione / copia (locale, Gemma 4)
│   └── langs.py                # mappe dei codici lingua condivise
├── pyproject.toml            # dipendenze (gestite con uv)
├── uv.lock                   # versioni bloccate (riproducibilità)
├── .env.example              # template per la chiave Gemini
├── .cache/                   # modelli scaricati        (ignorato da git)
└── out/                      # trascrizioni e traduzioni (ignorato da git)
```

---

## Licenza

Rilasciato sotto **licenza [MIT](LICENSE)** © 2026 Giampaolo Saporito: puoi usarlo,
copiarlo, modificarlo, venderlo e ridistribuirlo per qualsiasi scopo (commerciale o
meno); l'unica condizione è mantenere la nota di copyright. Vedi il file [`LICENSE`](LICENSE).

