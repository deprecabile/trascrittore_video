# Trascrittore Video

Pipeline che, dato l'URL di un video di YouTube, ne produce la **trascrizione in Markdown** e — se il video non è in italiano — la **traduzione automatica in italiano**.

---

## Cos'è e a cosa serve

Il progetto automatizza in tre passi il percorso "video → testo leggibile in italiano":

1. **Trascrizione** — scarica l'audio del video e lo trascrive con **Google Gemini** (`gemini-2.5-flash`), producendo un Markdown pulito (paragrafi, grassetti sui concetti chiave).
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
- **[ffmpeg](https://ffmpeg.org/)** — usato per estrarre l'audio dal video (dev'essere nel `PATH`).
- **[Node.js](https://nodejs.org/)** — richiesto da yt-dlp per risolvere i controlli JavaScript di YouTube.
- **GPU NVIDIA** (consigliata) — per far girare la traduzione in modo veloce. Senza GPU funziona comunque su CPU, ma è molto più lento (serve parecchia RAM).

### 1. Installa le dipendenze Python

Dalla cartella del progetto:

```powershell
uv sync
```

`uv` crea l'ambiente virtuale `.venv/` e installa tutto quanto dichiarato in `pyproject.toml`. La build CUDA di PyTorch (indice `cu128`) è già configurata in modo dichiarativo: **non serve installare torch a mano**.

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

---

## Struttura dei modelli usati

| Passo | Modello | Dove gira |
|---|---|---|
| Trascrizione | `gemini-2.5-flash` (Google) | Cloud (API) |
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

