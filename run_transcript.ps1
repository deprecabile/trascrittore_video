param (
    [Parameter(Mandatory=$true, Position=0, HelpMessage="Inserisci l'URL del video di YouTube")]
    [string]$Url,

    [Parameter(Position=1)]
    [string]$Name,

    [switch]$f
)

$ErrorActionPreference = "Stop"

# --- Risoluzione dell'eseguibile Python del venv ---
$pythonExe = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-Not (Test-Path $pythonExe)) {
    Write-Error "Ambiente virtuale non trovato in .\.venv. Esegui prima 'uv sync'."
    exit 1
}

# Python e PowerShell devono usare la stessa codifica per i canali rediretti.
# Ripristina la codifica del terminale anche se lo step fallisce.
function Invoke-PythonStep {
    param([string]$Step, [string[]]$Arguments)
    $codificaPrecedente = [Console]::OutputEncoding
    try {
        [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
        $righe = & $pythonExe -X utf8 @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "${Step}: processo Python terminato con codice $LASTEXITCODE. Vedi l'errore precedente."
        }
        return $righe
    } finally {
        [Console]::OutputEncoding = $codificaPrecedente
    }
}

# Estrae il valore di "OUTPUT_FILE=..." dalle righe stdout di uno script.
function Get-OutputFile {
    param([string[]]$Lines)
    $riga = $Lines | Where-Object { $_ -match '^OUTPUT_FILE=' } | Select-Object -Last 1
    if ($riga) { return ($riga -replace '^OUTPUT_FILE=', '').Trim() }
    return $null
}

# ==========================================================================
# STEP 1 — Trascrizione con Gemini
# ==========================================================================
Write-Host "`n[1/3] Trascrizione con Gemini..." -ForegroundColor Cyan
$argomenti = @((Join-Path $PSScriptRoot "src\trascrittore_gemini.py"), $Url)
if ($Name) { $argomenti += @("--name", $Name) }
if ($f)    { $argomenti += "--f" }
$out1 = Invoke-PythonStep -Step "Trascrizione fallita" -Arguments $argomenti
$out1 | ForEach-Object { Write-Host $_ }

$fileTrascrizione = Get-OutputFile -Lines $out1
if (-Not $fileTrascrizione) {
    Write-Error "Trascrizione fallita: nessun file prodotto."
    exit 1
}
Write-Host "File di trascrizione: $fileTrascrizione" -ForegroundColor DarkGray

# ==========================================================================
# STEP 2 — Rilevamento della lingua (modello locale PyTorch)
# ==========================================================================
Write-Host "`n[2/3] Rilevamento della lingua..." -ForegroundColor Cyan
$out2 = Invoke-PythonStep -Step "Rilevamento lingua fallito" -Arguments @(
    (Join-Path $PSScriptRoot "src\lang_detector.py"), $fileTrascrizione
)
$lang = ($out2 | Where-Object { $_ -match '^[A-Za-z]{3}$' } | Select-Object -Last 1)
if (-Not $lang) {
    Write-Error "Rilevamento lingua fallito: nessun codice restituito."
    exit 1
}
$lang = $lang.Trim().ToUpper()
Write-Host "Lingua rilevata: $lang" -ForegroundColor Green

# ==========================================================================
# STEP 3 — Versione italiana: traduzione, oppure copia se gia' in italiano
# ==========================================================================
if ($lang -eq "ITA") {
    Write-Host "`n[3/3] Video gia' in italiano: copio la trascrizione in translated/..." -ForegroundColor Cyan
} else {
    Write-Host "`n[3/3] Traduzione in italiano con Gemma 4 (da $lang)..." -ForegroundColor Cyan
}

# traduttore.py gestisce entrambi i casi: se la lingua e' ITA copia il file,
# altrimenti traduce con Gemma. In ogni caso produce OUTPUT_FILE=<path>.
$out3 = Invoke-PythonStep -Step "Produzione della versione italiana fallita" -Arguments @(
    (Join-Path $PSScriptRoot "src\traduttore.py"), $fileTrascrizione, $lang
)
$out3 | ForEach-Object { Write-Host $_ }

$fileFinale = Get-OutputFile -Lines $out3
if (-Not $fileFinale) {
    Write-Error "Produzione della versione italiana fallita: nessun file prodotto."
    exit 1
}

Write-Host "`nPipeline completata." -ForegroundColor Green
Write-Host "Trascrizione (originale): $fileTrascrizione"
Write-Host "Versione italiana:        $fileFinale"
