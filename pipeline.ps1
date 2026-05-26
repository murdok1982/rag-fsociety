Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  PIPELINE DE ENTRENAMIENTO v2 - fsociety" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "`n[1/5] Extrayendo texto de PDFs..." -ForegroundColor Yellow
python "$scriptDir\extract_pdfs.py"
if ($LASTEXITCODE -ne 0) { Write-Host "Error en extraccion" -ForegroundColor Red; exit 1 }

Write-Host "`n[2/5] Construyendo índice TurboVec (2-bit, 16x compresión)..." -ForegroundColor Yellow
python "$scriptDir\build_vector_db.py"
if ($LASTEXITCODE -ne 0) { Write-Host "Error en vector DB" -ForegroundColor Red; exit 1 }

Write-Host "`n[3/5] Limpiando dataset (filtrar, deduplicar)..." -ForegroundColor Yellow
python "$scriptDir\cleanup_dataset.py"
if ($LASTEXITCODE -ne 0) { Write-Host "Error en limpieza" -ForegroundColor Red; exit 1 }

Write-Host "`n[4/5] Preparando dataset de fine-tuning (Q&A diversos)..." -ForegroundColor Yellow
python "$scriptDir\prepare_dataset.py"
if ($LASTEXITCODE -ne 0) { Write-Host "Error en dataset" -ForegroundColor Red; exit 1 }

Write-Host "`n[5/5] Fine-tuning del modelo (QLoRA Qwen2.5-7B)..." -ForegroundColor Yellow
python "$scriptDir\finetune.py"
if ($LASTEXITCODE -ne 0) { Write-Host "Error en fine-tuning" -ForegroundColor Red; exit 1 }

Write-Host "`n============================================" -ForegroundColor Green
Write-Host "  PIPELINE COMPLETADO" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
