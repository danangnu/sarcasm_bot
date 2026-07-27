$ErrorActionPreference = "Stop"
Remove-Item .\build -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item .\dist -Recurse -Force -ErrorAction SilentlyContinue
python -m PyInstaller `
  --noconfirm `
  --clean `
  --onedir `
  --name SarcasmBot `
  --icon "static\favicon.ico" `
  --add-data "static;static" `
  --add-data "models;models" `
  --collect-all google.genai `
  launcher.py
Write-Host "Build created at dist\SarcasmBot\SarcasmBot.exe"
