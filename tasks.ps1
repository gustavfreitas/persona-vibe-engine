<#
.SYNOPSIS
    Atalhos do projeto: .\tasks.ps1 <tarefa> [-Csv caminho]
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet("help", "install", "train", "test", "app", "evaluate", "predict", "clean")]
    [string]$Task = "help",

    # CSV de entrada: dataset de treino (train) ou faixas novas (predict)
    [string]$Csv = "data/dataset.csv",

    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot   # funciona de qualquer pasta

$Clustered = "data/clustered_tracks.csv"

function Invoke-Py {
    # Corre o Python e interrompe se o comando falhar
    & $Python @args
    if ($LASTEXITCODE -ne 0) {
        throw "Falhou: $Python $args (código $LASTEXITCODE)"
    }
}

switch ($Task) {
    "install" {
        Invoke-Py -m pip install -r requirements-dev.txt
    }
    "train" {
        Invoke-Py -m src.model --input-csv $Csv
    }
    "test" {
        Invoke-Py -m pytest -v
    }
    "app" {
        # Treina automaticamente se o CSV com clusters ainda não existir
        if (-not (Test-Path $Clustered)) {
            Write-Host "$Clustered não existe: a treinar primeiro..."
            Invoke-Py -m src.model --input-csv $Csv
        }
        Invoke-Py -m streamlit run app.py
    }
    "evaluate" {
        Invoke-Py -m src.evaluate_k
    }
    "predict" {
        if ($Csv -eq "data/dataset.csv") {
            throw "Indique as faixas novas: .\tasks.ps1 predict -Csv data/novas_faixas.csv"
        }
        Invoke-Py -m src.predict --input-csv $Csv
    }
    "clean" {
        # Só remove caches; não toca em data/ nem em models/
        Get-ChildItem -Recurse -Directory -Force |
            Where-Object { $_.Name -in "__pycache__", ".pytest_cache" } |
            Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "Caches removidas."
    }
    default {
        Write-Host @"
Uso: .\tasks.ps1 <tarefa> [-Csv caminho]

  install   instala as dependências (inclui pytest)
  train     treina o modelo -> data/clustered_tracks.csv + models/
  test      corre os testes
  app       abre a app Streamlit (treina antes, se for preciso)
  evaluate  silhouette e inércia para vários k
  predict   classifica faixas novas (-Csv obrigatório)
  clean     apaga __pycache__ e .pytest_cache
"@
    }
}