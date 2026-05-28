$years = 2021,2022,2023,2024,2025

foreach ($y in $years) {
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "  Starting orchestrator for year $y"
    Write-Host "============================================================"
    python scripts\orchestrator.py --year $y
    if (-not $?) {
        Write-Host "Year $y failed - stopping queue." -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "All years completed." -ForegroundColor Green
