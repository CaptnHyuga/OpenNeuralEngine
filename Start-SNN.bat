@echo off
:: SNN One-Click Launcher
:: Double-click this file to start everything!

title SNN - Starting...

:: Run PowerShell script with execution policy bypass
powershell.exe -ExecutionPolicy Bypass -NoProfile -File "%~dp0Start-SNN.ps1"

:: Keep window open if there was an error
if errorlevel 1 (
    echo.
    echo An error occurred. Press any key to exit...
    pause >nul
)
