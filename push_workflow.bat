@echo off
setlocal

cd /d %~dp0

if "%~1"=="gold" goto gold
if "%~1"=="silver" goto silver
if "%~1"=="all" goto all

echo Usage:
echo   push_workflow.bat gold
echo   push_workflow.bat silver
echo   push_workflow.bat all
exit /b 1

:gold
echo Stage gold-related files...
git add .gitignore gold_model.py GOLD_LSTM_MODEL_DESCRIPTION.md requirements.txt
if errorlevel 1 goto :error

git commit -m "Add gold LSTM workflow and gitignore"
if errorlevel 1 goto :error

git push origin main
if errorlevel 1 goto :error

echo Done.
exit /b 0

:silver
echo Stage silver-related files...
git add .gitignore silver_model.py SILVER_LSTM_MODEL_DESCRIPTION.md requirements.txt
if errorlevel 1 goto :error

git commit -m "Add silver LSTM workflow"
if errorlevel 1 goto :error

git push origin main
if errorlevel 1 goto :error

echo Done.
exit /b 0

:all
echo Stage gold and silver files...
git add .gitignore gold_model.py silver_model.py GOLD_LSTM_MODEL_DESCRIPTION.md SILVER_LSTM_MODEL_DESCRIPTION.md requirements.txt
if errorlevel 1 goto :error

git commit -m "Add gold and silver LSTM workflows"
if errorlevel 1 goto :error

git push origin main
if errorlevel 1 goto :error

echo Done.
exit /b 0

:error
echo.
echo Workflow failed.
exit /b 1
