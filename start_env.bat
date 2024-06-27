@echo off

REM Attiva il virtual environment
call venv\Scripts\activate.bat

python -m pip install --upgrade pip
pip install --upgrade setuptools

REM Mantieni aperta la finestra del prompt dei comandi
cmd /k
