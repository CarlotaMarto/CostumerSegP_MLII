@echo off
REM Start the Streamlit app and open it in the default browser
cd /d "%~dp0"
start "Streamlit App" cmd /k "py -3 -m streamlit run app.py --server.port 8501"
timeout /t 6 > nul
start "" "http://localhost:8501"
