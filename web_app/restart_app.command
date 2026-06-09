#!/bin/bash
cd "$(dirname "$0")"
# Kill any running Streamlit on port 8501
pkill -f "streamlit run app.py" 2>/dev/null
sleep 1
# Restart
source /opt/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate base 2>/dev/null || true
streamlit run app.py --server.port 8501
