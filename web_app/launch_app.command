#!/bin/bash
cd "$(dirname "$0")"
source /opt/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate base 2>/dev/null || true
streamlit run app.py --server.port 8501
