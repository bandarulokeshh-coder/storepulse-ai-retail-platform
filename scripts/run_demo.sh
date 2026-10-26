#!/bin/bash
# Run StorePulse AI demo (Linux/Mac)
# Task 9: Easy demo startup script

echo "Starting StorePulse AI..."
echo ""

# Activate virtual environment
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
else
    echo "Error: Virtual environment not found at .venv/bin/activate"
    echo "Please run: python -m venv .venv"
    echo "Then: source .venv/bin/activate"
    echo "Then: pip install -r requirements.txt"
    exit 1
fi

# Run Streamlit app
streamlit run app.py
