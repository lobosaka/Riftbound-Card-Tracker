#!/bin/bash

# Start the FastAPI backend in the background
python src/api.py &

# Start the Streamlit dashboard in the foreground
streamlit run dashboard/app.py --server.address=0.0.0.0