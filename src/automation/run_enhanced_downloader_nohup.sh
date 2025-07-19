#!/bin/bash

# Enhanced Market Data Downloader - Nohup Script
# Runs the enhanced downloader in the background with comprehensive logging

# Set up environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CONDA_ENV="financial-mesh"

# Create log directory
mkdir -p logs

# Generate timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/enhanced_downloader_${TIMESTAMP}.log"
ERROR_LOG="logs/enhanced_downloader_${TIMESTAMP}_error.log"

echo "Starting Enhanced Market Data Downloader at $(date)"
echo "Log file: $LOG_FILE"
echo "Error log: $ERROR_LOG"

# Activate conda environment and run the downloader
nohup conda run -n $CONDA_ENV python src/automation/enhanced_market_downloader.py > "$LOG_FILE" 2> "$ERROR_LOG" &

# Get the process ID
PID=$!
echo "Process started with PID: $PID"
echo "PID: $PID" > logs/enhanced_downloader.pid

# Monitor the process
echo "Monitoring process..."
while kill -0 $PID 2>/dev/null; do
    echo "$(date): Process $PID is still running..."
    sleep 300  # Check every 5 minutes
done

echo "Process $PID completed at $(date)"
echo "Check logs for details:"
echo "  Main log: $LOG_FILE"
echo "  Error log: $ERROR_LOG" 