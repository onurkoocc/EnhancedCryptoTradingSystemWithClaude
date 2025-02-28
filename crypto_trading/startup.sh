#!/bin/bash

# Startup script for Cryptocurrency Trading System
# Usage: ./startup.sh [mode] [options]

# Default values
MODE="backtest"
CONFIG_FILE="config/default_config.yaml"
LOG_LEVEL="INFO"
DATA_FETCH="false"
MONITOR_MEMORY="false"
MONITOR_TEMP="false"
USE_GPU="false"

# Function to display help
show_help() {
    echo "Cryptocurrency Trading System"
    echo "Usage: ./startup.sh [options]"
    echo ""
    echo "Options:"
    echo "  --mode <mode>           Operating mode: train, backtest, live, fetch_data (default: backtest)"
    echo "  --config <file>         Path to config file (default: config/default_config.yaml)"
    echo "  --log-level <level>     Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)"
    echo "  --fetch-data            Fetch fresh data before running"
    echo "  --monitor-memory        Enable memory monitoring"
    echo "  --monitor-temp          Enable temperature monitoring"
    echo "  --use-gpu               Enable GPU for training/inference"
    echo "  --output-dir <dir>      Directory for output files (default from config)"
    echo "  --help                  Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./startup.sh --mode train --use-gpu"
    echo "  ./startup.sh --mode backtest --config custom_config.yaml"
    echo "  ./startup.sh --mode fetch_data"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --fetch-data)
            DATA_FETCH="true"
            shift
            ;;
        --monitor-memory)
            MONITOR_MEMORY="true"
            shift
            ;;
        --monitor-temp)
            MONITOR_TEMP="true"
            shift
            ;;
        --use-gpu)
            USE_GPU="true"
            shift
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate mode
if [[ ! "$MODE" =~ ^(train|backtest|live|fetch_data)$ ]]; then
    echo "Error: Invalid mode '$MODE'"
    echo "Valid modes are: train, backtest, live, fetch_data"
    exit 1
fi

# Check if config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Warning: Config file '$CONFIG_FILE' not found. Using default configuration."
fi

# Create necessary directories
mkdir -p logs
mkdir -p data
mkdir -p results
mkdir -p models

# Ensure script is executable
chmod +x crypto_trading/main.py

# Set environment variables
if [[ "$USE_GPU" == "true" ]]; then
    export TF_FORCE_GPU_ALLOW_GROWTH=true
    GPU_ARGS="--use-gpu"
else
    unset TF_FORCE_GPU_ALLOW_GROWTH
    GPU_ARGS=""
fi

# Check if we need to fetch data first
if [[ "$DATA_FETCH" == "true" && "$MODE" != "fetch_data" ]]; then
    echo "Fetching fresh data..."
    python -m crypto_trading.main --mode fetch_data --config "$CONFIG_FILE" --log-level "$LOG_LEVEL"

    # Check if data fetch was successful
    if [[ $? -ne 0 ]]; then
        echo "Error: Data fetch failed. Exiting."
        exit 1
    fi
fi

# Set additional arguments based on options
EXTRA_ARGS=""
if [[ "$MONITOR_MEMORY" == "true" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --monitor-memory"
fi

if [[ "$MONITOR_TEMP" == "true" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --monitor-temperature"
fi

if [[ -n "$OUTPUT_DIR" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --output-dir $OUTPUT_DIR"
fi

# Print startup information
echo "Starting Cryptocurrency Trading System"
echo "- Mode: $MODE"
echo "- Config file: $CONFIG_FILE"
echo "- Log level: $LOG_LEVEL"
echo "- GPU enabled: $USE_GPU"
echo "- Memory monitoring: $MONITOR_MEMORY"
echo "- Temperature monitoring: $MONITOR_TEMP"
echo ""

# Run the main application
python -m crypto_trading.main --mode "$MODE" --config "$CONFIG_FILE" --log-level "$LOG_LEVEL" $GPU_ARGS $EXTRA_ARGS

exit_code=$?

if [[ $exit_code -eq 0 ]]; then
    echo "Execution completed successfully"
else
    echo "Execution failed with exit code $exit_code"
fi

exit $exit_code