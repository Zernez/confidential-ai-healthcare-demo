#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# Benchmark Suite Runner
# ═══════════════════════════════════════════════════════════════════════════
#
# Usage:
#   ./run_benchmark.sh              # Run 20 iterations
#   ./run_benchmark.sh -n 10        # Run 10 iterations
#   ./run_benchmark.sh --plot-only  # Only generate plots from existing data
#
# ═══════════════════════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check for required packages
check_dependencies() {
    echo "Checking dependencies..."
    
    # Check matplotlib
    python3 -c "import matplotlib" 2>/dev/null || {
        echo "Installing matplotlib..."
        pip install matplotlib seaborn numpy --quiet
    }
    
    echo "✓ Dependencies OK"
}

# Main
main() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║   Benchmark Suite - Confidential AI Healthcare Demo        ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    
    check_dependencies
    
    # Run benchmark suite
    python3 main.py "$@"
}

main "$@"
