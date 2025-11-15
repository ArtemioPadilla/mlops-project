#!/bin/bash
# Fix dependency conflicts and install all requirements

set -e  # Exit on error

echo "========================================="
echo "Fixing Dependencies"
echo "========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3.10 --version 2>&1)
echo "✓ $python_version"
echo ""

# Uninstall conflicting package
echo "Removing conflicting 'multipart' package..."
python3.10 -m pip uninstall -y multipart 2>/dev/null || echo "  'multipart' not installed (OK)"
echo ""

# Upgrade pip
echo "Upgrading pip..."
python3.10 -m pip install --upgrade pip
echo ""

# Install package in development mode
echo "Installing package in development mode..."
python3.10 -m pip install -e .
echo ""

# Install all requirements
echo "Installing requirements..."
python3.10 -m pip install -r requirements.txt
echo ""

echo "========================================="
echo "✓ Dependencies fixed successfully!"
echo "========================================="
echo ""
echo "You can now run:"
echo "  make test-serving    # Run serving tests"
echo "  make test-coverage   # Run tests with coverage"
echo "  make serve           # Start API server"
echo ""
