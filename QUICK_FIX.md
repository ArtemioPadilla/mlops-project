# Quick Fix Guide

## Issue: Dependency Conflict

You're seeing this error:
```
RuntimeError: Form data requires "python-multipart" to be installed.
It seems you installed "multipart" instead.
```

## Solution

### Option 1: Use the Fix Script (Recommended)

```bash
bash scripts/fix_dependencies.sh
```

This will:
1. Remove conflicting `multipart` package
2. Upgrade pip
3. Install package in development mode
4. Install all requirements including `python-multipart`

### Option 2: Manual Fix

```bash
# 1. Remove conflicting package
pip uninstall multipart

# 2. Install package in development mode
pip install -e .

# 3. Install all requirements
pip install -r requirements.txt

# 4. Verify installation
pip list | grep python-multipart
```

## Verify the Fix

After running the fix, test it:

```bash
# Run serving tests
make test-serving

# Expected output: 83 tests should run
```

## Environment Setup (Fixed)

The `make create_environment` command has been fixed to work on macOS/Linux:

```bash
# Create virtual environment
make create_environment

# Activate it
source venv/bin/activate

# Install dependencies
bash scripts/fix_dependencies.sh
```

## Common Issues

### Issue: ModuleNotFoundError

**Solution**: Install package in development mode
```bash
pip install -e .
```

### Issue: Tests fail with model not found

**Solution**: This is expected in tests - they use mock models from fixtures

### Issue: Port 8000 already in use

**Solution**: Change port or kill existing process
```bash
# Find process
lsof -i :8000

# Kill it
kill -9 <PID>

# Or use different port
API_PORT=8001 make serve
```

## Next Steps

After fixing dependencies:

1. **Run tests**:
   ```bash
   make test-serving
   ```

2. **Check coverage**:
   ```bash
   make test-coverage
   open htmlcov/index.html
   ```

3. **Start API server**:
   ```bash
   make serve
   open http://localhost:8000/docs
   ```

4. **Test API**:
   ```bash
   make test-api
   ```

## Need More Help?

- Check [docs/serving/troubleshooting.md](docs/serving/troubleshooting.md)
- Review [docs/serving/getting-started.md](docs/serving/getting-started.md)
- Open an issue on GitHub

## Summary of Changes Made

✅ Fixed Makefile `create_environment` to work cross-platform
✅ Created `scripts/fix_dependencies.sh` to resolve conflicts
✅ Created this QUICK_FIX.md guide

All tests and documentation are ready to use once dependencies are fixed!
