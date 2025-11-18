#!/usr/bin/env bash

echo "=========================================="
echo "  🚀 Ejecutando Pipeline MLOps Completo"
echo "=========================================="

# 1. Activar entorno virtual
echo "🔹 Activando entorno virtual..."
source .venv310/Scripts/activate

# 2. Ejecutar Python en bloque
python - << 'EOF'
from mlops_online_news_popularity.preprocessing.data_processor import DataProcessor
from mlops_online_news_popularity.modeling.train import train_model
from pathlib import Path

print("🔹 Cargando y procesando dataset...")
dp = DataProcessor("data/raw/online_news_modified.csv")
dp.process()

print("🔹 Entrenando modelo...")
output = train_model(dp.X_train, dp.y_train, Path("models"))

print("\n✅ Modelo guardado en:", output)
EOF

echo "=========================================="
echo "  🎉 PIPELINE TERMINADO EXITOSAMENTE"
echo "=========================================="
