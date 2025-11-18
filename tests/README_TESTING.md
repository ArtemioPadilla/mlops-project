# Testing Guide – MLOps Online News Popularity

## 1. Ejecutar TODAS las pruebas

pytest -q


## 2. Ejecutar solo pruebas de preprocesamiento

pytest tests/test_preprocessing -q


## 3. Ejecutar solo pruebas del pipeline completo

pytest tests/test_pipeline -q


## 4. Estructura cubierta por las pruebas

- DataCleaner (unit)
- DataProcessor (unit)
- DataLoader (unit)
- Pipeline end-to-end (integration)

## 5. Cobertura esperada
> Preprocessing + Pipeline cubierto completamente  
> Serving ya incluye 87 tests adicionales
