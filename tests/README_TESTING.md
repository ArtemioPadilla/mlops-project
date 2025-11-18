# Ejecutar todas las pruebas
pytest -q

# Ejecutar solo una carpeta
pytest tests/test_preprocessing -q

# Ver pruebas con cobertura
pytest --cov=mlops_online_news_popularity -q
