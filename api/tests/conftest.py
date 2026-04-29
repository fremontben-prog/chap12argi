import pytest
 
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: tests d'intégration nécessitant les vrais modèles .joblib"
    )