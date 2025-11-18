import pytest
import numpy as np

class MockModel:
    feature_names_in_ = [f"f{i}" for i in range(5)]

    def predict(self, X):
        return np.array([123.0])

@pytest.fixture
def mock_model():
    return MockModel()
