# model_trainer.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Union, Iterable, Any

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_validate
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from mlops_online_news_popularity.preprocess.cleaning_eda import classify_numeric_columns

# =========================================================
# Base: DataProcessor (mínima)
# ---------------------------------------------------------
# Requisitos mínimos: exponer X_train, y_train, X_val, y_val, X_test, y_test
# =========================================================

@dataclass
class DataProcessor:
    """
    Clase base mínima que únicamente almacena los datos ya preprocesados.
    """
    X_train: Union[pd.DataFrame, np.ndarray]
    y_train: Union[pd.Series, np.ndarray]
    X_val:   Union[pd.DataFrame, np.ndarray]
    y_val:   Union[pd.Series, np.ndarray]
    X_test:  Union[pd.DataFrame, np.ndarray]
    y_test:  Union[pd.Series, np.ndarray]


# =========================================================
# ModelTrainer
# =========================================================

class ModelTrainer(DataProcessor):
    """
    Entrenador de modelos con Pipeline de scikit-learn.

    Ejemplo de uso:
    ----------------
    from sklearn.linear_model import Ridge

    trainer = ModelTrainer(
        X_train, y_train, X_val, y_val, X_test, y_test,
        estimator=Ridge(random_state=42),
        scaler="standard",          # "standard" | "robust" | None
        model_name="Ridge Regression"
    )
    trainer.train_model()
    results = trainer.evaluate_model()
    cv = trainer.cross_validate_model(cv=5)
    """

    def __init__(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val:   Union[pd.DataFrame, np.ndarray],
        y_val:   Union[pd.Series, np.ndarray],
        X_test:  Union[pd.DataFrame, np.ndarray],
        y_test:  Union[pd.Series, np.ndarray],
        estimator: BaseEstimator,
        scaler: Optional[str] = None,      # "standard" | "robust" | None
        model_name: Optional[str] = None,
        fit_params: Optional[Dict[str, Any]] = None
    ):
        super().__init__(X_train, y_train, X_val, y_val, X_test, y_test)

        self.model_name = model_name or estimator.__class__.__name__
        self.fit_params = fit_params or {}

        # Construir Pipeline
        self.pipeline = self._create_pipeline(estimator, scaler)

        # Marcadores de estado
        self._is_fitted: bool = False

    # --------------------------
    # Pipeline
    # --------------------------

    def _build_preprocessor(self) -> ColumnTransformer:
        """
        Crea el ColumnTransformer basado en las columnas de self.X_train, para aplicar transformaciones de forma avanzada por tipo de elemento en cada columna
        """
        
        # Buscando columnas numericas int o float
        numeric_features = self.X_train.select_dtypes(include=np.number).columns
        
        # clasificamos columnas de acuerdo al tipo 
        cols_bin, cols_no_bin = classify_numeric_columns(self.X_train[numeric_features]) 

        # Procesamos columnas numericas no binarias 
        numeric_non_binary_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('power', PowerTransformer(method='yeo-johnson')),
            ('scaler', StandardScaler())
        ])
        
        # Procesamos columnas binarias 
        binary_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent'))
        ])

        # 5. Preparamos el procesamiento principal
        preprocessor = ColumnTransformer(transformers=[
            ('num_non_bin', numeric_non_binary_transformer, cols_no_bin),
            ('num_bin', binary_transformer, cols_bin)
        ], remainder='passthrough')
        
        return preprocessor


    def _create_pipeline(
        self,
        estimator: BaseEstimator,
        preprocessor: ColumnTransformer
    ) -> Pipeline:
        """
        Creamos el pipeline final con el modelo y el preprocesamiento
        """
        # Unimos preprocesador y modelo en un Pipeline
        steps = [
            ("preprocessor", preprocessor),
            ("model", estimator)
        ]
        return Pipeline(steps)

    # --------------------------
    # Entrenamiento
    # --------------------------
    def train_model(self) -> "ModelTrainer":
        """
        Entrena el Pipeline con el conjunto de entrenamiento.
        """
        self.pipeline.fit(self.X_train, self.y_train, **self.fit_params)
        self._is_fitted = True
        return self

    # --------------------------
    # Evaluación
    # --------------------------
    @staticmethod
    def _regression_metrics(
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """
        Métricas de regresión estándar.
        """
        rmse = root_mean_squared_error(y_true, y_pred, squared=False)
        mae  = mean_absolute_error(y_true, y_pred)
        r2   = r2_score(y_true, y_pred)
        return {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}

    def evaluate_model(self) -> Dict[str, Dict[str, float]]:
        """
        Evalúa el modelo en train/val/test y devuelve métricas.
        """
        if not self._is_fitted:
            raise RuntimeError("Debes entrenar el modelo antes de evaluar (train_model).")

        # Predicciones
        yhat_train = self.pipeline.predict(self.X_train)
        yhat_val   = self.pipeline.predict(self.X_val)
        yhat_test  = self.pipeline.predict(self.X_test)

        # Métricas
        metrics = {
            "train": self._regression_metrics(self.y_train, yhat_train),
            "val":   self._regression_metrics(self.y_val,   yhat_val),
            "test":  self._regression_metrics(self.y_test,  yhat_test),
        }

        # Gap útil para diagnosticar sobre/infraentrenamiento
        metrics["gaps"] = {
            "rmse_train_val": metrics["train"]["rmse"] - metrics["val"]["rmse"],
            "rmse_val_test":  metrics["val"]["rmse"]   - metrics["test"]["rmse"],
        }

        # Resumen legible
        self._pretty_print_metrics(metrics)
        return metrics

    def _pretty_print_metrics(self, metrics: Dict[str, Dict[str, float]]) -> None:
        name = self.model_name
        print("\n" + "=" * 70)
        print(f"RESULTADOS - {name}")
        print("=" * 70)
        for split in ("train", "val", "test"):
            m = metrics[split]
            print(f"{split.upper():5s} | RMSE: {m['rmse']:.4f} | MAE: {m['mae']:.4f} | R²: {m['r2']:.4f}")
        gaps = metrics["gaps"]
        print("-" * 70)
        print(f"Gap RMSE (train - val): {gaps['rmse_train_val']:.4f}")
        print(f"Gap RMSE (val   - test): {gaps['rmse_val_test']:.4f}")
        print("=" * 70)

    # --------------------------
    # Validación cruzada
    # --------------------------
    def cross_validate_model(
        self,
        cv: int = 5,
        scoring: Optional[Union[str, Iterable[str], Dict[str, Any]]] = None,
        n_jobs: Optional[int] = None,
        return_train_score: bool = True
    ) -> Dict[str, Any]:
        """
        Ejecuta validación cruzada sobre (X_train, y_train) o, si se prefiere,
        concatena train+val antes de llamar a este método en tu flujo.

        Parameters
        ----------
        cv : int
            Número de particiones de CV.
        scoring : None | str | Iterable[str] | dict
            *Scorers* de scikit-learn. Si es None, se usa un conjunto por defecto.
        n_jobs : None | int
            Paralelismo en cross_validate.
        return_train_score : bool
            Devuelve métricas de train en la CV.

        Returns
        -------
        Dict con promedios y desviaciones estándar por métrica.
        """
        if scoring is None:
            # Conjunto razonable para regresión
            scoring = {
                "rmse": "neg_root_mean_squared_error",
                "mae":  "neg_mean_absolute_error",
                "r2":   "r2",
            }

        if not self._is_fitted:
            # cross_validate ajusta internamente; no es obligatorio pre-ajustar,
            # pero mantener esto explícito ayuda a evitar confusiones.
            pass

        # Ejecutar CV sobre TRAIN (recomendado) o sobre train+val si así los pasaste al inicializar
        cv_results = cross_validate(
            estimator=self.pipeline,
            X=self.X_train,
            y=self.y_train,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            return_train_score=return_train_score,
        )

        # Resumen: promedios (+ desviación) por métrica
        summary: Dict[str, Dict[str, float]] = {}
        for key, values in cv_results.items():
            if key.startswith("test_") or key.startswith("train_"):
                metric = key.split("_", 1)[1]
                vals = np.array(values, dtype=float)
                # scikit-learn usa signo negativo para "neg_*"
                if metric in ("rmse", "mae"):
                    vals = -vals
                summary[key] = {
                    "mean": float(vals.mean()),
                    "std":  float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                }

        # Impresión formateada
        self._pretty_print_cv(summary, cv)
        return {"raw": cv_results, "summary": summary}

    def _pretty_print_cv(self, summary: Dict[str, Dict[str, float]], cv: int) -> None:
        print("\n" + "=" * 70)
        print(f"VALIDACIÓN CRUZADA (cv={cv}) - {self.model_name}")
        print("=" * 70)
        # Ordenar para mostrar TRAIN primero, luego TEST
        for split in ("train", "test"):
            for metric in ("rmse", "mae", "r2"):
                key = f"{split}_{metric}"
                if key in summary:
                    m = summary[key]
                    print(f"{key:12s} | mean: {m['mean']:.4f} | std: {m['std']:.4f}")
        print("=" * 70)
