"""
Data exploration module for EDA and profiling.

This module provides the DataExplorer class for performing exploratory data analysis
and generating profiling reports.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# --- Importación de YData Profiling ---
try:
    from ydata_profiling import ProfileReport
except ImportError:
    print(
        "Warning: ydata-profiling no está instalado. "
        "Por favor, instálalo con: pip install ydata-profiling"
    )
    ProfileReport = None


class DataExplorer:
    """
    Static class for performing Exploratory Data Analysis (EDA) and generating profiling reports.

    This class provides methods for basic statistical analysis, correlation visualization,
    and automated profiling reports using ydata-profiling.
    """

    @staticmethod
    def explore_data(data):
        """
        Perform basic EDA: show info, statistics, and plots.

        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame to explore
        """
        print("=" * 30)
        print("INICIANDO ANÁLISIS EXPLORATORIO (EDA)")
        print("=" * 30)

        print("\n--- Información General del DataFrame ---")
        data.info()

        print("\n--- Primeras 5 Filas ---")
        print(data.head())

        print("\n--- Estadísticas Descriptivas ---")
        try:
            numeric_features = data.select_dtypes(include=np.number)
            stats_num = DataExplorer.get_numeric_stats(numeric_features)
            print(stats_num)
        except Exception as e:
            print(f"No se pudieron calcular estadísticas descriptivas: {e}")

        DataExplorer.plot_correlation_matrix(data, title="Matriz de Correlación (Datos Crudos)")

        print("=" * 30)
        print("FIN DE ANÁLISIS EXPLORATORIO (EDA)")
        print("=" * 30)

    @staticmethod
    def get_numeric_stats(df_numeric: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive statistics for numeric columns.

        Parameters
        ----------
        df_numeric : pd.DataFrame
            DataFrame with numeric columns only

        Returns
        -------
        pd.DataFrame
            DataFrame with statistics including skewness and kurtosis
        """
        stats_num = df_numeric.describe().T
        stats_num["skew"] = df_numeric.skew()
        stats_num["kurtosis"] = df_numeric.kurtosis()
        return stats_num

    @staticmethod
    def plot_correlation_matrix(data, title="Matriz de Correlación", save_path=None):
        """
        Plot correlation matrix heatmap and optionally save it.

        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame to analyze
        title : str, optional
            Title for the heatmap
        save_path : str, optional
            Path to save the heatmap image
        """
        print(f"\n--- Generando Heatmap: {title} ---")
        plt.figure(figsize=(14, 12))
        try:
            corr_matrix = data.corr(numeric_only=True)
            sns.heatmap(corr_matrix, cmap="mako_r", annot=False)
            plt.title(title, fontsize=16)
            plt.tight_layout()

            if save_path:
                try:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    plt.savefig(save_path, bbox_inches="tight")
                    print(f"Heatmap guardado en: {save_path}")
                except Exception as e:
                    print(f"Error al guardar el heatmap en {save_path}: {e}")

            plt.show()
            print(f"Heatmap '{title}' generado.")
        except Exception as e:
            print(f"No se pudo generar el heatmap de correlación '{title}': {e}")

    @staticmethod
    def generate_profiling_report(data, title, output_dir, filename):
        """
        Generate an automated profiling report using ydata-profiling.

        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame to profile
        title : str
            Title for the report
        output_dir : str
            Directory to save the report
        filename : str
            Filename for the report (e.g., 'report.html')
        """
        if ProfileReport is None:
            print(f"SKIPPING: Reporte de Profiling '{title}' " "(ydata-profiling no encontrado).")
            return

        output_path = os.path.join(output_dir, filename)
        print(f"\n--- Generando Reporte de Profiling: {title} ---")
        print(f"Guardando en: {output_path}")
        os.makedirs(output_dir, exist_ok=True)

        try:
            profile = ProfileReport(data, title=title, minimal=True)
            profile.to_file(output_path)
            print(f"Reporte '{filename}' guardado exitosamente.")
        except Exception as e:
            print(f"Error al generar el reporte de profiling: {e}")
