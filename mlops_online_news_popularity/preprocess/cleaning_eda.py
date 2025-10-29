import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Importación de YData Profiling ---
try:
    from ydata_profiling import ProfileReport
except ImportError:
    print("Error: ydata-profiling no está instalado. Por favor, instálalo con: pip install ydata-profiling")
    ProfileReport = None

# =====================================================================
# CLASE: DataExplorer
# =====================================================================

class DataExplorer:
    """
    Clase estática para realizar un Análisis Exploratorio de Datos (EDA)
    y generar reportes de profiling.
    """

    @staticmethod
    def explore_data(data):
        """
        Realiza un EDA básico: muestra info, estadísticas y gráficos.
        """
        print("="*30)
        print("INICIANDO ANÁLISIS EXPLORATORIO (EDA)")
        print("="*30)
        
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
        
        print("="*30)
        print("FIN DE ANÁLISIS EXPLORATORIO (EDA)")
        print("="*30)

    @staticmethod
    def get_numeric_stats(df_numeric: pd.DataFrame) -> pd.DataFrame:
        stats_num = df_numeric.describe().T
        stats_num['skew'] = df_numeric.skew()
        stats_num['kurtosis'] = df_numeric.kurtosis()
        return stats_num

    @staticmethod
    def plot_correlation_matrix(data, title="Matriz de Correlación", save_path=None):
        """
        Grafica un heatmap de correlación y opcionalmente lo guarda en save_path.
        """
        print(f"\n--- Generando Heatmap: {title} ---")
        plt.figure(figsize=(14, 12))
        try:
            corr_matrix = data.corr(numeric_only=True)
            sns.heatmap(corr_matrix, cmap='mako_r', annot=False)
            plt.title(title, fontsize=16)
            plt.tight_layout()
            
            if save_path:
                try:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    plt.savefig(save_path, bbox_inches='tight')
                    print(f"Heatmap guardado en: {save_path}")
                except Exception as e:
                    print(f"Error al guardar el heatmap en {save_path}: {e}")
            
            plt.show()
            print(f"Heatmap '{title}' generado.")
        except Exception as e:
            print(f"No se pudo generar el heatmap de correlación '{title}': {e}")

    @staticmethod
    def generate_profiling_report(data, title, output_dir, filename):
        if ProfileReport is None:
            print(f"SKIPPING: Reporte de Profiling '{title}' (ydata-profiling no encontrado).")
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

# =====================================================================
# CLASE: DataCleaner
# =====================================================================

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()

    def filter_expected_columns(self, expected_cols):
        print("Filtrando columnas esperadas...")
        extra = [c for c in self.df.columns if c not in expected_cols]
        missing = [c for c in expected_cols if c not in self.df.columns]
        if extra: print(f"⚠️ Extras ignoradas: {extra}")
        if missing: print(f"⚠️ Faltan columnas: {missing}")
        self.df = self.df[[c for c in expected_cols if c in self.df.columns]]
        return self

    def force_numeric(self, exclude=["url"]):
        print("Forzando columnas a tipo numérico...")
        for c in self.df.columns:
            if c in exclude:
                continue
            if self.df[c].dtype == "O":
                self.df[c] = (
                    self.df[c].astype(str)
                    .str.replace(",", ".", regex=False)
                    .replace({"nan": np.nan, "None": np.nan, "": np.nan})
                )
            self.df[c] = pd.to_numeric(self.df[c], errors="coerce")
        return self

    def apply_business_rules(self):
        print("Aplicando reglas de negocio...")
        if "timedelta" in self.df:
            self.df["timedelta"] = self.df["timedelta"].clip(0, 731)
        clip_01 = ["n_unique_tokens", "global_subjectivity"]
        for c in clip_01:
            if c in self.df:
                self.df[c] = self.df[c].clip(0, 1)
        return self

    def normalize_lda(self, lda_cols=None):
        if not lda_cols: return self
        print("Normalizando columnas LDA...")
        lda_cols = [c for c in lda_cols if c in self.df]
        if lda_cols:
            s = self.df[lda_cols].sum(axis=1)
            mask = s > 0
            self.df.loc[mask, lda_cols] = self.df.loc[mask, lda_cols].div(s[mask], axis=0)
        return self

    def clean_primary_key(self, key="url"):
        print(f"Limpiando clave primaria '{key}'...")
        if key not in self.df.columns:
            print(f"Advertencia: Clave primaria '{key}' no encontrada para limpiar.")
            return self
            
        self.df = self.df[self.df[key].notna() & (self.df[key] != "")]
        self.df[key] = self.df[key].astype(str).str.strip().str.lower()
        self.df = self.df[self.df[key].str.startswith("http", na=False)]
        return self

    def get_df(self):
        return self.df

# =====================================================================
# FUNCIONES AUXILIARES
# =====================================================================

def classify_numeric_columns(df_numeric: pd.DataFrame) -> (list, list):
    cols_bin = [col for col in df_numeric.columns if set(df_numeric[col].dropna().unique()) <= {0, 1}]
    cols_no_bin = [col for col in df_numeric.columns if col not in cols_bin]
    print(f"Columnas binarias identificadas: {len(cols_bin)}")
    print(f"Columnas numéricas no binarias: {len(cols_no_bin)}")
    return cols_bin, cols_no_bin



# --- Código de prueba ---

if __name__ == "__main__":
    print("Nothing")
