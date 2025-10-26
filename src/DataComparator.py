import pandas as pd

# ==============================
# 3. DataComparator
# ==============================
class DataComparator:
    def __init__(self, orig, clean):
        self.orig = orig
        self.clean = clean
        self.report = pd.DataFrame()  # evita problemas con None

    def compare_stats(self):
        """Calcula estadísticas descriptivas (media y mediana)."""
        self.report = pd.DataFrame({
            "mean_orig": self.orig.mean(numeric_only=True),
            "mean_clean": self.clean.mean(numeric_only=True),
            "median_orig": self.orig.median(numeric_only=True),
            "median_clean": self.clean.median(numeric_only=True)
        })
        return self

    def add_differences(self):
        """Agrega diferencias absolutas entre original y limpio."""
        if self.report.empty:
            raise ValueError("Primero ejecuta compare_stats() antes de add_differences().")
        self.report["diff_mean"] = (self.report["mean_clean"] - self.report["mean_orig"]).abs()
        self.report["diff_median"] = (self.report["median_clean"] - self.report["median_orig"]).abs()
        return self

    def missing_values_ratio(self):
        """Calcula proporción de valores faltantes en %."""
        self.report["missing_orig_%"] = (self.orig.isna().sum() / len(self.orig)) * 100
        self.report["missing_clean_%"] = (self.clean.isna().sum() / len(self.clean)) * 100
        return self

    def export_report(self, path):
        """Exporta el reporte a CSV."""
        if self.report.empty:
            raise ValueError("No hay reporte que exportar. Ejecuta los métodos primero.")
        self.report.to_csv(path, index=False)
        print(f"📊 Reporte exportado a {path}")
        return self.report
