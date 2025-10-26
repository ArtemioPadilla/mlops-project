# ================================
# 2. DataCleaner
# ================================

import pandas as pd
import numpy as np


class DataCleaner:
    def __init__(self, df):
        self.df = df

    def filter_expected_columns(self, expected_cols):
        extra = [c for c in self.df.columns if c not in expected_cols]
        missing = [c for c in expected_cols if c not in self.df.columns]
        if extra: print("⚠️ Extras ignoradas:", extra)
        if missing: print("⚠️ Faltan columnas:", missing)
        self.df = self.df[[c for c in expected_cols if c in self.df.columns]]
        return self

    def force_numeric(self, exclude=["url"]):
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
        # ejemplo timedelta
        if "timedelta" in self.df:
            self.df["timedelta"] = self.df["timedelta"].clip(0, 731)
        # clip proporciones
        clip_01 = ["n_unique_tokens", "global_subjectivity"]
        for c in clip_01:
            if c in self.df:
                self.df[c] = self.df[c].clip(0, 1)
        return self

    def winsorize_columns(self, exclude=set()):
        def winsorize(s, low=0.01, high=0.99):
            if s.notna().sum() == 0: return s
            ql, qh = s.quantile(low), s.quantile(high)
            return s.clip(ql, qh)
        num_cols = [c for c in self.df.select_dtypes(include=[np.number]).columns if c not in exclude]
        for c in num_cols:
            self.df[c] = winsorize(self.df[c])
        return self

    def normalize_lda(self, lda_cols=None):
        if not lda_cols: return self
        lda_cols = [c for c in lda_cols if c in self.df]
        if lda_cols:
            s = self.df[lda_cols].sum(axis=1)
            mask = s > 0
            self.df.loc[mask, lda_cols] = self.df.loc[mask, lda_cols].div(s[mask], axis=0)
        return self

    def clean_primary_key(self, key="url"):
        self.df = self.df[self.df[key].notna() & (self.df[key] != "")]
        self.df[key] = self.df[key].astype(str).str.strip().str.lower()
        self.df = self.df[self.df[key].str.startswith("http", na=False)]
        return self

    def impute_missing_values(self):
        for col in self.df.columns[1:]:
            skew = self.df[col].skew()
            if -1 < skew < 1:
                val = self.df[col].mean()
                self.df[col] = self.df[col].fillna(val)
            else:
                val = self.df[col].median()
                self.df[col] = self.df[col].fillna(val)
        return self

    def get_df(self):
        return self.df