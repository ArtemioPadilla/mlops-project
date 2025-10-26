# main.py
import os
from src import DataLoader, DataCleaner, DataComparator

# ================================
# Definir rutas de trabajo
# ================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # sube un nivel desde src/
DATA_DIR = os.path.join(BASE_DIR, "Data")

def main():
    loader = DataLoader()

    # ================================
    # 1. Carga de datos
    # ================================
    orig = loader.load_csv(os.path.join(DATA_DIR, "online_news_original.csv"))
    mod = loader.load_csv(os.path.join(DATA_DIR, "online_news_modified.csv"))

    # ================================
    # 2. Definir columnas esperadas
    # ================================
    expected_cols = [
        "url","timedelta","n_tokens_title","n_tokens_content","n_unique_tokens",
        "n_non_stop_words","n_non_stop_unique_tokens","num_hrefs","num_self_hrefs",
        "num_imgs","num_videos","average_token_length","num_keywords",
        "data_channel_is_lifestyle","data_channel_is_entertainment","data_channel_is_bus",
        "data_channel_is_socmed","data_channel_is_tech","data_channel_is_world",
        "kw_min_min","kw_max_min","kw_avg_min","kw_min_max","kw_max_max","kw_avg_max",
        "kw_min_avg","kw_max_avg","kw_avg_avg",
        "self_reference_min_shares","self_reference_max_shares","self_reference_avg_sharess",
        "weekday_is_monday","weekday_is_tuesday","weekday_is_wednesday","weekday_is_thursday",
        "weekday_is_friday","weekday_is_saturday","weekday_is_sunday","is_weekend",
        "LDA_00","LDA_01","LDA_02","LDA_03","LDA_04",
        "global_subjectivity","global_sentiment_polarity",
        "global_rate_positive_words","global_rate_negative_words",
        "rate_positive_words","rate_negative_words",
        "avg_positive_polarity","min_positive_polarity","max_positive_polarity",
        "avg_negative_polarity","min_negative_polarity","max_negative_polarity",
        "title_subjectivity","title_sentiment_polarity",
        "abs_title_subjectivity","abs_title_sentiment_polarity",
        "shares"
    ]

    extra_cols = [c for c in mod.columns if c not in expected_cols]
    missing_cols = [c for c in expected_cols if c not in mod.columns]

    if extra_cols:
        print("⚠️ Columnas extra ignoradas:", extra_cols)
    if missing_cols:
        print("⚠️ Columnas esperadas que no encontré (seguiré sin ellas):", missing_cols)

    keep_cols = [c for c in expected_cols if c in mod.columns]
    mod = mod[keep_cols]

    # ================================
    # 3. Limpieza
    # ================================
    cleaner = DataCleaner(mod)
    mod_clean = (cleaner
        .filter_expected_columns(expected_cols=keep_cols)
        .force_numeric()
        .apply_business_rules()
        .winsorize_columns()
        .normalize_lda(["LDA_00", "LDA_01", "LDA_02", "LDA_03", "LDA_04"])
        .clean_primary_key()
        .impute_missing_values()
        .get_df())

    loader.save_csv(mod_clean, os.path.join(DATA_DIR, "online_news_cleaned.csv"))

    # ================================
    # 4. Comparación con original
    # ================================
    comparator = DataComparator(orig, mod_clean)
    report = (comparator
        .compare_stats()
        .add_differences()
        .missing_values_ratio()
        .export_report(os.path.join(DATA_DIR, "comparacion_final.csv")))

    print(report.head(15))


if __name__ == "__main__":
    main()
