import streamlit as st
import os
import pandas as pd

from custom_styles import (
    load_custom_css, create_header, create_section, 
    info_box, success_box, warning_box, error_box,
    create_stat_card, create_comparison_table, add_footer
)

from eda import EDAApp
from src.preprocessing.preprocess import CartAbandonmentPreprocessor
from src.preprocessing.feature_engineering import FeatureEngineer
from modeling_evaluation import ModelingEvaluationTab

class BaseTab:
    def __init__(self, name):
        self.name = name

    @staticmethod
    @st.cache_data
    def load_data(file_path):
        return pd.read_csv(file_path)

    @staticmethod
    def ensure_data_directory():
        os.makedirs("data", exist_ok=True)

class EDATab(BaseTab):
    def __init__(self):
        super().__init__("📊 EDA (Raw Data)")

    def run(self):
        create_header("Exploratory Data Analysis", "Raw Dataset Insights", "📊")
        
        RAW_PATH = os.path.join("data", "cart_abandonment_dataset.csv")
        df = self.load_data(RAW_PATH)

        if df is None:
            error_box(f"Raw dataset not found at {RAW_PATH}")
            return

        app = EDAApp(df)
        app.dataset_overview()
        app.missing_values()

        create_section("Data Analysis", "🔍")
        tabs = st.tabs(["📈 Categorical", "📊 Numerical", "🎯 Target", "🔗 Correlation"])

        with tabs[0]:
            col = st.selectbox("Select Categorical Column",
                               app.categorical_cols + app.binary_categorical_cols,
                               key="raw_cat")
            app.categorical_analysis(col)

        with tabs[1]:
            col = st.selectbox("Select Numerical Column", app.numerical_cols, key="raw_num")
            app.numerical_analysis(col)

        with tabs[2]:
            app.target_analysis()

        with tabs[3]:
            app.correlation_analysis()

        st.markdown("---")
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Raw Data as CSV",
            data=csv,
            file_name="cart_abandonment_raw.csv",
            mime="text/csv",
        )


class PreprocessingTab(BaseTab):
    def __init__(self):
        super().__init__("🔧 Preprocessing")

    def run(self):
        create_header("Data Preprocessing", "Cleaning & Transformation", "🔧")

        RAW_PATH = os.path.join("data", "cart_abandonment_dataset.csv")
        PROCESSED_PATH = os.path.join("data", "cart_abandonment_preprocessed.csv")
        ENCODERS_PATH = os.path.join("data", "label_encoders.json")
        SCALER_PATH = os.path.join("data", "scaler_info.json")

        df_raw = self.load_data(RAW_PATH)
        if df_raw is None:
            error_box(f"Raw dataset not found at {RAW_PATH}")
            return

        if st.button("🚀 Run Preprocessing", type="primary", use_container_width=True):
            with st.spinner("⏳ Processing data..."):
                preprocessor = CartAbandonmentPreprocessor(RAW_PATH, PROCESSED_PATH, ENCODERS_PATH, SCALER_PATH)
                preprocessor.run()
            success_box("✅ Preprocessing completed successfully!")

        df = self.load_data(PROCESSED_PATH)
        if df is None:
            info_box("Run preprocessing to generate processed data")
            return

        app = EDAApp(df)
        app.dataset_overview()
        app.missing_values()

        create_section("Preprocessed Data Analysis", "🔍")
        tabs = st.tabs(["📈 Categorical", "📊 Numerical", "🎯 Target", "🔗 Correlation"])

        with tabs[0]:
            col = st.selectbox("Select Categorical Column",
                               app.categorical_cols + app.binary_categorical_cols,
                               key="pre_cat")
            app.categorical_analysis(col)

        with tabs[1]:
            col = st.selectbox("Select Numerical Column", app.numerical_cols, key="pre_num")
            app.numerical_analysis(col)

        with tabs[2]:
            app.target_analysis()

        with tabs[3]:
            app.correlation_analysis()

        st.markdown("---")
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Processed Data as CSV",
            data=csv,
            file_name="cart_abandonment_processed.csv",
            mime="text/csv",
        )


class FeatureEngineeringTab(BaseTab):
    def __init__(self):
        super().__init__("✨ Feature Engineering")

    def run(self):
        create_header("Feature Engineering", "Creating & Optimizing Features", "✨")

        PROCESSED_PATH = os.path.join("data", "cart_abandonment_preprocessed.csv")
        FEATURED_PATH = os.path.join("data", "cart_abandonment_featured.csv")
        ENCODERS_PATH = os.path.join("data", "label_encoders.json")
        PCA_LOADINGS_PATH = os.path.join("data", "pca_loadings.csv")

        df_processed = self.load_data(PROCESSED_PATH)
        if df_processed is None:
            error_box("Preprocessed dataset not found. Please run preprocessing first.")
            return

        if st.button("⚙️ Run Feature Engineering", type="primary", use_container_width=True):
            with st.spinner("⏳ Engineering features..."):
                fe = FeatureEngineer(PROCESSED_PATH, ENCODERS_PATH, FEATURED_PATH)
                fe.create_features()
                fe.apply_pca()
                fe.correlation_report()
                fe.save_features()
            success_box("✅ Feature engineering completed successfully!")

        df = self.load_data(FEATURED_PATH)
        if df is None:
            info_box("Run feature engineering to generate featured data")
            return

        sub_tabs = st.tabs(["🆕 New Features"])

        with sub_tabs[0]:
            create_section("Newly Added Features", "🆕")
            new_features = [
                "engagement_intensity", "scroll_engagement", "is_weekend",
                "has_multiple_items", "has_high_engagement", "research_behavior",
                "quick_browse", "engagement_score", "peak_hours", "returning_peak",
                "day_sin", "day_cos", "time_sin", "time_cos"
            ]

            added = [f for f in new_features if f in df.columns]

            if added:
                feature_info = []
                for feature in added:
                    desc = df[feature].describe().to_dict()
                    feature_info.append({
                        "Feature": feature,
                        "Type": str(df[feature].dtype),
                        "Mean": round(desc.get("mean", 0), 3) if "mean" in desc else "-",
                        "Std": round(desc.get("std", 0), 3) if "std" in desc else "-",
                        "Min": round(desc.get("min", 0), 3) if "min" in desc else "-",
                        "Max": round(desc.get("max", 0), 3) if "max" in desc else "-",
                        "NA Count": int(df[feature].isna().sum())
                    })

                create_comparison_table(pd.DataFrame(feature_info))
            else:
                info_box("No new features detected.")

            create_section("Full Dataset with Features", "📊")
            app = EDAApp(df)
            app.dataset_overview()
            app.missing_values()

            st.markdown("---")
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Featured Data as CSV",
                data=csv,
                file_name="cart_abandonment_featured.csv",
                mime="text/csv",
            )

class CartAbandonmentDashboard:
    def __init__(self):
        self.tabs = [
            EDATab(),
            PreprocessingTab(),
            FeatureEngineeringTab(),
            ModelingEvaluationTab()  
        ]

    def run(self):
        st.set_page_config(
            page_title="Cart Abandonment Dashboard",
            page_icon="🛒",
            layout="wide",
            initial_sidebar_state="collapsed"
        )

        load_custom_css()

        st.markdown("""
        <div style='text-align: center; padding: 30px 0 20px 0;'>
            <h1 style='
                background: linear-gradient(135deg, #60a5fa, #3b82f6);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                font-size: 3em;
                margin: 0;
            '>🛒 Cart Abandonment Dashboard</h1>
            <p style='color: #94a3b8; font-size: 1.1em; margin: 10px 0;'>
                Comprehensive Analysis & Predictive Modeling
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        main_tabs = st.tabs([tab.name for tab in self.tabs])

        BaseTab.ensure_data_directory()

        for tab, ui in zip(main_tabs, self.tabs):
            with tab:
                ui.run()

        add_footer()


if __name__ == "__main__":
    dashboard = CartAbandonmentDashboard()
    dashboard.run()