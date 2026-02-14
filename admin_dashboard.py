import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pickle
import os
import sys
from pathlib import Path
import json
from analytics_helper import run_analytics_dashboard

# Import custom styles
from custom_styles import (
    load_custom_css, create_header, create_section, 
    info_box, success_box, warning_box, error_box,
    create_stat_card, create_comparison_table, add_footer
)

class BaseTab:
    def __init__(self, name):
        self.name = name

    @staticmethod
    @st.cache_data
    def load_data(file_path):
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
        return None

    @staticmethod
    def ensure_data_directory():
        os.makedirs("data", exist_ok=True)
        os.makedirs("saved_models", exist_ok=True)


class AdminDashboardTab(BaseTab):
    def __init__(self):
        super().__init__("📊 Admin Dashboard")

    def get_correct_paths(self):
        """Get correct file paths based on project structure"""
        current_dir = Path(__file__).parent
        project_root = current_dir.parent
        
        paths = {
            'featured_data': project_root / "data" / "cart_abandonment_featured.csv",
            'original_data': project_root / "data" / "cart_abandonment_dataset.csv", 
            'test_data': project_root / "test_data" / "test_data_for_prediction.csv",
            'models_dir': project_root / "src" / "models"
        }
        
        return paths

    def load_datasets(self):
        self.paths = self.get_correct_paths()
        self.df = self.load_data(self.paths['featured_data'])
        self.df1 = self.load_data(self.paths['original_data'])
        self.df2 = self.load_data(self.paths['test_data'])
        self.feature_names = []


    def calculate_real_metrics(self):
        if self.df1 is None:
            # Fallback to normalized data if original not available
            return self._calculate_metrics_from_normalized()
        
        # Use original dataset for accurate business metrics
        metrics = {
            'total_sessions': len(self.df1),
            'abandonment_rate': self.df1['abandoned'].mean() * 100,
            'avg_cart_value': self.df1['cart_value'].mean(),
            'total_abandoned': self.df1['abandoned'].sum(),
            'return_user_rate': self.df1['return_user'].mean() * 100,
            'avg_engagement': self.df1['num_pages_viewed'].mean(),  # Using pages viewed as engagement proxy
            'payment_reach_rate': self.df1['if_payment_page_reached'].mean() * 100,
            'avg_pages_viewed': self.df1['num_pages_viewed'].mean(),
            'avg_session_duration': self.df1['session_duration'].mean(),
            'recovery_potential': self.df1[self.df1['abandoned'] == 1]['cart_value'].sum(),
            'total_revenue_lost': self.df1[self.df1['abandoned'] == 1]['cart_value'].sum(),
            'avg_abandoned_cart_value': self.df1[self.df1['abandoned'] == 1]['cart_value'].mean(),
            'conversion_rate': (1 - self.df1['abandoned'].mean()) * 100
        }
        return metrics

    def render_dashboard_overview(self):
        """Render main dashboard with REAL data"""
        create_header("CAIRE Analytics", "Cart Abandonment Intelligence & Recovery Engine", "📊")
        
        if self.df is None:
            error_box("No data available. Please ensure featured data exists in the data folder.")
            return
        
        metrics = self.calculate_real_metrics()
        create_section("Key Performance Indicators", "📈")
        pastel_colors = {
            'blue': '#A7C7E7',     
            'red': '#F8BBD0',      
            'green': '#C8E6C9',     
            'yellow': '#FFEAA7',  
            'purple': '#D4BFF9',    
            'teal': '#B2EBF2',    
            'orange': '#FFCCBC',  
            'lime': '#E6EE9C'       
        }
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            create_stat_card(
                "Total Sessions", 
                f"{metrics['total_sessions']:,}",
                "👥", 
                pastel_colors['blue']
            )
        
        with col2:
            create_stat_card(
                "Abandonment Rate", 
                f"{metrics['abandonment_rate']:.1f}%",
                "📊", 
                pastel_colors['red']
            )
        
        with col3:
            cart_val = metrics['avg_cart_value']
            display_val = f"${cart_val:,.2f}" if cart_val > 10 else f"{cart_val:.3f}"
            create_stat_card(
                "Avg Cart Value", 
                display_val,
                "🛒",  
                pastel_colors['green']
            )
        
        with col4:
            create_stat_card(
                "Abandoned Carts", 
                f"{metrics['total_abandoned']:,}",
                "⚠️", 
                pastel_colors['yellow']
            )
        
       
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            create_stat_card(
                "Return Users", 
                f"{metrics['return_user_rate']:.1f}%",
                "🔄",  
                pastel_colors['purple']
            )
        
        with col2:
            create_stat_card(
                "Avg Engagement", 
                f"{metrics['avg_engagement']:.2f}",
                "📈", 
                pastel_colors['teal']
            )
        
        with col3:
            create_stat_card(
                "Payment Reach", 
                f"{metrics['payment_reach_rate']:.1f}%",
                "💳",
                pastel_colors['orange']
            )
        
        with col4:
            recovery_pot = metrics['recovery_potential']
            display_pot = f"${recovery_pot:,.0f}" if recovery_pot > 0 else "$0"
            create_stat_card(
                "Recovery Potential", 
                display_pot,
                "💰", 
                pastel_colors['lime']
            )

    def render_behavior_analysis(self):
        create_section("User Behavior Analysis", "🔍")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Abandonment by engagement level
            st.subheader("🎯 Abandonment by Engagement Level")
            
            if 'engagement_score' in self.df.columns:
                # Create engagement segments from real data
                engagement_bins = [-1, -0.5, 0, 0.5, 1]
                engagement_labels = ['Very Low', 'Low', 'Medium', 'High']
                
                self.df['engagement_level'] = pd.cut(
                    self.df['engagement_score'], 
                    bins=engagement_bins, 
                    labels=engagement_labels
                )
                
                engagement_abandonment = self.df.groupby('engagement_level', observed=True)['abandoned'].mean() * 100
                
                # Use pastel color scale
                fig = px.bar(
                    x=engagement_abandonment.index,
                    y=engagement_abandonment.values,
                    labels={'x': 'Engagement Level', 'y': 'Abandonment Rate (%)'},
                    color=engagement_abandonment.values,
                    color_continuous_scale='blues',  # Pastel blue scale
                    template='plotly_white'
                )
                
                # Customize appearance
                fig.update_traces(
                    marker_line_color='rgba(0,0,0,0.2)',
                    marker_line_width=1,
                    opacity=0.8
                )
                
                fig.update_layout(
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="#2c3e50"),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
        with col2:
            # Cart value distribution
            st.subheader("💰 Cart Value Distribution")
            cart_value_col = 'original_cart_value' if 'original_cart_value' in self.df.columns else 'cart_value'
            
            if cart_value_col in self.df.columns:
                fig = px.histogram(
                    self.df, 
                    x=cart_value_col,
                    nbins=20,
                    labels={cart_value_col: 'Cart Value'},
                    color_discrete_sequence=['#A7C7E7'],  # Pastel blue
                    template='plotly_white'
                )
                
                fig.update_traces(
                    marker_line_color='rgba(0,0,0,0.2)',
                    marker_line_width=1,
                    opacity=0.7
                )
                
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="#2c3e50"),
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)

    def render_user_patterns(self):
        """Render user behavior patterns with pastel colors"""
        create_section("User Behavior Patterns", "📊")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Return vs New User Comparison
            st.subheader("👥 Return vs New User Behavior")
            
            if 'return_user' in self.df.columns:
                user_comparison = self.df.groupby('return_user').agg({
                    'abandoned': 'mean',
                    'engagement_score': 'mean',
                    'session_duration': 'mean'
                })
                user_comparison.index = ['New Users', 'Return Users']
                user_comparison['abandonment_rate'] = user_comparison['abandoned'] * 100
                
                fig = go.Figure()
                
                # Pastel colors for bars
                fig.add_trace(go.Bar(
                    name='Abandonment Rate (%)', 
                    x=user_comparison.index, 
                    y=user_comparison['abandonment_rate'],
                    marker_color='#F8BBD0',  # Pastel pink
                    marker_line_color='rgba(0,0,0,0.2)',
                    marker_line_width=1
                ))
                fig.add_trace(go.Bar(
                    name='Avg Engagement', 
                    x=user_comparison.index, 
                    y=user_comparison['engagement_score'] * 100,
                    marker_color='#A7C7E7',  # Pastel blue
                    marker_line_color='rgba(0,0,0,0.2)',
                    marker_line_width=1
                ))
                
                fig.update_layout(
                    barmode='group', 
                    showlegend=True,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="#2c3e50"),
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Payment Page Analysis
            st.subheader("💳 Payment Page Reach Impact")
            
            if 'if_payment_page_reached' in self.df.columns:
                payment_analysis = self.df.groupby('if_payment_page_reached').agg({
                    'abandoned': 'mean',
                    'cart_value': 'mean'
                })
                payment_analysis.index = ['Did Not Reach', 'Reached Payment']
                payment_analysis['abandonment_rate'] = payment_analysis['abandoned'] * 100
                
                # Use pastel color scale
                fig = px.bar(
                    payment_analysis,
                    y='abandonment_rate',
                    labels={'value': 'Abandonment Rate (%)'},
                    color='abandonment_rate',
                    color_continuous_scale='blues',  # Pastel blue scale
                    template='plotly_white'
                )
                
                fig.update_traces(
                    marker_line_color='rgba(0,0,0,0.2)',
                    marker_line_width=1,
                    opacity=0.8
                )
                
                fig.update_layout(
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="#2c3e50"),
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

    def render_recent_data(self):
        """Render recent session data from raw sessions CSV"""
        create_section("Recent Session Data", "📋")
        
        # Load the raw sessions data
        raw_data_path = "analytics_data/raw_user_sessions.csv"
        
        if not os.path.exists(raw_data_path):
            error_box("No raw session data available yet. Complete some shopping sessions first.")
            
            # Show sample of what data will look like
            st.info("📝 **Sample of what will be recorded:**")
            sample_data = {
                'session_id': ['S12345', 'S12346'],
                'user_id': ['U6789', 'U6790'],
                'timestamp': ['2024-01-15T14:30:00', '2024-01-15T15:45:00'],
                'return_user': ['No', 'Yes'],
                'cart_value': [299.99, 89900.99],
                'cart_items_count': [2, 1],
                'cart_items_names': ['Smartphone, Phone Case', 'Laptop'],
                'abandoned': ['Yes', 'No'],
                'payment_page_reached': ['No', 'Yes'],
                'engagement_score': [7.2, 8.5]
            }
            st.dataframe(pd.DataFrame(sample_data), use_container_width=True)
            return
        
        # Load raw data
        raw_df = pd.read_csv(raw_data_path)
        
        if raw_df.empty:
            warning_box("Raw session data file exists but is empty.")
            return
        
        # Define human-readable columns for display
        display_cols = [
            'session_id', 'user_id', 'timestamp', 'return_user', 
            'cart_value', 'cart_items_count', 'cart_items_names',
            'abandoned', 'payment_page_reached', 'engagement_score'
        ]
        
        # Filter to only available columns
        available_cols = [col for col in display_cols if col in raw_df.columns]
        
        if available_cols:
            # Sort by timestamp (most recent first) and get last 15 sessions
            if 'timestamp' in raw_df.columns:
                raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'])
                recent_data = raw_df.sort_values('timestamp', ascending=False).head(15)
            else:
                recent_data = raw_df.tail(15)
            
            # Format the display
            formatted_data = recent_data[available_cols].copy()
            
            # Format numeric columns
            if 'cart_value' in formatted_data.columns:
                formatted_data['cart_value'] = formatted_data['cart_value'].apply(
                    lambda x: f"₹{x:,.2f}" if pd.notnull(x) else "₹0.00"
                )
            
            if 'engagement_score' in formatted_data.columns:
                formatted_data['engagement_score'] = formatted_data['engagement_score'].apply(
                    lambda x: f"{x:.1f}/10" if pd.notnull(x) else "N/A"
                )
            
            st.dataframe(
                formatted_data, 
                use_container_width=True,
                height=400
            )
            
            # Show some statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Sessions", len(raw_df))
            with col2:
                abandoned_count = len(raw_df[raw_df['abandoned'] == 'Yes']) if 'abandoned' in raw_df.columns else 0
                st.metric("Abandoned Sessions", abandoned_count)
            with col3:
                avg_engagement = raw_df['engagement_score'].mean() if 'engagement_score' in raw_df.columns else 0
                st.metric("Avg Engagement", f"{avg_engagement:.1f}/10")
                
        else:
            warning_box("Required columns not found in raw session data")
            st.write("Available columns in raw data:", list(raw_df.columns))
            
    def run(self):
        """Main method to run the admin dashboard"""
        self.load_datasets()
        
        # Navigation sidebar
        st.sidebar.title("🎯 CAIRE Analytics")
        st.sidebar.markdown("---")
        
        # Data status - REMOVED "Model not available" text
        st.sidebar.subheader("📊 Data Status")
        
        if self.df is not None:
            st.sidebar.success(f"✅ Data: {len(self.df):,} sessions")
        else:
            st.sidebar.error("❌ Data: Not loaded")
            
        # Only show model status if it exists, don't show negative status
        if hasattr(self, 'model') and self.model is not None:
            st.sidebar.success("🤖 Model: Ready for predictions")
                
        
        # Main content
        self.render_dashboard_overview()
        self.render_behavior_analysis()
        self.render_user_patterns()
        self.render_recent_data()


class PredictionTab(BaseTab):
    def __init__(self):
        super().__init__("🎯 Predictions")

    def run(self):
        """Main method to run the prediction tab by importing from prediction_tab.py"""
        try:
            # Import the prediction component
            from prediction_tab import PredictionEngine
            
            # Initialize and run the prediction engine
            prediction_engine = PredictionEngine()
            prediction_engine.run()
            
        except ImportError as e:
            # Fallback UI if import fails
            error_box(f"Prediction module import failed: {e}")
            self.render_fallback_ui()
        except Exception as e:
            error_box(f"Prediction error: {e}")
            self.render_fallback_ui()

    def render_fallback_ui(self):
        """Render fallback UI when prediction system is not available"""
        create_header("Abandonment Predictions", "Real-time Prediction Engine", "🎯")
        
        info_box("""
        **Prediction Engine System**
        
        Our advanced prediction system uses your trained machine learning model to 
        provide real-time cart abandonment predictions with high accuracy.
        
        **Key Features:**
        🔮 **Single Session Prediction** - Real-time abandonment probability for individual sessions
        📊 **Batch Processing** - Upload CSV files for multiple predictions
        📈 **Risk Assessment** - Color-coded risk levels and actionable insights
        💡 **Feature Analysis** - Understand which factors influence predictions
        📥 **Export Results** - Download prediction results for further analysis
        """)
        
        # Show current model status
        st.markdown("---")
        st.subheader("🔍 System Status Check")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Check for model file
            model_paths = [
                "final_manual_model.pkl",
                "saved_models/final_manual_model.pkl",
                "../saved_models/final_manual_model.pkl"
            ]
            
            model_found = False
            for path in model_paths:
                if os.path.exists(path):
                    st.success(f"✅ Model found: {path}")
                    model_found = True
                    break
            
            if not model_found:
                st.error("❌ Model file not found")
        
        with col2:
            # Check for prediction_tab.py
            if os.path.exists("prediction_tab.py"):
                st.success("✅ Prediction module available")
            else:
                st.error("❌ Prediction module not found")
        
        # Quick setup assistance
        st.markdown("---")
        st.subheader("🚀 Quick Setup")
        
        if st.button("🔄 Check System Again", type="secondary"):
            st.rerun()


class SegmentAnalysisTab(BaseTab):
    def __init__(self):
        super().__init__("👥 Segments")

    def run(self):
        create_header("Customer Segmentation", "Intelligent Behavior-Based Segments", "👥")
        
        # Try to import and use the segmentation tab
        try:
            # Import the segmentation component
            from segmentation_tab import SegmentAnalysisTab as SegmentationComponent
            
            # Initialize and run the segmentation tab
            segmentation_component = SegmentationComponent()
            segmentation_component.run()
            
        except ImportError as e:
            # Fallback UI if import fails
            error_box(f"Segmentation module import failed: {e}")
            self.render_fallback_ui()
        except Exception as e:
            error_box(f"Segmentation error: {e}")
            self.render_fallback_ui()

    def render_fallback_ui(self):
        """Render fallback UI when segmentation system is not available"""
        info_box("""
        **Customer Segmentation System**
        
        Our enhanced segmentation system uses advanced machine learning to identify 
        distinct customer groups based on their shopping behavior and characteristics.
        
        **Key Features:**
        🔍 **Behavioral Analysis** - Groups customers by engagement patterns
        🎯 **Smart Identification** - Automatically names segments meaningfully  
        📊 **Priority Scoring** - Ranks segments by recovery urgency
        💡 **Actionable Insights** - Provides segment-specific strategies
        📈 **Visual Analytics** - Interactive charts and comparisons
        """)
        
        # Show current data status
        featured_path = os.path.join("data", "cart_abandonment_featured.csv")
        if os.path.exists(featured_path):
            df = pd.read_csv(featured_path)
            st.success(f"✅ Featured dataset available: {len(df)} records, {len(df.columns)} features")
            
            # Show available features
            with st.expander("📋 Available Features in Dataset"):
                feature_cols = [col for col in df.columns if col not in ['session_id', 'user_id', 'abandoned']]
                st.write(f"**{len(feature_cols)} features available:**")
                cols = st.columns(3)
                for i, feature in enumerate(feature_cols):
                    cols[i % 3].write(f"• {feature}")
        else:
            warning_box("📋 Featured dataset not found. Please run feature engineering first.")
        
        # Quick manual segmentation option
        st.markdown("---")
        st.subheader("Quick Manual Segmentation")
        
        if st.button("🔄 Check System Availability", type="secondary"):
            st.rerun()


class AdvancedAnalyticsTab(BaseTab):
    def __init__(self):
        super().__init__("📈 Analytics")

    def run(self):
        """Run the analytics tab using the standalone function"""
        run_analytics_dashboard()
        
class AdminDashboard:
    def __init__(self):
        self.tabs = [
            AdminDashboardTab(),
            PredictionTab(),
            SegmentAnalysisTab()
            # AdvancedAnalyticsTab()
        ]

    def run(self):
        st.set_page_config(
            page_title="CAIRE - Admin Dashboard",
            page_icon="🎯",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        load_custom_css()

        st.markdown("""
        <div style='text-align: center; padding: 20px 0 10px 0;'>
            <h1 style='
                background: linear-gradient(135deg, #60a5fa, #3b82f6);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                font-size: 2.5em;
                margin: 0;
            '>🎯 CAIRE - Admin Dashboard</h1>
            <p style='color: #94a3b8; font-size: 1.1em; margin: 10px 0;'>
                Cart Abandonment Intelligence & Recovery Engine
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Create tabs for different admin functionalities
        main_tabs = st.tabs([tab.name for tab in self.tabs])

        BaseTab.ensure_data_directory()

        for tab, ui in zip(main_tabs, self.tabs):
            with tab:
                ui.run()

        add_footer()


if __name__ == "__main__":
    dashboard = AdminDashboard()
    dashboard.run()