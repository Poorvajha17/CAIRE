import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from custom_styles import create_header, create_section, info_box, warning_box, success_box, error_box

class SegmentAnalysisTab:
    def __init__(self):
        self.name = "👥 Segments"
        self.segmenter = None
        self.df = None
        
        # Static strategies data
        self.segment_strategies = {
            "High-Value Loyalists": {
                "priority": "Low",
                "strategies": [
                    "💎 VIP early access to new products",
                    "🎫 Double loyalty points campaign", 
                    "📧 Regular updates about products matching their preferences",
                    "🎁 Surprise free shipping or small gifts on next purchase"
                ],
                "channel": "Email + Mobile App Notification",
                "loyalty_points_multiplier": 2.0,
                "complementary_products": ["Phone Case", "Screen Protector", "Wireless Earbuds"]
            },
            "At-Risk Converters": {
                "priority": "Very High",
                "strategies": [
                    "🔥 Limited-time discount (10-15%) on abandoned items",
                    "🚀 Personal executive email follow-up",
                    "📞 Personal shopping assistant offer", 
                    "⏰ Stock availability alerts for items in cart"
                ],
                "channel": "Email + SMS + Push Notification",
                "discount_range": (10, 15),
                "stock_alert_threshold": 5
            },
            "Engaged Researchers": {
                "priority": "High", 
                "strategies": [
                    "📚 Product expert consultation offer",
                    "🎥 Detailed product demonstration videos",
                    "💬 Live chat support promotion",
                    "🔍 Advanced product comparison tools"
                ],
                "channel": "Email + Retargeting Ads",
                "expert_consultation": True,
                "demo_videos": True
            },
            "Price-Sensitive Shoppers": {
                "priority": "Medium",
                "strategies": [
                    "💰 Tiered discounts based on cart value",
                    "🎟️ Additional promo codes for next purchase",
                    "📦 Free shipping threshold reduction",
                    "🔄 Price drop alerts for watched items"
                ],
                "channel": "Email + Browser Push",
                "free_shipping_threshold": 150,
                "tiered_discounts": {
                    "500": 5,
                    "1000": 10, 
                    "2000": 15
                }
            },
            "Casual Browsers": {
                "priority": "Low",
                "strategies": [
                    "🌐 Personalized product recommendations",
                    "📢 New arrival notifications", 
                    "🏆 Social proof and trending products",
                    "🔔 Re-engagement campaign after 7 days"
                ],
                "channel": "Email only",
                "reengagement_days": 7
            }
        }
        
    def load_data_and_segmenter(self):
        try:
            project_root = Path(__file__).parent  
            sys.path.insert(0, str(project_root))
            
            try:
                from src.recovery.recovery import EnhancedCustomerSegmenter
            except ImportError as import_err:
                st.warning(f"First import attempt failed: {import_err}. Trying alternative...")
                # Alternative: import directly from the file
                import importlib.util
                recovery_path = project_root / "src" / "recovery" / "recovery.py"
                if recovery_path.exists():
                    spec = importlib.util.spec_from_file_location("recovery_module", recovery_path)
                    recovery_module = importlib.util.module_from_spec(spec)
                    sys.modules["recovery_module"] = recovery_module
                    spec.loader.exec_module(recovery_module)
                    EnhancedCustomerSegmenter = recovery_module.EnhancedCustomerSegmenter
                else:
                    st.error(f"Recovery.py not found at: {recovery_path}")
                    return None, None
            
            possible_paths = [
                project_root / "data" / "cart_abandonment_featured.csv",
                project_root / "cart_abandonment_featured.csv",
                Path("data/cart_abandonment_featured.csv"),
            ]
            
            self.df = None
            for featured_path in possible_paths:
                if featured_path.exists():
                    self.df = pd.read_csv(featured_path)
                    #st.success(f"Data loaded from: {featured_path}")
                    break
            
            if self.df is None:
                st.error("Could not find featured data file in any location")
                return None, None
            
            required_features = [
                'engagement_score', 'num_items_carted', 'cart_value', 
                'session_duration', 'num_pages_viewed', 'scroll_depth',
                'return_user', 'if_payment_page_reached', 'discount_applied',
                'has_viewed_shipping_info', 'abandoned'
            ]
            
            missing_features = [f for f in required_features if f not in self.df.columns]
            if missing_features:
                st.error(f"Missing required features: {missing_features}")
                return None, None
            
            self.segmenter = EnhancedCustomerSegmenter(n_segments=5)
            self.segmenter.fit(self.df)
            self.df['segment'] = self.segmenter.predict_segment(self.df)
            self.df['segment_name'] = self.df['segment'].map(
                {k: v['segment_name'] for k, v in self.segmenter.segment_profiles.items()}
            )
            
            st.success("Segmentation completed successfully!")
            return self.df, self.segmenter
            
        except ImportError as e:
            st.error(f"Could not import EnhancedCustomerSegmenter: {e}")
            return None, None
        except Exception as e:
            st.error(f"Error in segmentation: {e}")
            import traceback
            st.code(traceback.format_exc())
            return None, None
        
    def render_segment_overview(self):
        """Render comprehensive segment overview"""
        create_section("🎯 Customer Segments Overview","")
        
        if self.segmenter is None or self.df is None:
            warning_box("Segmentation data not available. Please check data files.")
            return
        
        col1, col2, col3, col4 = st.columns(4)
        
        total_segments = len(self.segmenter.segment_profiles)
        total_customers = len(self.df)
        avg_abandonment = self.df['abandoned'].mean() * 100
        high_priority_segments = sum(1 for profile in self.segmenter.segment_profiles.values() 
                                   if profile['recovery_priority'] in ['Very High', 'High'])
        
        with col1:
            st.metric("Total Segments", total_segments)
        with col2:
            st.metric("Total Customers", f"{total_customers:,}")
        with col3:
            st.metric("Avg Abandonment", f"{avg_abandonment:.1f}%")
        with col4:
            st.metric("High Priority Segments", high_priority_segments)
        
        # Segment distribution
        st.subheader("📊 Segment Distribution")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            segment_counts = self.df['segment_name'].value_counts()
            fig = px.pie(
                values=segment_counts.values,
                names=segment_counts.index,
                title="Customer Distribution by Segment",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Segment Sizes**")
            for segment, count in segment_counts.items():
                percentage = (count / total_customers) * 100
                st.write(f"• **{segment}**: {count} ({percentage:.1f}%)")

    def render_segment_performance(self):
        """Render segment performance metrics"""
        create_section("Segment Performance Metrics", "Key Behavioral Indicators")
        
        if self.segmenter is None or self.df is None:
            return
        
        # Calculate metrics for each segment
        segments_data = []
        for segment_id, profile in self.segmenter.segment_profiles.items():
            segment_mask = self.df['segment'] == segment_id
            segment_df = self.df[segment_mask]
            
            # Calculate metrics safely with fallbacks
            abandonment_rate = (segment_df['abandoned'].mean() * 100) if len(segment_df) > 0 else 0
            avg_cart_value = segment_df['cart_value'].mean() if len(segment_df) > 0 else 0
            avg_engagement = segment_df['engagement_score'].mean() if len(segment_df) > 0 else 0
            return_user_rate = (segment_df['return_user'].mean() * 100) if len(segment_df) > 0 else 0
            
            segments_data.append({
                'Segment': profile.get('segment_name', f'Segment {segment_id}'),
                'Abandonment Rate': abandonment_rate,
                'Avg Cart Value': avg_cart_value,
                'Avg Engagement': avg_engagement,
                'Return User Rate': return_user_rate,
                'Recovery Priority': profile.get('recovery_priority', 'Medium'),
                'Business Value': profile.get('business_value', 'Medium'),
                'Size': len(segment_df)
            })
        
        segments_df = pd.DataFrame(segments_data)
        
        # Performance comparison - Abandonment rate only
        fig = px.bar(
            segments_df,
            x='Segment',
            y='Abandonment Rate',
            title="Abandonment Rate by Segment",
            color='Abandonment Rate',
            color_continuous_scale='RdYlGn_r',
            text='Abandonment Rate'
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
                
        # Detailed metrics table
        st.subheader("Detailed Segment Metrics")
        display_df = segments_df[['Segment', 'Size', 'Abandonment Rate', 'Avg Cart Value', 
                                'Avg Engagement', 'Return User Rate', 'Recovery Priority', 'Business Value']].copy()
        display_df['Abandonment Rate'] = display_df['Abandonment Rate'].round(1)
        display_df['Avg Cart Value'] = display_df['Avg Cart Value'].round(2)
        display_df['Avg Engagement'] = display_df['Avg Engagement'].round(3)
        display_df['Return User Rate'] = display_df['Return User Rate'].round(1)
        
        st.dataframe(display_df, use_container_width=True)

    def render_recovery_strategies(self):
        """Render static recovery strategies"""
        create_section("🛠️ Recovery Strategies", "Pre-defined Segment-Specific Actions")
        
        if self.segmenter is None or self.df is None:
            return
        
        st.subheader("Segment-Specific Recovery Strategies")
        
        # Display strategies for each segment
        for segment_id, profile in self.segmenter.segment_profiles.items():
            segment_name = profile.get('segment_name', f'Segment {segment_id}')
            strategy = self.segment_strategies.get(segment_name, {})
            
            # Calculate actual metrics from data
            segment_mask = self.df['segment'] == segment_id
            segment_df = self.df[segment_mask]
            segment_size = len(segment_df)
            abandonment_rate = (segment_df['abandoned'].mean() * 100) if segment_size > 0 else 0
            avg_cart_value = segment_df['cart_value'].mean() if segment_size > 0 else 0
            
            with st.expander(f"{segment_name} - {len(strategy.get('strategies', []))} Strategies", expanded=False):
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    if strategy:
                        st.write(f"**Priority:** `{strategy.get('priority', 'Medium')}`")
                        st.write(f"**Channel:** `{strategy.get('channel', 'Email')}`")
                        
                        st.write("**Strategies:**")
                        for i, action in enumerate(strategy.get('strategies', []), 1):
                            st.write(f"{i}. {action}")
                        
                        # Show additional configurations
                        if segment_name == "High-Value Loyalists":
                            st.info(f"**Loyalty Multiplier:** {strategy.get('loyalty_points_multiplier', 1.0)}x")
                            st.info(f"**Complementary Products:** {', '.join(strategy.get('complementary_products', []))}")
                        
                        elif segment_name == "At-Risk Converters":
                            discount_range = strategy.get('discount_range', (10, 15))
                            st.info(f"**Discount Range:** {discount_range[0]}-{discount_range[1]}%")
                            st.info(f"**Stock Alert Threshold:** {strategy.get('stock_alert_threshold', 5)} items")
                        
                        elif segment_name == "Price-Sensitive Shoppers":
                            threshold = strategy.get('free_shipping_threshold', 150)
                            st.info(f"**Free Shipping Threshold:** ₹{threshold}")
                            discounts = strategy.get('tiered_discounts', {})
                            discount_str = ", ".join([f"₹{k}: {v}%" for k, v in discounts.items()])
                            st.info(f"**Tiered Discounts:** {discount_str}")
                    
                with col2:
                    st.write("**Segment Stats:**")
                    st.metric("Customers", f"{segment_size:,}")
                    st.metric("Abandonment", f"{abandonment_rate:.1f}%")
                    st.metric("Avg Cart Value", f"₹{avg_cart_value:.2f}")


    def run(self):
        
        # Load data and segmenter
        with st.spinner("Loading segmentation data and models..."):
            self.df, self.segmenter = self.load_data_and_segmenter()
        
        if self.segmenter and self.df is not None:
            # Create tabs for different segmentation views
            seg_tabs = st.tabs([
                "Overview", 
                "Performance", 
                "Strategies"
            ])
            
            with seg_tabs[0]:
                self.render_segment_overview()
            
            with seg_tabs[1]:
                self.render_segment_performance()
            
            with seg_tabs[2]:
                self.render_recovery_strategies()
            
        else:
            error_box("""
            **Unable to load segmentation system.** Please ensure:
            
            1. **Data File**: `data/cart_abandonment_featured.csv` exists with required features
            2. **Recovery Module**: `recovery.py` is available in the recovery directory
            3. **Dependencies**: All required packages are installed (scikit-learn, etc.)
            
            **Required Features**: engagement_score, cart_value, session_duration, num_pages_viewed, 
            scroll_depth, return_user, if_payment_page_reached, discount_applied, has_viewed_shipping_info, abandoned
            """)
            
            # Show available files for debugging
            with st.expander("🔧 Debug Information"):
                st.write("**Current Directory Files:**")
                current_files = os.listdir('.')
                st.write([f for f in current_files if f.endswith('.py') or f == 'data'])
                
                if os.path.exists('data'):
                    st.write("**Data Directory Files:**")
                    data_files = os.listdir('data')
                    st.write(data_files)