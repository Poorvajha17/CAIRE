import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os
import math
import time
import json
from apriori import AprioriRecommender

# Page config
st.set_page_config(
    page_title="ShopEasy - Test Data Store",
    page_icon="🛒",
    layout="wide"
)

# ============================================================================
# PREPROCESSOR CLASS
# ============================================================================

class CartAbandonmentPreprocessor:
    def __init__(self, encoders_path, scaler_path):
        self.encoders_path = encoders_path
        self.scaler_path = scaler_path
        self.label_encoders = {}
        self.scaler_info = {}

    def load_preprocessing_artifacts(self):
        """Load the preprocessing artifacts (encoders and scalers)"""
        try:
            with open(self.encoders_path, 'r') as f:
                self.label_encoders = json.load(f)
            with open(self.scaler_path, 'r') as f:
                self.scaler_info = json.load(f)
            return True
        except FileNotFoundError:
            st.error(f"❌ Preprocessing files not found at {self.encoders_path} and {self.scaler_path}")
            st.info("💡 Please run the preprocessing script first to generate label_encoders.json and scaler_info.json")
            return False

    def preprocess_features(self, features_dict):
        """Apply the exact same preprocessing steps as in the training data"""
        if not self.label_encoders or not self.scaler_info:
            if not self.load_preprocessing_artifacts():
                return features_dict

        # Create a copy to avoid modifying original
        processed_features = features_dict.copy()
        
        # Apply the exact same encoding as in preprocessing
        categorical_cols = ["day_of_week", "time_of_day", "device_type", "browser", 
                           "referral_source", "location", "most_viewed_category"]
        
        for col in categorical_cols:
            if col in processed_features and col in self.label_encoders:
                original_val = processed_features[col]
                mapping = self.label_encoders[col]
                
                # Handle both string and numeric inputs
                if isinstance(original_val, str):
                    # String input - map using the encoder
                    processed_features[col] = mapping.get(original_val, 0)
                else:
                    # Numeric input - ensure it's within valid range
                    valid_values = list(mapping.values())
                    if original_val not in valid_values:
                        processed_features[col] = 0  # Default to first category
        
        # Apply the exact same scaling as in preprocessing
        to_standardize = ["session_duration", "num_pages_viewed", "scroll_depth"]
        
        for col in to_standardize:
            if col in processed_features and col in self.scaler_info:
                scaler = self.scaler_info[col]
                mean_val = scaler["mean"]
                std_val = scaler["std"]
                processed_features[col] = (processed_features[col] - mean_val) / std_val if std_val != 0 else 0

        # Apply the exact same cart_value transformation (log + standardize)
        if "cart_value" in processed_features and "cart_value" in self.scaler_info:
            cart_val = processed_features["cart_value"]
            # Apply log transform (same as np.log1p in preprocessing)
            cart_val_log = np.log1p(cart_val)
            # Apply standardization with saved parameters
            scaler = self.scaler_info["cart_value"]
            mean_val = scaler["mean"]
            std_val = scaler["std"]
            processed_features["cart_value"] = (cart_val_log - mean_val) / std_val if std_val != 0 else 0

        # Apply the exact same shipping_fee transformation (log transform only)
        if "shipping_fee" in processed_features:
            processed_features["shipping_fee"] = np.log1p(processed_features["shipping_fee"])

        return processed_features

# ============================================================================
# LOAD LABEL ENCODERS FROM JSON FILE
# ============================================================================

# Initialize preprocessor
preprocessor = CartAbandonmentPreprocessor(
    encoders_path='data/label_encoders.json',
    scaler_path='data/scaler_info.json'
)

# Load preprocessing artifacts
preprocessing_loaded = preprocessor.load_preprocessing_artifacts()

if preprocessing_loaded:
    # Use the loaded encoders for the app
    DAY_OF_WEEK_MAP = preprocessor.label_encoders.get('day_of_week', {})
    TIME_OF_DAY_MAP = preprocessor.label_encoders.get('time_of_day', {})
    DEVICE_TYPE_MAP = preprocessor.label_encoders.get('device_type', {})
    BROWSER_MAP = preprocessor.label_encoders.get('browser', {})
    REFERRAL_SOURCE_MAP = preprocessor.label_encoders.get('referral_source', {})
    LOCATION_MAP = preprocessor.label_encoders.get('location', {})
    CATEGORY_MAP = preprocessor.label_encoders.get('most_viewed_category', {})
else:
    # Fallback mappings
    DAY_OF_WEEK_MAP = {"Saturday": 0, "Thursday": 1, "Monday": 2, "Sunday": 3, "Wednesday": 4, "Friday": 5, "Tuesday": 6}
    TIME_OF_DAY_MAP = {"Evening": 0, "Afternoon": 1, "Morning": 2, "Night": 3}
    DEVICE_TYPE_MAP = {"Mobile": 0, "Desktop": 1, "Tablet": 2}
    BROWSER_MAP = {"Opera": 0, "Safari": 1, "Chrome": 2, "Edge": 3, "Firefox": 4}
    REFERRAL_SOURCE_MAP = {"Search Engine": 0, "Ads": 1, "Direct": 2, "Social Media": 3, "Email Campaign": 4}
    LOCATION_MAP = {"Nagpur, Maharashtra": 0, "Mumbai, Maharashtra": 1, "Delhi, Delhi": 2}
    CATEGORY_MAP = {"Electronics": 0, "Clothing": 1, "Home & Kitchen": 2}

FEATURE_ORDER = [
    'session_id', 'user_id','return_user', 'day_of_week', 'time_of_day', 'session_duration', 'num_pages_viewed',
    'num_items_carted', 'has_viewed_shipping_info', 'scroll_depth', 'cart_value',
    'discount_applied', 'shipping_fee', 'free_shipping_eligible', 'device_type', 'browser',
    'referral_source', 'location', 'if_payment_page_reached', 'most_viewed_category',
    'engagement_intensity', 'scroll_engagement', 'is_weekend', 'has_multiple_items',
    'has_high_engagement', 'research_behavior', 'quick_browse', 'engagement_score',
    'peak_hours', 'returning_peak', 'day_sin', 'day_cos', 'time_sin', 'time_cos', 'pca1', 'pca2','customer_segment'
]

PRODUCTS = [
    {"id": 1, "name": "Smartphone", "price": 299.99, "category": "Electronics", "description": "Latest 5G smartphone with amazing camera", "image": "https://images.unsplash.com/photo-1511707171634-5f897ff02aa9?w=400&h=300&fit=crop"},
    {"id": 2, "name": "Car Accessories", "price": 899.99, "category": "Automotive", "description": "Premium car mats and organizers", "image": "https://images.unsplash.com/photo-1549317661-bd32c8ce0db2?w=400&h=300&fit=crop"},
    {"id": 3, "name": "Cookware Set", "price": 1499.99, "category": "Home & Kitchen", "description": "12-piece non-stick cookware set", "image": "https://images.unsplash.com/photo-1556909114-f6e7ad7d3136?w=400&h=300&fit=crop"},
    {"id": 4, "name": "Skincare Kit", "price": 499.99, "category": "Beauty", "description": "Complete skincare routine set", "image": "https://images.unsplash.com/photo-1556228578-8c89e6adf883?w=400&h=300&fit=crop"},
    {"id": 5, "name": "Novel Collection", "price": 1029.99, "category": "Books", "description": "Pack of bestselling fiction novels", "image": "https://images.unsplash.com/photo-1544716278-ca5e3f4abd8c?w=400&h=300&fit=crop"},
    {"id": 6, "name": "T-Shirt", "price": 1009.99, "category": "Clothing", "description": "100% cotton comfortable t-shirt", "image": "https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?w=400&h=300&fit=crop"},
    {"id": 7, "name": "Action Figure", "price": 2400.99, "category": "Toys", "description": "Collectible action figure with accessories", "image": "https://images.unsplash.com/photo-1587654780291-39c9404d746b?w=400&h=300&fit=crop"},
    {"id": 8, "name": "Running Shoes", "price": 7900.99, "category": "Sports", "description": "Professional running shoes with cushioning", "image": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=400&h=300&fit=crop"},
    {"id": 9, "name": "Organic Snacks", "price": 1400.99, "category": "Groceries", "description": "Healthy organic snack pack", "image": "https://images.unsplash.com/photo-1542838132-92c53300491e?w=400&h=300&fit=crop"},
    {"id": 10, "name": "Laptop", "price": 89900.99, "category": "Electronics", "description": "High-performance laptop for professionals", "image": "https://images.unsplash.com/photo-1496181133206-80ce9b88a853?w=400&h=300&fit=crop"},
     {"id": 11, "name": "Wireless Earbuds", "price": 1999.99, "category": "Electronics", "description": "Noise cancellation wireless earbuds", "image": "https://images.unsplash.com/photo-1590658165737-15a047b8b5e2?w=400&h=300&fit=crop"},
    {"id": 12, "name": "Phone Case", "price": 499.99, "category": "Electronics", "description": "Protective phone case", "image": "https://images.unsplash.com/photo-1556656793-08538906a9f8?w=400&h=300&fit=crop"},
    {"id": 13, "name": "Jeans", "price": 1599.99, "category": "Clothing", "description": "Comfortable denim jeans", "image": "https://images.unsplash.com/photo-1542272604-787c3835535d?w=400&h=300&fit=crop"},
    {"id": 14, "name": "Coffee Maker", "price": 3499.99, "category": "Home & Kitchen", "description": "Automatic coffee brewing machine", "image": "https://images.unsplash.com/photo-1495474472287-4d71bcdd2085?w=400&h=300&fit=crop"},
    {"id": 15, "name": "Backpack", "price": 1299.99, "category": "Accessories", "description": "Waterproof laptop backpack", "image": "https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=400&h=300&fit=crop"},
    {"id": 16, "name": "Fitness Tracker", "price": 2999.99, "category": "Electronics", "description": "Smart health monitoring band", "image": "https://images.unsplash.com/photo-1576243345690-4e4b79b63288?w=400&h=300&fit=crop"},
    {"id": 17, "name": "Desk Lamp", "price": 899.99, "category": "Home & Kitchen", "description": "LED adjustable desk lamp", "image": "https://images.unsplash.com/photo-1507473885765-e6ed057f782c?w=400&h=300&fit=crop"},
    {"id": 18, "name": "Water Bottle", "price": 599.99, "category": "Sports", "description": "Insulated stainless steel bottle", "image": "https://images.unsplash.com/photo-1523362628745-0c100150b504?w=400&h=300&fit=crop"},
    {"id": 19, "name": "Sunglasses", "price": 1299.99, "category": "Accessories", "description": "UV protection sunglasses", "image": "https://images.unsplash.com/photo-1511499767150-a48a237f0083?w=400&h=300&fit=crop"},
    {"id": 20, "name": "Bluetooth Speaker", "price": 2499.99, "category": "Electronics", "description": "Portable wireless speaker", "image": "https://images.unsplash.com/photo-1608043152269-423dbba4e7e1?w=400&h=300&fit=crop"},
    {"id": 21, "name": "Screen Protector", "price": 299.99, "category": "Electronics", "description": "Tempered glass screen protection", "image": "https://images.unsplash.com/photo-1583394838336-acd977736f90?w=400&h=300&fit=crop"},
    {"id": 22, "name": "Wireless Mouse", "price": 799.99, "category": "Electronics", "description": "Ergonomic wireless computer mouse", "image": "https://images.unsplash.com/photo-1527864550417-7fd91fc51a46?w=400&h=300&fit=crop"},
    {"id": 23, "name": "Sports Socks", "price": 399.99, "category": "Sports", "description": "Moisture-wicking athletic socks", "image": "https://images.unsplash.com/photo-1544966503-7cc5ac882d5d?w=400&h=300&fit=crop"},
    {"id": 24, "name": "Kitchen Utensils", "price": 1299.99, "category": "Home & Kitchen", "description": "12-piece stainless steel utensil set", "image": "https://images.unsplash.com/photo-1556909114-6c679fa076b4?w=400&h=300&fit=crop"},
    {"id": 25, "name": "Face Mask", "price": 199.99, "category": "Beauty", "description": "Hydrating facial sheet masks", "image": "https://images.unsplash.com/photo-1556228578-7c79d0e69c15?w=400&h=300&fit=crop"},
    {"id": 26, "name": "Coffee Beans", "price": 599.99, "category": "Groceries", "description": "Premium arabica coffee beans", "image": "https://images.unsplash.com/photo-1587734195503-904fca47e0e9?w=400&h=300&fit=crop"},
    {"id": 27, "name": "Power Bank", "price": 1499.99, "category": "Electronics", "description": "10000mAh portable charger", "image": "https://images.unsplash.com/photo-1574944985071-1a1b42d160cf?w=400&h=300&fit=crop"},
    {"id": 28, "name": "Sports Bag", "price": 1899.99, "category": "Sports", "description": "Multi-compartment gym duffle bag", "image": "https://images.unsplash.com/photo-1553062407-7db6c0a6bed6?w=400&h=300&fit=crop"},
    {"id": 29, "name": "Watch", "price": 4599.99, "category": "Accessories", "description": "Classic analog wristwatch", "image": "https://images.unsplash.com/photo-1523170335258-f5ed11844a49?w=400&h=300&fit=crop"},
    {"id": 30, "name": "Laptop Sleeve", "price": 899.99, "category": "Accessories", "description": "Padded laptop protective case", "image": "https://images.unsplash.com/photo-1548611635-b6e3c9c5d92a?w=400&h=300&fit=crop"}
]

# Valid promo codes with discount percentages
VALID_PROMO_CODES = {
    "SAVE10": 10,
    "SAVE20": 20,
    "WELCOME": 15,
    "SUMMER50": 50
}

# ============================================================================
# DIRECT RULE-BASED SEGMENTATION
# ============================================================================

class DirectRuleBasedSegmenter:  
    def segment_single_customer(self, features):
        try:
            cart_value = features.get('cart_value', 0)
            engagement_score = features.get('engagement_score', 0)
            return_user = features.get('return_user', 0)
            abandoned = features.get('abandoned', 0)
            payment_reached = features.get('if_payment_page_reached', 0)
            discount_applied = features.get('discount_applied', 0)
            session_duration = features.get('session_duration', 0)
            num_items_carted = features.get('num_items_carted', 0)
            
            print(f"🔍 Segmentation Input - Cart: ₹{cart_value}, Engagement: {engagement_score}, Return: {return_user}, Abandoned: {abandoned}, Payment: {payment_reached}")
            
            # Rule 1: High-Value Loyalists
            if (return_user == 1 and 
                cart_value > 20000 and 
                engagement_score >= 6 and 
                abandoned == 0):
                return "High-Value Loyalists"
            elif (abandoned == 1 and 
                  cart_value > 20000 and 
                  payment_reached == 0 and
                  engagement_score >= 5):
                return "At-Risk Converters"
            elif (engagement_score >= 7 and 
                  session_duration >= 60 and 
                  num_items_carted > 0 and
                  abandoned == 1):
                return "Engaged Researchers"
            elif (discount_applied == 1 or 
                  cart_value < 400 or
                  (abandoned == 1 and cart_value < 450)):
                return "Price-Sensitive Shoppers"
            else:
                return "Casual Browsers"
                
        except Exception as e:
            print(f"❌ Direct segmentation error: {e}")
            return "Casual Browsers"

# RECOVERY STRATEGY MANAGER

class RecoveryStrategyManager:
    
    def __init__(self):
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
        self.applied_strategies = {}
    
    def get_recovery_strategy(self, segment_name, user_data=None):
        """Get recovery strategy for specific segment"""
        segment_info = self.segment_strategies.get(segment_name, {})
        
        if not segment_info:
            return {
                "segment": "Unknown",
                "priority": "Medium",
                "message": "Continue with standard engagement strategy",
                "strategies": ["Standard follow-up email after 24 hours"]
            }
        
        return {
            "segment": segment_name,
            "priority": segment_info["priority"],
            "message": f"Targeted recovery for {segment_name}",
            "strategies": segment_info["strategies"],
            "channel": segment_info["channel"],
            "user_context": user_data
        }
    
    def apply_high_value_loyalist_actions(self, session_data):
        actions = []
        base_points = int(session_data['cart_value'] / 100)  # 1 point per ₹100
        bonus_points = base_points * 2
        actions.append(f"🎫 Loyalty points doubled! Earned {bonus_points} points (normally {base_points})")
        complementary_product = np.random.choice(self.segment_strategies["High-Value Loyalists"]["complementary_products"])
        actions.append(f"🎁 Free complementary {complementary_product} added to your order!")
        actions.append("🚚 Free express shipping applied to your order")
        return actions
    
    def apply_at_risk_converter_actions(self, session_data):
        actions = []
        discount_pct = np.random.randint(10, 16)  # 10-15% discount
        original_total = session_data['cart_value']
        discount_amount = original_total * (discount_pct / 100)
        new_total = original_total - discount_amount
        
        actions.append(f"🔥 Limited-time {discount_pct}% discount applied! Saved ₹{discount_amount:.2f}")
        actions.append(f"💰 New total: ₹{new_total:.2f} (was ₹{original_total:.2f})")
        
        low_stock_items = []
        for item in session_data['items'][:2]: 
            if np.random.random() < 0.3: 
                low_stock_items.append(item['name'])
        
        if low_stock_items:
            items_list = ", ".join(low_stock_items)
            actions.append(f"⏰ Low stock alert: {items_list} - Only {np.random.randint(1, 6)} left!")
        actions.append("👨‍💼 Personal shopping assistant assigned -他们会联系您 within 1 hour")
        
        return actions
    
    def apply_engaged_researcher_actions(self, session_data):
        actions = []
        actions.append("📚 Product expert consultation scheduled -他们会联系您 tomorrow")
        
        if session_data['viewed_products']:
            demo_product = session_data['viewed_products'][0]['name']
            actions.append(f"🎥 Detailed demo video available for {demo_product} - Check your email")
        actions.append("💬 Priority live chat support activated - Get instant help")
        
        return actions
    
    def apply_price_sensitive_shopper_actions(self, session_data):
        """Apply practical actions for Price-Sensitive Shoppers"""
        actions = []
        
        cart_value = session_data['cart_value']
        
        # 1. Tiered discounts based on cart value
        tiered_discounts = self.segment_strategies["Price-Sensitive Shoppers"]["tiered_discounts"]
        applied_discount = 0
        
        for threshold, discount in sorted(tiered_discounts.items()):
            if cart_value >= float(threshold):
                applied_discount = discount
        
        if applied_discount > 0:
            discount_amount = cart_value * (applied_discount / 100)
            actions.append(f"💰 Tiered discount: {applied_discount}% off for cart above ₹{list(tiered_discounts.keys())[list(tiered_discounts.values()).index(applied_discount)]}")
            actions.append(f"💸 Save ₹{discount_amount:.2f} on this order!")        
        new_threshold = self.segment_strategies["Price-Sensitive Shoppers"]["free_shipping_threshold"]
        if cart_value < new_threshold:
            remaining = new_threshold - cart_value
            actions.append(f"📦 Free shipping at ₹{new_threshold} (normally ₹200) - Add ₹{remaining:.2f} more!")
        else:
            actions.append("📦 Free shipping applied! (Special lowered threshold)")
        actions.append(f"🎟️ Extra promo for next order: SAVE20, WELCOME - Save 15-20%")
        
        return actions
    
    def apply_casual_browser_actions(self, session_data):
        """Apply practical actions for Casual Browsers"""
        actions = []
        
        # 1. Personalized recommendations based on viewed items
        if session_data['viewed_products']:
            viewed_categories = list(set([p['category'] for p in session_data['viewed_products']]))
            if viewed_categories:
                actions.append(f"🌐 Personalized recommendations for {viewed_categories[0]} category coming to your email")
        
        # 2. Social proof notifications
        trending_products = np.random.choice([p['name'] for p in PRODUCTS], size=2, replace=False)
        actions.append(f"🏆 Trending now: {', '.join(trending_products)}")
        
        # 3. Re-engagement reminder
        reengage_days = self.segment_strategies["Casual Browsers"]["reengagement_days"]
        actions.append(f"🔔 We'll remind you in {reengage_days} days about products you liked")
        
        return actions
    
    def execute_segment_actions(self, segment_name, session_data):
        """Execute practical actions for the given segment"""
        session_id = session_data['session_id']
        
        if session_id not in self.applied_strategies:
            self.applied_strategies[session_id] = {}
        
        actions = []
        
        if segment_name == "High-Value Loyalists":
            actions = self.apply_high_value_loyalist_actions(session_data)
        elif segment_name == "At-Risk Converters":
            actions = self.apply_at_risk_converter_actions(session_data)
        elif segment_name == "Engaged Researchers":
            actions = self.apply_engaged_researcher_actions(session_data)
        elif segment_name == "Price-Sensitive Shoppers":
            actions = self.apply_price_sensitive_shopper_actions(session_data)
        elif segment_name == "Casual Browsers":
            actions = self.apply_casual_browser_actions(session_data)
        else:
            actions = ["📧 Standard follow-up email will be sent in 24 hours"]
        
        # Store applied actions
        self.applied_strategies[session_id] = {
            'segment': segment_name,
            'actions': actions,
            'timestamp': datetime.now().isoformat()
        }
        
        return actions

# INITIALIZATION
if 'session_data' not in st.session_state:
    st.session_state.session_data = {
        'session_id': f"S{int(time.time())}",
        'user_id': f"U{np.random.randint(1000, 9999)}",
        'items': [],
        'start_time': datetime.now(),
        'events': [],
        'cart_value': 0.0,
        'pages_viewed': [],  # Changed to list to track order and count
        'current_page': 'home',  # Track current page
        'categories_viewed': set(),
        'scroll_depth': 0.0,  # Real scroll depth percentage
        'view_product_count': 0,
        'shipping_viewed': False,
        'payment_reached': False,
        'return_user': np.random.choice([0, 1], p=[0.7, 0.3]),
        'device_type': np.random.choice(list(DEVICE_TYPE_MAP.keys())),
        'browser': np.random.choice(list(BROWSER_MAP.keys())),
        'referral_source': np.random.choice(list(REFERRAL_SOURCE_MAP.keys())),
        'location': np.random.choice(list(LOCATION_MAP.keys())),
        'discount_applied': 0,
        'discount_percentage': 0,
        'discount_code': None,
        'viewed_products': [],
        'time_on_page': {}  # Track time spent on each page
    }

if 'all_sessions' not in st.session_state:
    st.session_state.all_sessions = []

if 'viewing_product' not in st.session_state:
    st.session_state.viewing_product = None

if 'promo_message' not in st.session_state:
    st.session_state.promo_message = None

# ============================================================================
# INITIALIZE REAL APRIORI RECOMMENDER ✅
# ============================================================================

if 'recommender' not in st.session_state:
    print("🔄 Initializing Real Apriori Recommender...")
    
    # Create instance with tuned parameters
    st.session_state.recommender = AprioriRecommender(
        min_support=0.02,      # 2% minimum support
        min_confidence=0.3,    # 30% confidence
        min_lift=1.0          # Positive correlation only
    )
    
    # CRITICAL: Set product catalog
    st.session_state.recommender.set_product_catalog(PRODUCTS)
    
    # Try loading pre-trained model, otherwise train fresh
    model_loaded = st.session_state.recommender.load_model('data/apriori_model.json')
    
    if not model_loaded:
        print("📚 No saved model found. Training from scratch...")
        st.session_state.recommender.train()
        # Save for next time
        st.session_state.recommender.save_model('data/apriori_model.json')
    else:
        print("✅ Loaded pre-trained model!")
    
    # Optional: Print rules for debugging (comment out in production)
    if st.session_state.recommender.rules:
        st.session_state.recommender.print_rules(top_n=5)

# Initialize other components
st.session_state.segmenter = DirectRuleBasedSegmenter()
recovery_manager = RecoveryStrategyManager()


# ============================================================================
# HELPER FUNCTIONS (UPDATED WITH PREPROCESSING)
# ============================================================================

def get_time_of_day(hour):
    """Convert hour to time category"""
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 22:
        return "Evening"
    else:
        return "Night"

def log_event(event_type, product_id=None, product_name=None, category=None, page_type=None):
    """Enhanced user interaction logging"""
    event = {
        'timestamp': datetime.now().isoformat(),
        'event_type': event_type,
        'product_id': product_id,
        'product_name': product_name,
        'category': category,
        'page_type': page_type,
        'session_id': st.session_state.session_data['session_id'],
        'user_id': st.session_state.session_data['user_id']
    }
    st.session_state.session_data['events'].append(event)

    if page_type:
        st.session_state.session_data['pages_viewed'].append(page_type)  
    
    if category:
        st.session_state.session_data['categories_viewed'].add(category)
    if event_type in ['add_to_cart', 'purchase'] and product_id:
        log_transaction_for_apriori(product_id, product_name, category, event_type)

def log_transaction_for_apriori(product_id, product_name, category, event_type):
    """Log transactions for Apriori association rule mining"""
    transaction = {
        'timestamp': datetime.now().isoformat(),
        'session_id': st.session_state.session_data['session_id'],
        'user_id': st.session_state.session_data['user_id'],
        'product_id': product_id,
        'product_name': product_name,
        'category': category,
        'event_type': event_type
    }
    
    # Save to transactions file
    os.makedirs('data', exist_ok=True)
    transactions_file = 'data/transactions.csv'
    
    df = pd.DataFrame([transaction])
    if os.path.exists(transactions_file):
        df.to_csv(transactions_file, mode='a', header=False, index=False)
    else:
        df.to_csv(transactions_file, index=False)

def navigate_page(page_name):
    """Track page navigation"""
    st.session_state.session_data['current_page'] = page_name
    log_event('page_view', None, None, None, page_name)

def add_to_cart(product):
    """Add product to cart"""
    st.session_state.session_data['items'].append(product)
    st.session_state.session_data['cart_value'] += product['price']
    log_event('add_to_cart', product['id'], product['name'], product['category'], 'product')

def remove_from_cart(product_id):
    """Remove product from cart by ID"""
    for item in st.session_state.session_data['items']:
        if item['id'] == product_id:
            st.session_state.session_data['cart_value'] -= item['price']
            st.session_state.session_data['items'].remove(item)
            log_event('remove_from_cart', product_id, item['name'], item['category'], 'cart')
            break

def view_product(product):
    """View product details"""
    if product not in st.session_state.session_data['viewed_products']:
        st.session_state.session_data['viewed_products'].append(product)
    st.session_state.session_data['view_product_count'] += 1
    log_event('view_product', product['id'], product['name'], product['category'], 'product_detail')

def apply_promo_code(code):
    """Validate and apply promo code"""
    code = code.upper().strip()
    if code in VALID_PROMO_CODES:
        st.session_state.session_data['discount_applied'] = 1
        st.session_state.session_data['discount_percentage'] = VALID_PROMO_CODES[code]
        st.session_state.session_data['discount_code'] = code
        log_event('promo_applied', None, None, None, 'checkout')
        return True, VALID_PROMO_CODES[code]
    return False, 0

def update_scroll_depth(page_scroll_percent):
    """Update realistic scroll depth based on actual page scrolling"""
    st.session_state.session_data['scroll_depth'] = max(
        st.session_state.session_data['scroll_depth'], 
        page_scroll_percent
    )
    if page_scroll_percent > 0:
        log_event('scroll', None, None, None, st.session_state.session_data['current_page'])

def calculate_realistic_engagement_score(session_data):
    """Calculate realistic engagement score (0-10 scale)"""
    
    # Extract metrics
    total_actions = len(session_data['events'])
    session_duration = (datetime.now() - session_data['start_time']).total_seconds()
    pages_viewed = len(session_data['pages_viewed'])
    products_viewed = session_data['view_product_count']
    items_carted = len(session_data['items'])
    scroll_depth = session_data['scroll_depth']
    
    # Weighted scoring system
    scores = {
        'action_density': min(2.0, total_actions / max(session_duration / 60, 1)),  # Actions per minute (0-2)
        'page_exploration': min(2.0, pages_viewed * 0.4),  # 0.4 points per page (0-2)
        'product_interest': min(2.0, products_viewed * 0.3),  # 0.3 points per product view (0-2)
        'purchase_intent': min(2.0, items_carted * 0.5),  # 0.5 points per cart item (0-2)
        'content_engagement': min(2.0, scroll_depth / 50)  # 0.02 points per % scroll (0-2)
    }
    
    # Calculate total score (0-10)
    total_score = sum(scores.values())
    
    # Apply bonus for specific behaviors
    bonuses = 0
    if session_data.get('shipping_viewed', False):
        bonuses += 0.5
    if session_data.get('payment_reached', False):
        bonuses += 1.0
    if session_data['return_user'] == 1:
        bonuses += 0.5
    
    final_score = min(10.0, total_score + bonuses)
    
    return final_score

def calculate_raw_features(abandoned):
    """Calculate raw features before preprocessing - INCLUDES ALL FEATURE ENGINEERING"""
    current_time = datetime.now()
    session_duration = (current_time - st.session_state.session_data['start_time']).total_seconds()
    
    # Generate session and user IDs if not exists
    if 'session_id' not in st.session_state.session_data:
        st.session_state.session_data['session_id'] = f"S{len(st.session_state.all_sessions) + 1000}"
    
    if 'user_id' not in st.session_state.session_data:
        st.session_state.session_data['user_id'] = f"U{np.random.randint(1, 500)}"
    
    # Get time info
    day_name = current_time.strftime("%A")
    hour = current_time.hour
    time_of_day = get_time_of_day(hour)
    
    # Count interactions
    add_events = len([e for e in st.session_state.session_data['events'] if e['event_type'] == 'add_to_cart'])
    view_events = len([e for e in st.session_state.session_data['events'] if e['event_type'] == 'view_product'])
    
    # Basic metrics - RAW VALUES (before preprocessing)
    num_pages_viewed = len(st.session_state.session_data['pages_viewed'])
    num_items_carted = len(st.session_state.session_data['items'])
    scroll_depth = st.session_state.session_data['scroll_depth']
    cart_value_raw = st.session_state.session_data['cart_value']  # Raw cart value
    shipping_fee = 0 if cart_value_raw >= 200 else 99
    free_shipping_eligible = 1 if cart_value_raw >= 200 else 0

    # Most viewed category
    if st.session_state.session_data['categories_viewed']:
        category_counts = {}
        for event in st.session_state.session_data['events']:
            if event['category']:
                category_counts[event['category']] = category_counts.get(event['category'], 0) + 1
        most_viewed_category = max(category_counts, key=category_counts.get) if category_counts else "Electronics"
    else:
        most_viewed_category = "Electronics"
    
    # ENGINEERED FEATURES (all your existing feature engineering)
    total_actions = len(st.session_state.session_data['events'])
    engagement_intensity = total_actions / max(session_duration / 60, 1)
    scroll_engagement = min(1.0, scroll_depth / 100.0)
    
    is_weekend = 1 if current_time.weekday() >= 5 else 0
    has_multiple_items = 1 if len(st.session_state.session_data['items']) > 1 else 0
    has_high_engagement = 1 if engagement_intensity > 1.5 else 0
    research_behavior = 1 if (view_events > 3 and num_items_carted > 0) else 0
    quick_browse = 1 if (session_duration < 120 and view_events <= 2) else 0
    
    # Use realistic engagement score
    engagement_score = calculate_realistic_engagement_score(st.session_state.session_data)
    
    peak_hours = 1 if 9 <= hour <= 18 else 0
    returning_peak = 1 if (st.session_state.session_data['return_user'] and peak_hours) else 0
    
    # Cyclical encoding
    day_sin = math.sin(2 * math.pi * current_time.weekday() / 7)
    day_cos = math.cos(2 * math.pi * current_time.weekday() / 7)
    time_sin = math.sin(2 * math.pi * hour / 24)
    time_cos = math.cos(2 * math.pi * hour / 24)
    
    # PCA components
    pca1 = np.random.normal(0, 1)
    pca2 = np.random.normal(0, 1)
    
    raw_features = {
        'session_id': st.session_state.session_data['session_id'],
        'user_id': st.session_state.session_data['user_id'],
        'return_user': st.session_state.session_data['return_user'],
        'day_of_week': day_name,  # Raw string
        'time_of_day': time_of_day,  # Raw string
        'session_duration': max(60, min(5000, session_duration)),  # Raw value
        'num_pages_viewed': num_pages_viewed,  # Raw value
        'num_items_carted': num_items_carted,
        'has_viewed_shipping_info': int(st.session_state.session_data['shipping_viewed']),
        'scroll_depth': scroll_depth,  # Raw value
        'cart_value': cart_value_raw,  # Raw cart value
        'discount_applied': st.session_state.session_data['discount_applied'],
        'shipping_fee': shipping_fee,  # Raw shipping fee
        'free_shipping_eligible': free_shipping_eligible,
        'device_type': st.session_state.session_data['device_type'],  # Raw string
        'browser': st.session_state.session_data['browser'],  # Raw string
        'referral_source': st.session_state.session_data['referral_source'],  # Raw string
        'location': st.session_state.session_data['location'],  # Raw string
        'if_payment_page_reached': 1 if st.session_state.session_data['payment_reached'] else 0,
        'most_viewed_category': most_viewed_category,  # Raw string
        'engagement_intensity': engagement_intensity,
        'scroll_engagement': scroll_engagement,
        'is_weekend': is_weekend,
        'has_multiple_items': has_multiple_items,
        'has_high_engagement': has_high_engagement,
        'research_behavior': research_behavior,
        'quick_browse': quick_browse,
        'engagement_score': engagement_score,
        'peak_hours': peak_hours,
        'returning_peak': returning_peak,
        'day_sin': day_sin,
        'day_cos': day_cos,
        'time_sin': time_sin,
        'time_cos': time_cos,
        'pca1': pca1,
        'pca2': pca2,
        'abandoned': abandoned
    }
    
    return raw_features

def calculate_features(abandoned):
    """Calculate features with preprocessing applied"""
    # Get raw features first (includes ALL feature engineering)
    raw_features = calculate_raw_features(abandoned)
    
    # Apply preprocessing transformations
    processed_features = preprocessor.preprocess_features(raw_features)
    
    return processed_features

def save_session_data(abandoned, segment=None):
    """Save preprocessed features for prediction"""
    # Get preprocessed features (feature engineering + preprocessing)
    features = calculate_features(abandoned)
    
    # Create ordered features dictionary
    ordered_features = {}
    ordered_features['session_id'] = features['session_id']
    ordered_features['user_id'] = features['user_id']

    for feature in FEATURE_ORDER:
        if feature in features and feature not in ['session_id', 'user_id']:
            ordered_features[feature] = features[feature]   

    ordered_features['abandoned'] = abandoned
    ordered_features['customer_segment'] = segment if segment else "Unknown"
    
    # Save to file
    os.makedirs('test_data', exist_ok=True)  
    df = pd.DataFrame([ordered_features])
    csv_file = 'test_data/test_data_for_prediction.csv'   
    
    if os.path.exists(csv_file):
        existing_df = pd.read_csv(csv_file)
        df = pd.concat([existing_df, df], ignore_index=True)   
    
    df.to_csv(csv_file, index=False)
    st.session_state.all_sessions.append(ordered_features)  
    return csv_file

def save_raw_session_data(abandoned_status=None):
    """Save raw session data (without preprocessing) for analytics"""
    raw_features = calculate_raw_features(abandoned=0)  # Get raw features
    
    # Convert to display format
    raw_data = {
        'session_id': raw_features['session_id'],
        'user_id': raw_features['user_id'],
        'timestamp': datetime.now().isoformat(),
        'return_user': 'Yes' if raw_features['return_user'] else 'No',
        'day_of_week': raw_features['day_of_week'],  # Raw string
        'time_of_day': raw_features['time_of_day'],  # Raw string
        'session_duration_seconds': raw_features['session_duration'],
        'num_pages_viewed': raw_features['num_pages_viewed'],
        'num_items_carted': raw_features['num_items_carted'],
        'has_viewed_shipping_info': 'Yes' if raw_features['has_viewed_shipping_info'] else 'No',
        'scroll_depth_percent': raw_features['scroll_depth'],
        'cart_value': raw_features['cart_value'],  # Raw cart value
        'discount_applied': 'Yes' if raw_features['discount_applied'] else 'No',
        'discount_code': st.session_state.session_data.get('discount_code', 'None'),
        'discount_percentage': st.session_state.session_data.get('discount_percentage', 0),
        'shipping_fee': raw_features['shipping_fee'],
        'free_shipping_eligible': 'Yes' if raw_features['free_shipping_eligible'] else 'No',
        'device_type': raw_features['device_type'],  # Raw string
        'browser': raw_features['browser'],  # Raw string
        'referral_source': raw_features['referral_source'],  # Raw string
        'location': raw_features['location'],  # Raw string
        'payment_page_reached': 'Yes' if raw_features['if_payment_page_reached'] else 'No',
        'most_viewed_category': raw_features['most_viewed_category'],  # Raw string
        'total_products_viewed': st.session_state.session_data['view_product_count'],
        'total_events': len(st.session_state.session_data['events']),
        'cart_items_count': len(st.session_state.session_data['items']),
        'cart_items_names': ', '.join([item['name'] for item in st.session_state.session_data['items']]),
        'viewed_categories': ', '.join(list(st.session_state.session_data['categories_viewed'])),
        'current_page': st.session_state.session_data['current_page'],
        'engagement_score': raw_features['engagement_score'],
        'abandoned': 'Yes' if abandoned_status == 1 else 'No'
    }
    
    # Define the order of columns for the raw data CSV
    raw_columns = [
        'session_id', 'user_id', 'timestamp', 'return_user', 'day_of_week', 'time_of_day',
        'session_duration_seconds', 'num_pages_viewed', 'num_items_carted', 
        'has_viewed_shipping_info', 'scroll_depth_percent', 'cart_value', 
        'discount_applied', 'discount_code', 'discount_percentage', 'shipping_fee',
        'free_shipping_eligible', 'device_type', 'browser', 'referral_source', 
        'location', 'payment_page_reached', 'most_viewed_category', 
        'total_products_viewed', 'total_events', 'cart_items_count', 
        'cart_items_names', 'viewed_categories', 'current_page', 'engagement_score',
        'abandoned'
    ]
    
    # Create ordered dictionary
    ordered_data = {}
    for col in raw_columns:
        if col in raw_data:
            ordered_data[col] = raw_data[col]
    
    # Save to CSV
    os.makedirs('analytics_data', exist_ok=True)
    csv_file = 'analytics_data/raw_user_sessions.csv'
    
    df = pd.DataFrame([ordered_data])
    
    if os.path.exists(csv_file):
        df.to_csv(csv_file, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_file, index=False)
    
    return csv_file

def start_new_session():
    st.session_state.session_data = {
        'session_id': f"S{int(time.time())}",
        'user_id': f"U{np.random.randint(1000, 9999)}",
        'items': [],
        'start_time': datetime.now(),
        'events': [],
        'cart_value': 0.0,
        'pages_viewed': [],
        'current_page': 'home',
        'categories_viewed': set(),
        'scroll_depth': 0.0,
        'view_product_count': 0,
        'shipping_viewed': False,
        'payment_reached': False,
        'return_user': np.random.choice([0, 1], p=[0.7, 0.3]),
        'device_type': np.random.choice(list(DEVICE_TYPE_MAP.keys())),
        'browser': np.random.choice(list(BROWSER_MAP.keys())),
        'referral_source': np.random.choice(list(REFERRAL_SOURCE_MAP.keys())),
        'location': np.random.choice(list(LOCATION_MAP.keys())),
        'discount_applied': 0,
        'discount_percentage': 0,
        'discount_code': None,
        'viewed_products': [],
        'time_on_page': {}
    }
    st.session_state.viewing_product = None
    st.session_state.promo_message = None
    st.session_state.current_segment = None
    st.session_state.current_recovery_strategy = None
    st.session_state.show_new_session_btn = False 

def predict_customer_segment(features):
    try:
        print("🎯 Starting direct rule-based segmentation...")
        segment = st.session_state.segmenter.segment_single_customer(features)
        print(f"✅ Direct rule-based segment: {segment}")
        return segment
        
    except Exception as e:
        print(f"❌ Direct segmentation error: {e}")
        return "Casual Browsers"

def trigger_segmentation_and_recovery(session_data, abandoned_status):
    """
    Trigger segmentation and recovery strategy based on user behavior
    Always runs - for both abandoned and purchased sessions
    """
    try:
        # Calculate features WITH PREPROCESSING
        features = calculate_features(abandoned=abandoned_status)
        
        # Predict segment using direct rule-based logic
        segment = predict_customer_segment(features)
        
        # Get recovery/engagement strategy
        user_context = {
            "cart_value": session_data['cart_value'],
            "items_count": len(session_data['items']),
            "products_viewed": len(session_data['viewed_products']),
            "session_duration": (datetime.now() - session_data['start_time']).total_seconds(),
            "abandoned": abandoned_status
        }
        
        strategy = recovery_manager.get_recovery_strategy(segment, user_context)
        log_event('customer_segmented', None, None, None, 'system')     
        return segment, strategy
        
    except Exception as e:
        print(f"Segmentation error: {e}")
        return "Casual Browsers", recovery_manager.get_recovery_strategy("Casual Browsers")

def handle_session_end(action_type):
    if len(st.session_state.session_data['items']) > 0:
        if action_type == "abandon":
            st.session_state.session_data['payment_reached'] = False
            segment, strategy = trigger_segmentation_and_recovery(
                st.session_state.session_data, 
                abandoned_status=1
            )
            
            recovery_manager = RecoveryStrategyManager()
            executed_actions = recovery_manager.execute_segment_actions(segment, st.session_state.session_data)
            
            # Save BOTH types of data with abandoned status
            featured_csv = save_session_data(abandoned=1, segment=segment)  # Preprocessed
            raw_csv = save_raw_session_data(abandoned_status=1)  # Raw
            
            st.session_state.current_segment = segment
            st.session_state.current_recovery_strategy = strategy
            st.session_state.executed_actions = executed_actions
            st.session_state.show_new_session_btn = True
            
            st.success("📊 Cart abandoned - recovery actions triggered!")
            st.info(f"💾 Data saved: Preprocessed ({featured_csv}), Raw ({raw_csv})")
            
        elif action_type == "purchase":
            st.session_state.session_data['payment_reached'] = True
            segment, strategy = trigger_segmentation_and_recovery(
                st.session_state.session_data,
                abandoned_status=0
            )
        
            recovery_manager = RecoveryStrategyManager()
            executed_actions = recovery_manager.execute_segment_actions(segment, st.session_state.session_data)

            # Save BOTH types of data with abandoned status
            featured_csv = save_session_data(abandoned=0, segment=segment)  # Preprocessed
            raw_csv = save_raw_session_data(abandoned_status=0)  # Raw
            
            st.session_state.current_segment = segment
            st.session_state.current_recovery_strategy = strategy
            st.session_state.executed_actions = executed_actions
            st.session_state.show_new_session_btn = True
            
            st.success("🎉 Purchase complete - retention actions applied!")
            st.info(f"💾 Data saved: Preprocessed ({featured_csv}), Raw ({raw_csv})")
    else:
        st.warning("❌ Add items to cart first!")

def display_segmentation_ui(segment=None, strategy=None):
    if segment is None:
        segment = st.session_state.get('current_segment')
    if strategy is None:
        strategy = st.session_state.get('current_recovery_strategy')
    
    executed_actions = st.session_state.get('executed_actions', [])
    
    if not segment or not strategy:
        return
    
    st.sidebar.divider()
    st.sidebar.header("🎯 Customer Segmentation")
    priority_colors = {
        "Very High": "#ff4444",
        "High": "#ffaa00", 
        "Medium": "#ffdd00",
        "Low": "#44cc44"
    }
    
    color = priority_colors.get(strategy["priority"], "#4444ff")
    
    st.sidebar.markdown(f"""
    <div style="padding: 15px; border-radius: 8px; border-left: 5px solid {color}; background-color: #f8f9fa; margin: 10px 0;">
        <h4 style="margin: 0 0 8px 0; color: {color};">{segment}</h4>
        <p style="margin: 4px 0; font-size: 0.9em;"><strong>Priority:</strong> {strategy['priority']}</p>
        <p style="margin: 4px 0; font-size: 0.85em;"><strong>Channel:</strong> {strategy['channel']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    if executed_actions:
        st.sidebar.write("**✅ Applied Actions:**")
        for action in executed_actions:
            st.sidebar.markdown(f"<div style='margin: 8px 0; padding-left: 10px; border-left: 2px solid {color}; font-size: 0.85em;'>{action}</div>", unsafe_allow_html=True)
    else:
        st.sidebar.write("**Recommended Actions:**")
        for i, action in enumerate(strategy["strategies"][:3]):
            st.sidebar.markdown(f"<div style='margin: 8px 0; padding-left: 10px; border-left: 2px solid {color}; font-size: 0.85em;'>{action}</div>", unsafe_allow_html=True)

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.title("🛒 ShopEasy - Test Data Store")
    st.markdown("### Realistic shopping simulation with auto-saved features")
    
    # Show preprocessing status
    if not preprocessing_loaded:
        st.warning("⚠️ Preprocessing artifacts not loaded. Using fallback mappings.")
        st.info("💡 Run the preprocessing script first to generate label_encoders.json and scaler_info.json")
    else:
        st.success("✅ Preprocessing artifacts loaded successfully!")
    
    # Sidebar session info
    with st.sidebar:
        st.header("👤 Session Info")
        st.write(f"**Session:** `{st.session_state.session_data['session_id']}`")
        st.write(f"**User:** `{st.session_state.session_data['user_id']}`")
        st.write(f"**Return:** {'Yes ✅' if st.session_state.session_data['return_user'] else 'No ❌'}")
        st.write(f"**Device:** {st.session_state.session_data['device_type']}")
        st.write(f"**Source:** {st.session_state.session_data['referral_source']}")
        
        session_duration = (datetime.now() - st.session_state.session_data['start_time']).total_seconds()
        mins = int(session_duration // 60)
        secs = int(session_duration % 60)
        st.write(f"**Duration:** {mins}m {secs}s")
        
        st.divider()
        st.header("📊 Stats")
        st.metric("Cart Value", f"₹{st.session_state.session_data['cart_value']:.2f}")
        st.metric("Items", len(st.session_state.session_data['items']))
        st.metric("Pages Viewed", len(st.session_state.session_data['pages_viewed']))
        st.metric("Product Views", st.session_state.session_data['view_product_count'])
        st.metric("Scroll Depth", f"{st.session_state.session_data['scroll_depth']:.1f}%")
        st.metric("Total Actions", len(st.session_state.session_data['events']))
        
        if st.session_state.session_data['discount_applied']:
            st.success(f"✅ Promo: {st.session_state.session_data['discount_code']} (-{st.session_state.session_data['discount_percentage']}%)")
        
        st.divider()
        st.header("🏁 End Session")

        if st.button("🔄 New Session", width='stretch', type="secondary"):
            start_new_session()
            st.session_state.show_new_session_btn = False
            st.session_state.current_segment = None
            st.session_state.current_recovery_strategy = None
            st.rerun()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚪 Abandon", width='stretch'):
                handle_session_end("abandon")

        with col2:
            if st.button("💰 Purchase", width='stretch', type="primary"):
                handle_session_end("purchase")

        # Show "Continue" button only after abandonment/purchase
        if st.session_state.get('show_new_session_btn', False):
            st.divider()
            if st.button("🔄 Start New Session", width='stretch', type="primary"):
                start_new_session()
                st.session_state.show_new_session_btn = False
                st.session_state.current_segment = None
                st.session_state.current_recovery_strategy = None
                st.rerun()

        # Always display segmentation and strategy if they exist (PERSISTENT)
        if st.session_state.get('current_segment') and st.session_state.get('current_recovery_strategy'):
            display_segmentation_ui()  # This will use the segment and strategy from session state
        
        st.divider()
        st.write(f"**Total Sessions:** {len(st.session_state.all_sessions)}")
    
    # Check if viewing product details
    if st.session_state.viewing_product:
        product = st.session_state.viewing_product
        navigate_page('product_detail')
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(product['image'], width='stretch')
            st.markdown(f"""
            ### {product['name']}
            **Price:** ₹{product['price']:.2f}
            **Category:** {product['category']}
            **Description:** {product['description']}
            **Rating:** ⭐⭐⭐⭐⭐ (4.8/5)
            **In Stock:** ✅ Yes
            """)
        
        with col2:
            st.write("")
            if st.button("🛒 Add to Cart", width='stretch', type="primary", key="add_detail"):
                add_to_cart(product)
                st.success(f"✅ Added to cart!")
                time.sleep(1)
                st.session_state.viewing_product = None
                st.rerun()
            
            if st.button("💖 Wishlist", width='stretch'):
                st.info("❤️ Added to wishlist!")
            
            if st.button("← Back", width='stretch'):
                st.session_state.viewing_product = None
                st.rerun()
        
        # Simulated scroll depth tracker
        st.divider()
        scroll_val = st.slider("📜 Scroll depth on this page", 0, 100, int(st.session_state.session_data['scroll_depth']), key="product_scroll")
        update_scroll_depth(scroll_val)
        
        st.divider()
        similar = [p for p in PRODUCTS if p['category'] == product['category'] and p['id'] != product['id']][:3]
        if similar:
            st.write("**Similar Products:**")
            cols = st.columns(3)
            for i, sim_prod in enumerate(similar):
                with cols[i]:
                    st.image(sim_prod['image'], width='stretch')
                    st.write(f"**{sim_prod['name']}**")
                    st.write(f"₹{sim_prod['price']:.0f}")
                    if st.button("👁️ View", key=f"sim_{sim_prod['id']}", width='stretch'):
                        view_product(sim_prod)
                        st.session_state.viewing_product = sim_prod
                        st.rerun()
    
        return

    navigate_page('home')
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("🛍️ Browse Products")
        scroll_val = st.slider("📜 Scroll depth", 0, 100, int(st.session_state.session_data['scroll_depth']), key="main_scroll")
        update_scroll_depth(scroll_val)
        category_filter = st.selectbox("Filter by Category", ["All"] + list(set([p['category'] for p in PRODUCTS])))
        filtered_products = PRODUCTS if category_filter == "All" else [p for p in PRODUCTS if p['category'] == category_filter] 
        
        for i in range(0, len(filtered_products), 3):
            cols = st.columns(3)
            for j, product in enumerate(filtered_products[i:i+3]):
                with cols[j]:
                    st.image(product['image'], width='stretch')
                    st.subheader(product['name'])
                    st.write(f"**₹{product['price']:.0f}**")
                    st.caption(product['category'])
                    
                    btn_col1, btn_col2 = st.columns(2)
                    with btn_col1:
                        if st.button("🛒 Add", key=f"add_{product['id']}", width='stretch'):
                            add_to_cart(product)
                            st.toast(f"✅ Added {product['name']}")
                    with btn_col2:
                        if st.button("👁️ View", key=f"view_{product['id']}", width='stretch'):
                            view_product(product)
                            st.session_state.viewing_product = product
                            st.rerun()
    
    with col2:
        st.header("🛒 Cart")
        
        if st.session_state.session_data['items']:
            total = st.session_state.session_data['cart_value']
            discount = 0
            
            if st.session_state.session_data['discount_applied']:
                discount_pct = st.session_state.session_data['discount_percentage']
                discount = total * (discount_pct / 100)
                final_total = total - discount
                st.write(f"**Subtotal:** ₹{total:.2f}")
                st.success(f"**Discount (-{discount_pct}%):** -₹{discount:.2f}")
                st.write(f"**Total:** ₹{final_total:.2f}")
            else:
                st.write(f"**Total:** ₹{total:.2f}")
            
            cart_value_scaled = total * 100
            if cart_value_scaled >= 20000:
                st.success("🎉 FREE SHIPPING!")
            else:
                remaining = (20000 - cart_value_scaled) / 100
                st.info(f"Add ₹{remaining:.2f} for free shipping")
            
            st.divider()
            
            # Checkout section
            st.write("**Checkout:**")
            
            with st.expander("🚚 Shipping Info", expanded=False):
                st.session_state.session_data['shipping_viewed'] = True
                log_event('view_shipping', None, None, None, 'shipping')
                st.markdown("""
                **Standard Delivery:** 3-5 business days
                **Express Delivery:** 1-2 business days (₹100 extra)
                **Free Shipping:** On orders above ₹200
                """)
                st.success("✅ Shipping info viewed")
            
            with st.expander("🎟️ Promo Code", expanded=False):
                promo_code = st.text_input("Enter promo code", placeholder="e.g., SAVE10", key="promo_input_real")
                if st.button("Apply", key="apply_promo"):
                    if promo_code:
                        valid, discount_pct = apply_promo_code(promo_code)
                        if valid:
                            st.session_state.promo_message = f"✅ Applied! {promo_code} (-{discount_pct}%)"
                        else:
                            st.session_state.promo_message = "❌ Invalid promo code. Try: SAVE10, SAVE20, WELCOME, SUMMER50"
                
                if st.session_state.promo_message:
                    if "✅" in st.session_state.promo_message:
                        st.success(st.session_state.promo_message)
                    else:
                        st.error(st.session_state.promo_message)
            
            st.divider()

            for item in st.session_state.session_data['items']:
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.write(f"• {item['name']}")
                    st.caption(f"₹{item['price']:.2f}")
                with col_b:
                    if st.button("❌", key=f"rm_{item['id']}_{id(item)}", width='stretch'):
                        remove_from_cart(item['id'])
                        st.rerun()
            st.divider()
            st.subheader("💡 You Might Also Like")
            
            if 'recommender' not in st.session_state:
                st.session_state.recommender = AprioriRecommender()
            
            recommendations = st.session_state.recommender.get_recommendations(
                st.session_state.session_data['items']
            )
            
            if recommendations:
                for product in recommendations:
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.write(f"• {product['name']}")
                        st.caption(f"₹{product['price']:.2f} | {product['category']}")
                    with col_b:
                        if st.button("🛒 Add", key=f"rec_{product['id']}", width='stretch'):
                            add_to_cart(product)
                            st.rerun()
            else:
                st.caption("Add more items to get personalized recommendations!")
        else:
            st.info("🛍️ Your cart is empty")
            st.caption("Browse products and add items to get started!")
    
    # Feature preview at bottom
    st.divider()
    
    st.subheader("📋 Real-time Features")
    
    with st.expander("View calculated features for model", expanded=False):
        features = calculate_features(abandoned=0)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Duration (s)", f"{features['session_duration']:.0f}")
        with col2:
            st.metric("Pages Viewed", features['num_pages_viewed'])
        with col3:
            st.metric("Items Carted", features['num_items_carted'])
        with col4:
            st.metric("Engagement", f"{features['engagement_score']:.1f}/10")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Scroll Depth", f"{features['scroll_depth']:.1f}%")
        with col2:
            st.metric("Cart (₹)", f"{features['cart_value']:.0f}")
        with col3:
            st.metric("Discount", f"{'Yes ✅' if features['discount_applied'] else 'No ❌'}")
        with col4:
            st.metric("Shipping Info", f"{'Yes ✅' if features['has_viewed_shipping_info'] else 'No ❌'}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Intensity", f"{features['engagement_intensity']:.2f}")
        with col2:
            st.metric("Weekend", "✅" if features['is_weekend'] else "❌")
        with col3:
            st.metric("Peak Hours", "✅" if features['peak_hours'] else "❌")
        with col4:
            st.metric("Payment Page", "✅" if features['if_payment_page_reached'] else "❌")
        
        st.write("**Full Feature Set (PREPROCESSED):**")
        features_df = pd.DataFrame([features])
        st.dataframe(features_df, width='stretch')
        
        # Show preprocessing status
        if preprocessing_loaded:
            st.success("Features are preprocessed using the same transformations as training data")
        else:
            st.warning("Using fallback preprocessing - run preprocessing script for exact transformations")
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Download Current Features",
                data=features_df.to_csv(index=False),
                file_name=f"session_features_{st.session_state.session_data['session_id']}.csv",
                mime="text/csv"
            )
            
if __name__ == "__main__":
    main()