import pandas as pd
import numpy as np
from itertools import combinations
from collections import defaultdict
import os
import json

class AprioriRecommender:
    """
    Real Apriori Algorithm Implementation for Product Recommendations
    
    Based on market basket analysis to discover association rules between products.
    """
    
    def __init__(self, min_support=0.02, min_confidence=0.3, min_lift=1.0):
        """
        Initialize Apriori Recommender
        
        Args:
            min_support: Minimum support threshold (0-1)
            min_confidence: Minimum confidence threshold (0-1)
            min_lift: Minimum lift threshold (>1 means positive correlation)
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.rules = []
        self.frequent_itemsets = {}  # {frozenset: support}
        self.transactions_file = 'data/transactions.csv'
        self.product_catalog = None
        
    def set_product_catalog(self, products):
        """Set the product catalog for name-to-object mapping"""
        self.product_catalog = {p['name']: p for p in products}
    
    def load_transactions(self):
        """Load transaction data from CSV or generate sample data"""
        try:
            if os.path.exists(self.transactions_file):
                df = pd.read_csv(self.transactions_file)
                
                # Filter only 'add_to_cart' and 'purchase' events
                df = df[df['event_type'].isin(['add_to_cart', 'purchase'])]
                
                # Group by session_id to create transaction baskets
                transactions = df.groupby('session_id')['product_name'].apply(list).tolist()
                
                # Remove duplicate items within same transaction
                transactions = [list(set(t)) for t in transactions]
                
                # Only use real data if we have enough SESSIONS, not rows
                if len(transactions) < 5:
                    print(f"⚠️ Only {len(transactions)} sessions found. Using sample data + real data...")
                    # Combine real + sample for better patterns
                    sample = self._generate_sample_transactions()
                    transactions.extend(sample)
                    print(f"✅ Using {len(transactions)} total transactions (real + sample)")
                else:
                    print(f"✅ Loaded {len(transactions)} real transactions from file")
                
                return transactions
            else:
                print("📝 No transaction file found. Using sample data...")
                return self._generate_sample_transactions()
                
        except Exception as e:
            print(f"❌ Error loading transactions: {e}")
            return self._generate_sample_transactions()
    
    def _generate_sample_transactions(self):
        """Generate realistic sample transaction data matching your actual products"""
        sample_transactions = [
            # Electronics combinations
            ["Smartphone", "Phone Case", "Wireless Earbuds", "Screen Protector"],
            ["Smartphone", "Power Bank", "Screen Protector"],
            ["Laptop", "Backpack", "Wireless Mouse", "Laptop Sleeve"],
            ["Laptop", "Wireless Mouse", "Screen Protector"],
            ["Laptop", "Power Bank", "Backpack"],
            ["Wireless Earbuds", "Phone Case", "Power Bank"],
            ["Bluetooth Speaker", "Power Bank", "Wireless Earbuds"],
            
            # Clothing combinations  
            ["T-Shirt", "Jeans", "Sunglasses"],
            ["T-Shirt", "Sports Socks", "Watch"],
            ["T-Shirt", "Jeans", "Watch"],
            ["Jeans", "Sunglasses", "Watch"],
            
            # Sports combinations
            ["Running Shoes", "Sports Socks", "Water Bottle", "Sports Bag"],
            ["Running Shoes", "Fitness Tracker", "Sports Socks"],
            ["Water Bottle", "Sports Bag", "Backpack"],
            ["Water Bottle", "Fitness Tracker", "Sports Socks"],
            ["Sports Bag", "Water Bottle", "Backpack"],
            
            # Automotive combinations
            ["Car Accessories", "Power Bank", "Phone Case"],
            ["Car Accessories", "Bluetooth Speaker"],
            
            # Home & Kitchen combinations
            ["Cookware Set", "Kitchen Utensils", "Coffee Maker"],
            ["Coffee Maker", "Coffee Beans", "Desk Lamp"],
            ["Cookware Set", "Kitchen Utensils"],
            
            # Beauty combinations
            ["Skincare Kit", "Face Mask", "Sunglasses"],
            ["Skincare Kit", "Face Mask"],
            
            # Mixed practical combinations
            ["Novel Collection", "Desk Lamp", "Coffee Maker"],
            ["Action Figure", "Novel Collection"],
            ["Organic Snacks", "Coffee Beans", "Water Bottle"],
            ["Backpack", "Laptop Sleeve", "Water Bottle"],
            ["Watch", "Sunglasses", "Backpack"],
            
            # More realistic based on your transaction log
            ["Smartphone", "Screen Protector", "Power Bank", "Car Accessories"],
            ["T-Shirt", "Water Bottle", "Sunglasses"],
            ["Laptop", "Smartphone", "Power Bank"],
        ]
        print(f"📊 Using {len(sample_transactions)} sample transactions")
        return sample_transactions
    
    def get_support(self, itemset, transactions):
        """Calculate support for an itemset"""
        itemset = frozenset(itemset)
        count = sum(1 for transaction in transactions if itemset.issubset(set(transaction)))
        return count / len(transactions)
    
    def find_frequent_1_itemsets(self, transactions):
        """Find all frequent 1-itemsets (single items)"""
        item_counts = defaultdict(int)
        
        # Count occurrences of each item
        for transaction in transactions:
            for item in transaction:
                item_counts[item] += 1
        
        # Filter by minimum support
        num_transactions = len(transactions)
        frequent_1 = {}
        
        for item, count in item_counts.items():
            support = count / num_transactions
            if support >= self.min_support:
                frequent_1[frozenset([item])] = support
        
        print(f"✅ Found {len(frequent_1)} frequent 1-itemsets")
        return frequent_1
    
    def generate_candidates(self, prev_frequent, k):
        """
        Generate candidate k-itemsets from (k-1)-itemsets
        Uses efficient joining method
        """
        candidates = set()
        prev_itemsets = list(prev_frequent.keys())
        
        for i in range(len(prev_itemsets)):
            for j in range(i + 1, len(prev_itemsets)):
                # Join two (k-1)-itemsets if they differ by exactly one item
                union = prev_itemsets[i] | prev_itemsets[j]
                
                if len(union) == k:
                    candidates.add(union)
        
        return candidates
    
    def prune_candidates(self, candidates, prev_frequent):
        """
        Prune candidates using Apriori principle:
        All subsets of a frequent itemset must also be frequent
        """
        pruned = set()
        
        for candidate in candidates:
            # Generate all (k-1)-subsets
            subsets = [frozenset(s) for s in combinations(candidate, len(candidate) - 1)]
            
            # Check if all subsets are frequent
            if all(subset in prev_frequent for subset in subsets):
                pruned.add(candidate)
        
        return pruned
    
    def apriori(self, transactions):
        """
        Core Apriori algorithm to find all frequent itemsets
        """
        print("🔍 Running Apriori algorithm...")
        
        # Find frequent 1-itemsets
        self.frequent_itemsets = self.find_frequent_1_itemsets(transactions)
        
        k = 2
        all_frequent = self.frequent_itemsets.copy()
        
        # Iteratively find frequent k-itemsets
        while self.frequent_itemsets:
            # Generate candidates
            candidates = self.generate_candidates(self.frequent_itemsets, k)
            
            # Prune candidates
            candidates = self.prune_candidates(candidates, self.frequent_itemsets)
            
            if not candidates:
                break
            
            # Calculate support for candidates
            frequent_k = {}
            for candidate in candidates:
                support = self.get_support(candidate, transactions)
                if support >= self.min_support:
                    frequent_k[candidate] = support
            
            if not frequent_k:
                break
            
            print(f"✅ Found {len(frequent_k)} frequent {k}-itemsets")
            
            # Update for next iteration
            all_frequent.update(frequent_k)
            self.frequent_itemsets = frequent_k
            k += 1
        
        self.frequent_itemsets = all_frequent
        print(f"✅ Total frequent itemsets: {len(self.frequent_itemsets)}")
        return self.frequent_itemsets
    
    def generate_rules(self, transactions):
        """Generate association rules from frequent itemsets"""
        print("🎯 Generating association rules...")
        self.rules = []
        
        # Get support for single items (needed for lift calculation)
        single_item_support = {
            itemset: support 
            for itemset, support in self.frequent_itemsets.items() 
            if len(itemset) == 1
        }
        
        # Generate rules from itemsets with 2+ items
        for itemset, itemset_support in self.frequent_itemsets.items():
            if len(itemset) < 2:
                continue
            
            # Generate all possible antecedent-consequent pairs
            for i in range(1, len(itemset)):
                for antecedent_tuple in combinations(itemset, i):
                    antecedent = frozenset(antecedent_tuple)
                    consequent = itemset - antecedent
                    
                    # Calculate confidence
                    antecedent_support = self.frequent_itemsets.get(antecedent, 0)
                    
                    if antecedent_support == 0:
                        continue
                    
                    confidence = itemset_support / antecedent_support
                    
                    # Calculate lift for each consequent item
                    lifts = []
                    for item in consequent:
                        item_support = single_item_support.get(frozenset([item]), 0)
                        if item_support > 0:
                            lift = confidence / item_support
                            lifts.append(lift)
                    
                    avg_lift = np.mean(lifts) if lifts else 0
                    
                    # Filter by thresholds
                    if confidence >= self.min_confidence and avg_lift >= self.min_lift:
                        self.rules.append({
                            'antecedent': antecedent,
                            'consequent': consequent,
                            'support': itemset_support,
                            'confidence': confidence,
                            'lift': avg_lift
                        })
        
        # Sort by confidence * lift (best recommendations)
        self.rules.sort(key=lambda x: x['confidence'] * x['lift'], reverse=True)
        
        print(f"✅ Generated {len(self.rules)} association rules")
        return self.rules
    
    def train(self, transactions=None):
        """Train the Apriori model"""
        if transactions is None:
            transactions = self.load_transactions()
        
        if len(transactions) < 5:
            print("⚠️ Not enough transactions for meaningful patterns")
            return self
        
        self.apriori(transactions)
        self.generate_rules(transactions)
        
        return self
    
    def get_recommendations(self, current_items, top_n=4):
        """
        Get product recommendations based on current cart items
        
        Args:
            current_items: List of product dictionaries
            top_n: Number of recommendations to return
            
        Returns:
            List of recommended product dictionaries
        """
        if not self.rules:
            self.train()
        
        if not current_items:
            return self._get_popular_items(top_n)
        
        current_item_names = frozenset(item['name'] for item in current_items)
        recommendations = defaultdict(float)  # {product_name: score}
        
        # Find matching rules and score recommendations
        for rule in self.rules:
            # Check if antecedent is subset of current items
            if rule['antecedent'].issubset(current_item_names):
                # Score = confidence * lift
                score = rule['confidence'] * rule['lift']
                
                # Add consequent items
                for item in rule['consequent']:
                    if item not in current_item_names:
                        recommendations[item] = max(recommendations[item], score)
        
        # Sort by score and get top N
        sorted_recommendations = sorted(
            recommendations.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_n]
        
        # Convert to product objects
        recommended_products = []
        for product_name, score in sorted_recommendations:
            product = self._find_product_by_name(product_name)
            if product:
                recommended_products.append(product)
        
        # Fallback if no rules matched
        if not recommended_products:
            return self._get_category_recommendations(current_items, top_n)
        
        return recommended_products
    
    def get_frequently_bought_together(self, product, top_n=3):
        """
        Get products frequently bought together with given product
        """
        if not self.rules:
            self.train()
        
        product_name = product['name']
        recommendations = defaultdict(float)
        
        # Find rules where product appears in antecedent
        for rule in self.rules:
            if product_name in rule['antecedent']:
                score = rule['confidence'] * rule['lift']
                for item in rule['consequent']:
                    if item != product_name:
                        recommendations[item] = max(recommendations[item], score)
            
            # Also check if product is in consequent
            if product_name in rule['consequent']:
                score = rule['confidence'] * rule['lift'] * 0.8  # Slightly lower weight
                for item in rule['antecedent']:
                    if item != product_name:
                        recommendations[item] = max(recommendations[item], score)
        
        # Sort and convert to products
        sorted_recommendations = sorted(
            recommendations.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        result_products = []
        for product_name, score in sorted_recommendations:
            product_obj = self._find_product_by_name(product_name)
            if product_obj:
                result_products.append(product_obj)
        
        # Fallback to same category
        if not result_products:
            return self._get_same_category_items(product, top_n)
        
        return result_products
    
    def _find_product_by_name(self, product_name):
        """Find product by name from catalog"""
        if self.product_catalog:
            return self.product_catalog.get(product_name)
        return None
    
    def _get_popular_items(self, top_n):
        """Get most popular items (highest support)"""
        single_items = [
            (list(itemset)[0], support)
            for itemset, support in self.frequent_itemsets.items()
            if len(itemset) == 1
        ]
        
        sorted_items = sorted(single_items, key=lambda x: x[1], reverse=True)[:top_n]
        
        products = []
        for item_name, _ in sorted_items:
            product = self._find_product_by_name(item_name)
            if product:
                products.append(product)
        
        return products
    
    def _get_category_recommendations(self, current_items, top_n):
        """Fallback: recommend from same categories"""
        if not self.product_catalog:
            return []
        
        current_categories = {item['category'] for item in current_items}
        current_names = {item['name'] for item in current_items}
        
        category_products = [
            p for p in self.product_catalog.values()
            if p['category'] in current_categories and p['name'] not in current_names
        ]
        
        return category_products[:top_n]
    
    def _get_same_category_items(self, product, top_n):
        """Get items from same category"""
        if not self.product_catalog:
            return []
        
        same_category = [
            p for p in self.product_catalog.values()
            if p['category'] == product['category'] and p['name'] != product['name']
        ]
        
        return same_category[:top_n]
    
    def print_rules(self, top_n=10):
        """Print top association rules for debugging"""
        print(f"\n📋 Top {top_n} Association Rules:")
        print("=" * 80)
        
        for i, rule in enumerate(self.rules[:top_n], 1):
            antecedent = ', '.join(sorted(rule['antecedent']))
            consequent = ', '.join(sorted(rule['consequent']))
            
            print(f"\n{i}. IF {{{antecedent}}} THEN {{{consequent}}}")
            print(f"   Support:    {rule['support']:.3f}")
            print(f"   Confidence: {rule['confidence']:.3f}")
            print(f"   Lift:       {rule['lift']:.3f}")
    
    def save_model(self, filepath='data/apriori_model.json'):
        """Save trained model to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'rules': [
                {
                    'antecedent': list(r['antecedent']),
                    'consequent': list(r['consequent']),
                    'support': r['support'],
                    'confidence': r['confidence'],
                    'lift': r['lift']
                }
                for r in self.rules
            ],
            'frequent_itemsets': [
                {'itemset': list(k), 'support': v}
                for k, v in self.frequent_itemsets.items()
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath='data/apriori_model.json'):
        """Load trained model from file"""
        if not os.path.exists(filepath):
            print(f"⚠️ Model file not found: {filepath}")
            return False
        
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        self.rules = [
            {
                'antecedent': frozenset(r['antecedent']),
                'consequent': frozenset(r['consequent']),
                'support': r['support'],
                'confidence': r['confidence'],
                'lift': r['lift']
            }
            for r in model_data['rules']
        ]
        
        self.frequent_itemsets = {
            frozenset(item['itemset']): item['support']
            for item in model_data['frequent_itemsets']
        }
        
        print(f"✅ Model loaded from {filepath}")
        print(f"   Rules: {len(self.rules)}, Itemsets: {len(self.frequent_itemsets)}")
        return True


# Example usage
if __name__ == "__main__":
    # Sample product catalog
    SAMPLE_PRODUCTS = [
        {"id": 1, "name": "Smartphone", "category": "Electronics", "price": 299.99},
        {"id": 2, "name": "Phone Case", "category": "Electronics", "price": 19.99},
        {"id": 3, "name": "Wireless Earbuds", "category": "Electronics", "price": 79.99},
        {"id": 4, "name": "Laptop", "category": "Electronics", "price": 899.99},
        {"id": 5, "name": "Backpack", "category": "Accessories", "price": 49.99},
    ]
    
    # Initialize and train
    recommender = AprioriRecommender(min_support=0.02, min_confidence=0.3)
    recommender.set_product_catalog(SAMPLE_PRODUCTS)
    recommender.train()
    
    # Print discovered rules
    recommender.print_rules(top_n=5)
    
    # Get recommendations
    cart_items = [SAMPLE_PRODUCTS[0]]  # Smartphone
    recommendations = recommender.get_recommendations(cart_items, top_n=3)
    
    print("\n🎁 Recommendations for Smartphone:")
    for product in recommendations:
        print(f"   - {product['name']} (${product['price']})")