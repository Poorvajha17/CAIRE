import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal
import time
import os
import warnings
warnings.filterwarnings('ignore')

class EnhancedCustomerSegmenter:
    def __init__(self, n_segments=5):
        self.n_segments = n_segments
        self.kmeans = None
        self.scaler = StandardScaler()
        self.segment_profiles = {}
        self.feature_columns = []
        self.global_metrics = {}
        self.assigned_segments = set()
        
    def _select_features(self, df):
        """Select relevant features from pre-engineered dataset"""
        # Use the pre-computed features directly
        feature_cols = [
            'engagement_score', 'num_items_carted', 'cart_value', 
            'session_duration', 'num_pages_viewed', 'scroll_depth',
            'return_user', 'if_payment_page_reached', 'discount_applied',
            'has_viewed_shipping_info', 'engagement_intensity',
            'scroll_engagement', 'has_multiple_items', 'has_high_engagement',
            'research_behavior', 'quick_browse'
        ]
        
        # Only use columns that exist in the dataframe
        available_features = [col for col in feature_cols if col in df.columns]
        
        if len(available_features) == 0:
            raise ValueError("No valid features found in dataset")
        
        print(f"📊 Using {len(available_features)} pre-engineered features for segmentation")
        return df[available_features].copy()
    
    def _create_enhanced_segments(self, df, labels):
        df_segmented = df.copy()
        df_segmented['segment'] = labels
        
        self.global_metrics = {
            'global_avg_cart_value': df['cart_value'].mean(),
            'global_avg_engagement': df['engagement_score'].mean(),
            'global_avg_abandonment': df['abandoned'].mean() * 100,
            'global_avg_return_rate': df['return_user'].mean() * 100,
            'global_avg_payment_reach': df['if_payment_page_reached'].mean() * 100,
            'global_avg_items': df['num_items_carted'].mean(),
            'global_avg_pages_viewed': df['num_pages_viewed'].mean(),
            'global_max_cart_value': df['cart_value'].max(),
            'global_min_cart_value': df['cart_value'].min()
        }
        
        segment_profiles = {}
        
        for segment_id in range(self.n_segments):
            segment_data = df_segmented[df_segmented['segment'] == segment_id]
            
            if len(segment_data) == 0:
                continue
            profile = {
                'size': len(segment_data),
                'size_percentage': len(segment_data) / len(df) * 100,
                
                'abandonment_rate': segment_data['abandoned'].mean() * 100,
                'avg_cart_value': segment_data['cart_value'].mean(),
                'avg_engagement': segment_data['engagement_score'].mean(),
                'avg_items': segment_data['num_items_carted'].mean(),
                'avg_session_duration': segment_data['session_duration'].mean(),
                'avg_pages_viewed': segment_data['num_pages_viewed'].mean(),
                'avg_scroll_depth': segment_data['scroll_depth'].mean(),
                'return_user_rate': segment_data['return_user'].mean() * 100,
                'discount_sensitivity': segment_data['discount_applied'].mean() * 100,
                'payment_reach_rate': segment_data['if_payment_page_reached'].mean() * 100,
                'shipping_info_view_rate': segment_data['has_viewed_shipping_info'].mean() * 100,
            }
            
            profile.update(self.global_metrics)
            profile.update(self._enhanced_segment_identification(profile, segment_id))
            segment_profiles[segment_id] = profile
            
        return segment_profiles
    
    def _enhanced_segment_identification(self, profile, segment_id):
        rel_abandonment = profile['abandonment_rate'] - profile['global_avg_abandonment']
        rel_cart_value = profile['avg_cart_value'] - profile['global_avg_cart_value']
        rel_engagement = profile['avg_engagement'] - profile['global_avg_engagement']
        rel_return_rate = profile['return_user_rate'] - profile['global_avg_return_rate']
        rel_payment_reach = profile['payment_reach_rate'] - profile['global_avg_payment_reach']

        segment_scores = {
            'High-Value Loyalists': 0,
            'At-Risk Converters': 0,
            'Engaged Researchers': 0,
            'Price-Sensitive Shoppers': 0,
            'Casual Browsers': 0
        }
        
        # High-Value Loyalists
        if rel_cart_value > 0.03: segment_scores['High-Value Loyalists'] += 3
        if rel_return_rate > 3: segment_scores['High-Value Loyalists'] += 2
        if rel_abandonment < -3: segment_scores['High-Value Loyalists'] += 2
        if rel_payment_reach > 5: segment_scores['High-Value Loyalists'] += 1
        if profile['avg_cart_value'] > profile['global_avg_cart_value'] * 1.1: segment_scores['High-Value Loyalists'] += 1
        
        # At-Risk Converters 
        if rel_cart_value > 0.02: segment_scores['At-Risk Converters'] += 3
        if rel_return_rate < 5: segment_scores['At-Risk Converters'] += 2
        if rel_abandonment > 3: segment_scores['At-Risk Converters'] += 2
        if rel_payment_reach > 3: segment_scores['At-Risk Converters'] += 1
        if profile['payment_reach_rate'] > 40: segment_scores['At-Risk Converters'] += 1
        
        # Engaged Researchers 
        if rel_engagement > 0.1: segment_scores['Engaged Researchers'] += 3
        if profile['avg_pages_viewed'] > profile['global_avg_pages_viewed'] * 0.9: segment_scores['Engaged Researchers'] += 2
        if profile['shipping_info_view_rate'] > 40: segment_scores['Engaged Researchers'] += 1
        if abs(rel_cart_value) < 0.1: segment_scores['Engaged Researchers'] += 1
        if profile['avg_pages_viewed'] > 5: segment_scores['Engaged Researchers'] += 1
        
        # Price-Sensitive Shoppers 
        if profile['discount_sensitivity'] > 35: segment_scores['Price-Sensitive Shoppers'] += 3
        if rel_cart_value < 0.05: segment_scores['Price-Sensitive Shoppers'] += 2
        if rel_abandonment > 0: segment_scores['Price-Sensitive Shoppers'] += 1
        if profile['discount_sensitivity'] > profile['global_avg_abandonment']: segment_scores['Price-Sensitive Shoppers'] += 1
        
        # Casual Browsers 
        if rel_engagement < 0: segment_scores['Casual Browsers'] += 3
        if rel_cart_value < 0: segment_scores['Casual Browsers'] += 2
        if rel_payment_reach < -5: segment_scores['Casual Browsers'] += 2
        if profile['avg_session_duration'] < 0: segment_scores['Casual Browsers'] += 1
        if profile['avg_items'] < profile['global_avg_items']: segment_scores['Casual Browsers'] += 1
        
        # Get the highest scoring segment
        best_segment = max(segment_scores, key=segment_scores.get)
        best_score = segment_scores[best_segment]
        
        if best_segment in self.assigned_segments and best_score < 5:
            sorted_segments = sorted(segment_scores.items(), key=lambda x: x[1], reverse=True)
            for segment_name, score in sorted_segments:
                if segment_name not in self.assigned_segments and score >= 2:  # Lower threshold
                    best_segment = segment_name
                    best_score = score
                    break
        
        if best_score < 2 or best_segment in self.assigned_segments:
            best_segment = self._simplified_segment_fallback(profile, force_assign=True)
            
        self.assigned_segments.add(best_segment)
        
        segment_info = {
            'segment_name': best_segment,
            'description': self._get_segment_description(best_segment),
            'recovery_priority': self._get_segment_priority(best_segment, profile),
            'business_value': self._get_business_value(best_segment),
            'recovery_priority_score': self._calculate_enhanced_priority(profile),
            'segment_score': best_score
        }
        
        return segment_info
    
    def _simplified_segment_fallback(self, profile, force_assign=False):
        """Fallback method with optional force assignment to ensure all segments are used"""
        available_segments = [
            'High-Value Loyalists',
            'At-Risk Converters', 
            'Engaged Researchers',
            'Price-Sensitive Shoppers',
            'Casual Browsers'
        ]
        
        # If forcing assignment, find the first unassigned segment
        if force_assign:
            for segment in available_segments:
                if segment not in self.assigned_segments:
                    return segment
        
        # Original fallback logic
        if profile['return_user_rate'] > 50:
            if profile['avg_cart_value'] > profile['global_avg_cart_value']:
                if "High-Value Loyalists" not in self.assigned_segments:
                    return "High-Value Loyalists"
                else:
                    return "At-Risk Converters" 
            else:
                return "Price-Sensitive Shoppers" 
        
        if profile['avg_cart_value'] > profile['global_avg_cart_value']:
            if "At-Risk Converters" not in self.assigned_segments:
                return "At-Risk Converters"
        
        if profile['avg_engagement'] > profile['global_avg_engagement']:
            if "Engaged Researchers" not in self.assigned_segments:
                return "Engaged Researchers"

        if profile['discount_sensitivity'] > 40:
            if "Price-Sensitive Shoppers" not in self.assigned_segments:
                return "Price-Sensitive Shoppers"
        
        # Return first unassigned segment
        for segment in available_segments:
            if segment not in self.assigned_segments:
                return segment        
        return "Casual Browsers"
    
    def _get_segment_description(self, segment_name):
        descriptions = {
            "High-Value Loyalists": "Frequent high-spending customers with low abandonment rates",
            "At-Risk Converters": "High-value new customers who need conversion encouragement",
            "Engaged Researchers": "Highly engaged users researching products before purchase",
            "Price-Sensitive Shoppers": "Customers highly responsive to discounts and promotions",
            "Casual Browsers": "Low-engagement users exploring with minimal intent"
        }
        return descriptions.get(segment_name, "Users with typical shopping behavior")
    
    def _get_segment_priority(self, segment_name, profile):
        base_priority = {
            "At-Risk Converters": "Very High",
            "Engaged Researchers": "High",
            "Price-Sensitive Shoppers": "Medium",
            "High-Value Loyalists": "Low", 
            "Casual Browsers": "Low"
        }
        
        priority = base_priority.get(segment_name, "Medium")
        
        if profile['abandonment_rate'] > 70:
            priority = "Very High"
        elif profile['abandonment_rate'] > 50 and priority == "Medium":
            priority = "High"
            
        return priority
    
    def _get_business_value(self, segment_name):
        value_map = {
            "High-Value Loyalists": "Very High",
            "At-Risk Converters": "High",
            "Engaged Researchers": "Medium-High",
            "Price-Sensitive Shoppers": "Medium",
            "Casual Browsers": "Low"
        }
        return value_map.get(segment_name, "Medium")
    
    def _calculate_enhanced_priority(self, profile):
        weights = {
            'abandonment_rate': 0.35,
            'avg_cart_value': 0.20,
            'return_user_rate': 0.15,
            'payment_reach_rate': 0.15,
            'avg_engagement': 0.10,
            'discount_sensitivity': 0.05
        }
        
        abandonment_score = min(profile['abandonment_rate'] / 100, 1.0)
        cart_range = profile['global_max_cart_value'] - profile['global_min_cart_value']
        cart_value_score = (profile['avg_cart_value'] - profile['global_min_cart_value']) / cart_range
        return_user_score = profile['return_user_rate'] / 100
        payment_score = profile['payment_reach_rate'] / 100
        engagement_score = (profile['avg_engagement'] + 1) / 2 if profile['avg_engagement'] >= -1 else 0
        discount_score = profile['discount_sensitivity'] / 100
        
        priority_score = (
            weights['abandonment_rate'] * abandonment_score +
            weights['avg_cart_value'] * cart_value_score +
            weights['return_user_rate'] * return_user_score +
            weights['payment_reach_rate'] * payment_score +
            weights['avg_engagement'] * engagement_score +
            weights['discount_sensitivity'] * discount_score
        )
        
        return min(int(priority_score * 100), 100)
    
    def _evaluate_clustering(self, X_scaled, labels, method_name):
        try:
            silhouette = silhouette_score(X_scaled, labels)
            calinski = calinski_harabasz_score(X_scaled, labels)
            davies = davies_bouldin_score(X_scaled, labels)
            
            evaluation = {
                'method': method_name,
                'silhouette_score': silhouette,
                'calinski_harabasz_score': calinski,
                'davies_bouldin_score': davies,
                'n_clusters': len(np.unique(labels))
            }
            
            print(f"📊 {method_name} Evaluation:")
            print(f"   Silhouette Score: {silhouette :.4f} (higher is better)")
            print(f"   Calinski-Harabasz: {calinski :.4f} (higher is better)")
            print(f"   Davies-Bouldin: {davies :.4f} (lower is better)")
            
            return evaluation
        except Exception as e:
            print(f"⚠️ Evaluation failed for {method_name}: {e}")
            return None
    
    def kmeans_from_scratch(self, X, max_iters=100, restarts=5):
        """KMeans clustering from scratch with multiple restarts"""
        n_samples, n_features = X.shape
        best_inertia = np.inf
        best_labels, best_centroids = None, None

        for _ in range(restarts):
            init_idx = np.random.choice(n_samples, self.n_segments, replace=False)
            centroids = X[init_idx]

            for _ in range(max_iters):
                distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
                labels = np.argmin(distances, axis=1)

                new_centroids = np.array([
                    X[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
                    for i in range(self.n_segments)
                ])

                if np.allclose(centroids, new_centroids):
                    break
                centroids = new_centroids

            inertia = sum(((X[labels == i] - centroids[i]) ** 2).sum() for i in range(self.n_segments))

            if inertia < best_inertia:
                best_inertia, best_labels, best_centroids = inertia, labels, centroids

        self.centroids = best_centroids
        return best_labels, best_centroids, best_inertia
    
    def gmm_from_scratch(self, X, max_iters=100, tol=1e-4):
        """Gaussian Mixture Model from scratch using EM algorithm"""
        n_samples, n_features = X.shape
        n_components = self.n_segments
        
        # Initialize parameters using KMeans
        np.random.seed(42)
        kmeans_labels, kmeans_centroids, _ = self.kmeans_from_scratch(X, max_iters=50, restarts=3)
        
        means = kmeans_centroids.copy()
        covariances = np.array([np.eye(n_features) for _ in range(n_components)])
        
        weights = np.ones(n_components) / n_components
        
        log_likelihood_old = -np.inf
        
        for iteration in range(max_iters):
            responsibilities = np.zeros((n_samples, n_components))
            
            for k in range(n_components):
                try:
                    diff = X - means[k]
                    cov_inv = np.linalg.inv(covariances[k])
                    cov_det = np.linalg.det(covariances[k])
                    
                    exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
                    normalization = 1.0 / np.sqrt((2 * np.pi) ** n_features * cov_det)
                    
                    responsibilities[:, k] = weights[k] * normalization * np.exp(exponent)
                except np.linalg.LinAlgError:
                    covariances[k] = np.eye(n_features) * 0.01
                    responsibilities[:, k] = weights[k] * 1e-10
            
            responsibilities_sum = responsibilities.sum(axis=1, keepdims=True)
            responsibilities_sum[responsibilities_sum == 0] = 1e-10
            responsibilities /= responsibilities_sum
            
            Nk = responsibilities.sum(axis=0)
            
            weights = Nk / n_samples
            
            for k in range(n_components):
                means[k] = (responsibilities[:, k, np.newaxis] * X).sum(axis=0) / (Nk[k] + 1e-10)
            
            for k in range(n_components):
                diff = X - means[k]
                cov = (responsibilities[:, k, np.newaxis, np.newaxis] * 
                       diff[:, :, np.newaxis] @ diff[:, np.newaxis, :]).sum(axis=0)
                covariances[k] = cov / (Nk[k] + 1e-10)
                
                covariances[k] += np.eye(n_features) * 1e-6
            
            log_likelihood = 0
            for k in range(n_components):
                try:
                    diff = X - means[k]
                    cov_inv = np.linalg.inv(covariances[k])
                    cov_det = np.linalg.det(covariances[k])
                    
                    exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
                    normalization = 1.0 / np.sqrt((2 * np.pi) ** n_features * cov_det)
                    
                    log_likelihood += np.sum(np.log(weights[k] * normalization * np.exp(exponent) + 1e-10))
                except:
                    pass
            
            if abs(log_likelihood - log_likelihood_old) < tol:
                break
            
            log_likelihood_old = log_likelihood
        
        labels = np.argmax(responsibilities, axis=1)
        
        # Store parameters for prediction
        self.gmm_means = means
        self.gmm_covariances = covariances
        self.gmm_weights = weights
        
        return labels, means, covariances, weights
    
    def hdbscan_from_scratch(self, X, min_cluster_size=30, min_samples=10):
        """Optimized HDBSCAN implementation from scratch"""
        n_samples = X.shape[0]
        
        print(f"   Computing distances for {n_samples} samples...")
        distances = cdist(X, X, metric='euclidean')
        
        print(f"   Calculating core distances...")
        sorted_dists = np.sort(distances, axis=1)
        core_distances = sorted_dists[:, min_samples]
        
        print(f"   Computing mutual reachability distances...")
        core_dist_matrix = np.maximum(core_distances[:, np.newaxis], core_distances)
        mutual_reach_dist = np.maximum(core_dist_matrix, distances)
        
        print(f"   Building Minimum Spanning Tree...")
        mst_edges = []
        visited = np.zeros(n_samples, dtype=bool)
        visited[0] = True
        
        min_distances = mutual_reach_dist[0].copy()
        min_distances[0] = np.inf
        
        for _ in range(n_samples - 1):
            unvisited_mask = ~visited
            min_distances[visited] = np.inf
            min_j = np.argmin(min_distances)
            min_dist = min_distances[min_j]
            
            visited_indices = np.where(visited)[0]
            min_i = visited_indices[np.argmin(mutual_reach_dist[visited_indices, min_j])]
            
            mst_edges.append((min_i, min_j, min_dist))
            visited[min_j] = True
            
            min_distances = np.minimum(min_distances, mutual_reach_dist[min_j])
        
        print(f"   Extracting clusters...")
        mst_edges.sort(key=lambda x: x[2])
        
        parent = np.arange(n_samples)
        cluster_size = np.ones(n_samples, dtype=int)
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            root_x, root_y = find(x), find(y)
            if root_x != root_y:
                if cluster_size[root_x] < cluster_size[root_y]:
                    root_x, root_y = root_y, root_x
                parent[root_y] = root_x
                cluster_size[root_x] += cluster_size[root_y]
            return cluster_size[root_x]
        
        labels = np.full(n_samples, -1, dtype=int)
        current_cluster = 0
        cluster_formed = np.zeros(n_samples, dtype=bool)
        
        for i, j, dist in mst_edges:
            root_i, root_j = find(i), find(j)
            size = union(i, j)
            new_root = find(i)
            
            if size >= min_cluster_size and not cluster_formed[new_root]:
                cluster_formed[new_root] = True
                # Label all points in this cluster
                for k in range(n_samples):
                    if find(k) == new_root:
                        labels[k] = current_cluster
                current_cluster += 1
        
        noise_mask = labels == -1
        if np.any(noise_mask):
            labeled_mask = labels != -1
            if np.any(labeled_mask):
                noise_indices = np.where(noise_mask)[0]
                labeled_indices = np.where(labeled_mask)[0]
                
                noise_to_labeled_dist = distances[np.ix_(noise_indices, labeled_indices)]
                nearest_labeled = labeled_indices[np.argmin(noise_to_labeled_dist, axis=1)]
                labels[noise_indices] = labels[nearest_labeled]
        
        self.hdbscan_data = X
        self.hdbscan_labels = labels
        
        print(f"   HDBSCAN completed!")
        return labels

    def predict_segment(self, df):
        if not hasattr(self, 'feature_columns') or len(self.feature_columns) == 0:
            raise ValueError("Segmenter must be fitted before prediction")

        features_df = self._select_features(df)
        missing_cols = set(self.feature_columns) - set(features_df.columns)
        if missing_cols:
            raise ValueError(f"Missing features for prediction: {missing_cols}")
        
        X = features_df[self.feature_columns].values
        X_scaled = self.scaler.transform(X)

        # Use appropriate prediction method based on selected algorithm
        if self.selected_method == 'kmeans':
            if self.centroids is not None:
                distances = np.linalg.norm(X_scaled[:, np.newaxis] - self.centroids, axis=2)
                labels = np.argmin(distances, axis=1)
                return labels
            else:
                raise ValueError("No KMeans centroids found. Fit the model first.")
        
        elif self.selected_method == 'gmm':
            if hasattr(self, 'gmm_means'):
                # Predict using GMM parameters
                n_samples = X_scaled.shape[0]
                n_components = len(self.gmm_means)
                responsibilities = np.zeros((n_samples, n_components))
                
                for k in range(n_components):
                    try:
                        diff = X_scaled - self.gmm_means[k]
                        cov_inv = np.linalg.inv(self.gmm_covariances[k])
                        cov_det = np.linalg.det(self.gmm_covariances[k])
                        
                        exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
                        normalization = 1.0 / np.sqrt((2 * np.pi) ** X_scaled.shape[1] * cov_det)
                        
                        responsibilities[:, k] = self.gmm_weights[k] * normalization * np.exp(exponent)
                    except:
                        responsibilities[:, k] = 1e-10
                
                labels = np.argmax(responsibilities, axis=1)
                return labels
            else:
                raise ValueError("No GMM parameters found. Fit the model first.")
        
        elif self.selected_method == 'hdbscan':
            if hasattr(self, 'hdbscan_data') and hasattr(self, 'hdbscan_labels'):
                # For HDBSCAN, assign to nearest cluster centroid
                unique_labels = np.unique(self.hdbscan_labels[self.hdbscan_labels != -1])
                cluster_centers = np.array([
                    self.hdbscan_data[self.hdbscan_labels == label].mean(axis=0)
                    for label in unique_labels
                ])
                
                distances = cdist(X_scaled, cluster_centers, metric='euclidean')
                labels = unique_labels[np.argmin(distances, axis=1)]
                return labels
            else:
                raise ValueError("No HDBSCAN data found. Fit the model first.")
        
        else:
            raise ValueError(f"Unknown method: {self.selected_method}")

    def fit(self, df, method='auto'):
        """Perform enhanced segmentation with KMeans, GMM, or HDBSCAN"""
        try:
            print("\n🔄 Performing Enhanced Customer Segmentation (5 Segments)...")

            features_df = self._select_features(df)
            self.feature_columns = features_df.columns.tolist()
            X = features_df.values
            X_scaled = self.scaler.fit_transform(X)

            best_score = -1
            best_labels = None
            best_method = None
            evaluations = []

            # KMEANS
            print("\n🔍 Testing KMeans clustering...")
            start_time = time.time()
            kmeans_labels, kmeans_centroids, kmeans_inertia = self.kmeans_from_scratch(X_scaled)
            print(f"[INFO] KMeans from scratch | Inertia: {kmeans_inertia:.2f}")
            kmeans_time = time.time() - start_time

            kmeans_eval = self._evaluate_clustering(X_scaled, kmeans_labels, "KMeans")
            if kmeans_eval:
                kmeans_eval['time'] = kmeans_time
                evaluations.append(kmeans_eval)
                if kmeans_eval['silhouette_score'] > best_score:
                    best_score = kmeans_eval['silhouette_score']
                    best_labels = kmeans_labels
                    best_method = 'kmeans'

            # GMM
            print("\n🔍 Testing Gaussian Mixture Model (from scratch)...")
            start_time = time.time()
            gmm_labels, gmm_means, gmm_covs, gmm_weights = self.gmm_from_scratch(X_scaled)
            gmm_time = time.time() - start_time
            print(f"[INFO] GMM from scratch completed")

            gmm_eval = self._evaluate_clustering(X_scaled, gmm_labels, "GMM")
            if gmm_eval:
                gmm_eval['time'] = gmm_time
                evaluations.append(gmm_eval)
                if gmm_eval['silhouette_score'] > best_score:
                    best_score = gmm_eval['silhouette_score']
                    best_labels = gmm_labels
                    best_method = 'gmm'

            # HDBSCAN
            print("\n🔍 Testing HDBSCAN clustering (from scratch)...")
            start_time = time.time()
            n_samples = X_scaled.shape[0]
            hdb_labels = self.hdbscan_from_scratch(X_scaled, min_cluster_size=min(30, n_samples // 10))
            hdb_time = time.time() - start_time

            unique_clusters = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
            print(f"🌀 HDBSCAN found {unique_clusters} clusters (excluding noise)")

            if unique_clusters > 1 and np.sum(hdb_labels != -1) > 0:
                hdb_eval = self._evaluate_clustering(X_scaled[hdb_labels != -1], hdb_labels[hdb_labels != -1], "HDBSCAN")
                if hdb_eval:
                    hdb_eval['time'] = hdb_time
                    evaluations.append(hdb_eval)
                    if hdb_eval['silhouette_score'] > best_score:
                        best_score = hdb_eval['silhouette_score']
                        best_labels = hdb_labels
                        best_method = 'hdbscan'
            else:
                print("⚠️ HDBSCAN produced only noise or a single cluster — skipping evaluation.")

            # Manual override
            if method != 'auto':
                if method == 'kmeans':
                    best_labels = kmeans_labels
                    best_method = 'kmeans'
                elif method == 'gmm':
                    best_labels = gmm_labels
                    best_method = 'gmm'
                elif method == 'hdbscan':
                    best_labels = hdb_labels
                    best_method = 'hdbscan'

            print("\n✅ Selected:", best_method.upper())
            
            # Store selected method
            self.selected_method = best_method

            self.segment_profiles = self._create_enhanced_segments(df, best_labels)
            print("🎯 Enhanced segmentation completed!")
            print(f"📋 Assigned segments: {self.assigned_segments}")
            return self

        except Exception as e:
            print(f"Error in segmentation: {e}")
            raise

def main():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        featured_path = os.path.join(project_root, 'data', 'cart_abandonment_featured.csv')
        print("=== Loading Dataset ===")
        print(f"Featured data: {featured_path}")
        
        if os.path.exists(featured_path):
            df = pd.read_csv(featured_path)
            print(f"✅ Featured dataset loaded: {df.shape}")
            print(f"📊 Dataset overview:")
            print(f"   - Abandonment rate: {df['abandoned'].mean():.1%}")
            print(f"   - Cart value range: {df['cart_value'].min():.3f} to {df['cart_value'].max():.3f}")
            print(f"   - Engagement score range: {df['engagement_score'].min():.3f} to {df['engagement_score'].max():.3f}")
            print(f"   - Return user rate: {df['return_user'].mean():.1%}")
        else:
            print("Featured dataset not found!")
            return
        
        print("\n🔄 Performing Enhanced Customer Segmentation (5 Segments)...")
        
        segmenter = EnhancedCustomerSegmenter(n_segments=5)
        segmenter.fit(df, method='auto')
        
        df['segment'] = segmenter.predict_segment(df)
        
        print("\n" + "="*60)
        print("📊 SIMPLIFIED 5-SEGMENT CUSTOMER ANALYSIS")
        print("="*60)
        
        for segment_id, profile in segmenter.segment_profiles.items():
            print(f"\n🎯 Segment {segment_id}: {profile['segment_name']}")
            print(f"   📈 Size: {profile['size']} users ({profile['size_percentage']:.1f}%)")
            print(f"   💰 Avg Cart Value: {profile['avg_cart_value']:.3f}")
            print(f"   🚫 Abandonment Rate: {profile['abandonment_rate']:.1f}%")
            print(f"   🔄 Return User Rate: {profile['return_user_rate']:.1f}%")
            print(f"   💳 Payment Reach Rate: {profile['payment_reach_rate']:.1f}%")
            print(f"   ⭐ Recovery Priority: {profile['recovery_priority']} ({profile['recovery_priority_score']}/100)")
            print(f"   📝 Description: {profile['description']}")
            print(f"   💼 Business Value: {profile['business_value']}")
            
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()