import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from utils.config import get_config
from utils.helpers import save_plot, save_model, save_report

class JobClusterer:
    """Class to handle job clustering using machine learning models."""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.df = None
        self.X = None
        self.X_scaled = None
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.best_model = None
        self.best_score = 0
        self.cluster_labels = None
        self.n_clusters = 5  # Default number of clusters
        
    def load_data(self, file_path=None):
        """Load the dataset for clustering."""
        if file_path is None:
            file_path = self.config['dataset']['cleaned_file']
        
        self.df = pd.read_csv(file_path)
        print(f"Loaded dataset with {len(self.df)} rows for job clustering")
        return self.df
    
    def prepare_features(self, feature_columns=None):
        """Prepare features for job clustering."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if feature_columns is None:
            feature_columns = [
                'salary_usd', 'employment_type', 'company_size', 
                'remote_ratio', 'education_required', 'years_experience',
                'job_description_length', 'benefits_score'
            ]
        
        print(f"Preparing features for clustering: {feature_columns}")
        
        # Create feature matrix
        self.X = self.df[feature_columns].copy()
        
        # Encode categorical variables
        categorical_columns = ['employment_type', 'company_size', 'education_required']
        
        for col in categorical_columns:
            if col in self.X.columns:
                le = LabelEncoder()
                self.X[col] = le.fit_transform(self.X[col].astype(str))
                self.label_encoders[col] = le
        
        # Scale features
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        print(f"Feature matrix shape: {self.X.shape}")
        print(f"Features: {list(self.X.columns)}")
        
        return self.X, self.X_scaled
    
    def find_optimal_clusters(self, max_clusters=10):
        """Find optimal number of clusters using elbow method and silhouette analysis."""
        print("Finding optimal number of clusters...")
        
        # Calculate metrics for different numbers of clusters
        inertias = []
        silhouette_scores = []
        calinski_scores = []
        davies_scores = []
        
        K_range = range(2, max_clusters + 1)
        
        for k in K_range:
            print(f"Testing {k} clusters...")
            
            # Fit KMeans
            kmeans = KMeans(n_clusters=k, random_state=self.config['models']['random_state'], n_init=10)
            cluster_labels = kmeans.fit_predict(self.X_scaled)
            
            # Calculate metrics
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.X_scaled, cluster_labels))
            calinski_scores.append(calinski_harabasz_score(self.X_scaled, cluster_labels))
            davies_scores.append(davies_bouldin_score(self.X_scaled, cluster_labels))
        
        # Create elbow plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Optimal Number of Clusters Analysis', fontsize=16)
        
        # Elbow plot
        axes[0, 0].plot(K_range, inertias, 'bo-')
        axes[0, 0].set_xlabel('Number of Clusters')
        axes[0, 0].set_ylabel('Inertia')
        axes[0, 0].set_title('Elbow Method')
        axes[0, 0].grid(True)
        
        # Silhouette score
        axes[0, 1].plot(K_range, silhouette_scores, 'ro-')
        axes[0, 1].set_xlabel('Number of Clusters')
        axes[0, 1].set_ylabel('Silhouette Score')
        axes[0, 1].set_title('Silhouette Analysis')
        axes[0, 1].grid(True)
        
        # Calinski-Harabasz score
        axes[1, 0].plot(K_range, calinski_scores, 'go-')
        axes[1, 0].set_xlabel('Number of Clusters')
        axes[1, 0].set_ylabel('Calinski-Harabasz Score')
        axes[1, 0].set_title('Calinski-Harabasz Index')
        axes[1, 0].grid(True)
        
        # Davies-Bouldin score (lower is better)
        axes[1, 1].plot(K_range, davies_scores, 'mo-')
        axes[1, 1].set_xlabel('Number of Clusters')
        axes[1, 1].set_ylabel('Davies-Bouldin Score')
        axes[1, 1].set_title('Davies-Bouldin Index (Lower is Better)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        save_plot(fig, 'optimal_clusters_analysis.png')
        plt.close()
        
        # Find optimal k based on silhouette score
        optimal_k = K_range[np.argmax(silhouette_scores)]
        self.n_clusters = optimal_k
        
        print(f"Optimal number of clusters: {optimal_k}")
        print(f"Silhouette score: {max(silhouette_scores):.4f}")
        
        return {
            'K_range': list(K_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'calinski_scores': calinski_scores,
            'davies_scores': davies_scores,
            'optimal_k': optimal_k
        }
    
    def train_clustering_models(self, n_clusters=None):
        """Train multiple clustering models."""
        if self.X_scaled is None:
            raise ValueError("Features not prepared. Call prepare_features() first.")
        
        if n_clusters is None:
            n_clusters = self.n_clusters
        
        print(f"Training clustering models with {n_clusters} clusters...")
        
        # Define models to train
        models = {
            'K-Means': KMeans(n_clusters=n_clusters, random_state=self.config['models']['random_state'], n_init=10),
            'Agglomerative': AgglomerativeClustering(n_clusters=n_clusters),
            'Gaussian Mixture': GaussianMixture(n_components=n_clusters, random_state=self.config['models']['random_state']),
            'DBSCAN': DBSCAN(eps=0.5, min_samples=5)  # Note: DBSCAN doesn't use n_clusters
        }
        
        # Train and evaluate each model
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train the model
            if name == 'Gaussian Mixture':
                cluster_labels = model.fit_predict(self.X_scaled)
            else:
                model.fit(self.X_scaled)
                cluster_labels = model.labels_
            
            # Calculate metrics (skip for DBSCAN if too many noise points)
            if name == 'DBSCAN' and -1 in cluster_labels:
                n_noise = list(cluster_labels).count(-1)
                if n_noise > len(cluster_labels) * 0.1:  # More than 10% noise
                    print(f"  Skipping metrics for {name} due to high noise ({n_noise} noise points)")
                    results[name] = {
                        'model': model,
                        'cluster_labels': cluster_labels,
                        'n_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
                        'n_noise': n_noise
                    }
                    self.models[name] = model
                    continue
            
            # Calculate clustering metrics
            silhouette = silhouette_score(self.X_scaled, cluster_labels)
            calinski = calinski_harabasz_score(self.X_scaled, cluster_labels)
            davies = davies_bouldin_score(self.X_scaled, cluster_labels)
            
            results[name] = {
                'model': model,
                'cluster_labels': cluster_labels,
                'silhouette_score': silhouette,
                'calinski_score': calinski,
                'davies_score': davies,
                'n_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            }
            
            self.models[name] = model
            
            print(f"  Silhouette Score: {silhouette:.4f}")
            print(f"  Calinski-Harabasz Score: {calinski:.4f}")
            print(f"  Davies-Bouldin Score: {davies:.4f}")
            print(f"  Number of clusters: {results[name]['n_clusters']}")
        
        # Find best model based on silhouette score
        valid_models = {k: v for k, v in results.items() if 'silhouette_score' in v}
        if valid_models:
            best_model_name = max(valid_models.keys(), key=lambda x: valid_models[x]['silhouette_score'])
            self.best_model = results[best_model_name]['model']
            self.best_score = results[best_model_name]['silhouette_score']
            self.cluster_labels = results[best_model_name]['cluster_labels']
            print(f"\nBest model: {best_model_name} (Silhouette Score = {self.best_score:.4f})")
        
        return results
    
    def create_cluster_visualization(self, results):
        """Create visualizations for clustering results."""
        print("Creating cluster visualizations...")
        
        # PCA for dimensionality reduction
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X_scaled)
        
        # t-SNE for better visualization
        tsne = TSNE(n_components=2, random_state=self.config['models']['random_state'])
        X_tsne = tsne.fit_transform(self.X_scaled)
        
        # Create visualization plots
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Job Clustering Results Visualization', fontsize=16)
        
        # Plot 1: PCA visualization
        scatter1 = axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=self.cluster_labels, cmap='viridis', alpha=0.6)
        axes[0, 0].set_title('PCA Visualization of Clusters')
        axes[0, 0].set_xlabel('First Principal Component')
        axes[0, 0].set_ylabel('Second Principal Component')
        plt.colorbar(scatter1, ax=axes[0, 0])
        
        # Plot 2: t-SNE visualization
        scatter2 = axes[0, 1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=self.cluster_labels, cmap='viridis', alpha=0.6)
        axes[0, 1].set_title('t-SNE Visualization of Clusters')
        axes[0, 1].set_xlabel('t-SNE Component 1')
        axes[0, 1].set_ylabel('t-SNE Component 2')
        plt.colorbar(scatter2, ax=axes[0, 1])
        
        # Plot 3: Cluster size distribution
        cluster_counts = pd.Series(self.cluster_labels).value_counts().sort_index()
        axes[1, 0].bar(cluster_counts.index, cluster_counts.values, color='skyblue')
        axes[1, 0].set_title('Cluster Size Distribution')
        axes[1, 0].set_xlabel('Cluster ID')
        axes[1, 0].set_ylabel('Number of Jobs')
        
        # Plot 4: Model comparison
        valid_models = {k: v for k, v in results.items() if 'silhouette_score' in v}
        if valid_models:
            model_names = list(valid_models.keys())
            silhouette_scores = [valid_models[name]['silhouette_score'] for name in model_names]
            
            axes[1, 1].bar(model_names, silhouette_scores, color='lightcoral')
            axes[1, 1].set_title('Model Performance Comparison (Silhouette Score)')
            axes[1, 1].set_ylabel('Silhouette Score')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        save_plot(fig, 'clustering_visualization.png')
        plt.close()
        
        return fig
    
    def create_cluster_profiles(self, results):
        """Create detailed cluster profiles and characteristics."""
        if self.cluster_labels is None:
            raise ValueError("No clustering results available. Train models first.")
        
        print("Creating cluster profiles...")
        
        # Add cluster labels to dataframe
        df_with_clusters = self.df.copy()
        df_with_clusters['cluster'] = self.cluster_labels
        
        # Create cluster profiles
        cluster_profiles = {}
        
        for cluster_id in sorted(df_with_clusters['cluster'].unique()):
            if cluster_id == -1:  # Skip noise points
                continue
                
            cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
            
            profile = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(df_with_clusters) * 100,
                'avg_salary': cluster_data['salary_usd'].mean(),
                'median_salary': cluster_data['salary_usd'].median(),
                'avg_experience': cluster_data['years_experience'].mean(),
                'remote_ratio': cluster_data['remote_ratio'].mean(),
                'top_employment_type': cluster_data['employment_type'].mode().iloc[0] if len(cluster_data['employment_type'].mode()) > 0 else 'Unknown',
                'top_company_size': cluster_data['company_size'].mode().iloc[0] if len(cluster_data['company_size'].mode()) > 0 else 'Unknown',
                'top_education': cluster_data['education_required'].mode().iloc[0] if len(cluster_data['education_required'].mode()) > 0 else 'Unknown'
            }
            
            cluster_profiles[cluster_id] = profile
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Cluster Profiles Analysis', fontsize=16)
        
        cluster_ids = list(cluster_profiles.keys())
        
        # Plot 1: Cluster sizes
        sizes = [cluster_profiles[cid]['size'] for cid in cluster_ids]
        axes[0, 0].bar(cluster_ids, sizes, color='skyblue')
        axes[0, 0].set_title('Cluster Sizes')
        axes[0, 0].set_xlabel('Cluster ID')
        axes[0, 0].set_ylabel('Number of Jobs')
        
        # Plot 2: Average salaries
        avg_salaries = [cluster_profiles[cid]['avg_salary'] for cid in cluster_ids]
        axes[0, 1].bar(cluster_ids, avg_salaries, color='lightcoral')
        axes[0, 1].set_title('Average Salary by Cluster')
        axes[0, 1].set_xlabel('Cluster ID')
        axes[0, 1].set_ylabel('Average Salary (USD)')
        
        # Plot 3: Average experience
        avg_experience = [cluster_profiles[cid]['avg_experience'] for cid in cluster_ids]
        axes[1, 0].bar(cluster_ids, avg_experience, color='lightgreen')
        axes[1, 0].set_title('Average Years of Experience by Cluster')
        axes[1, 0].set_xlabel('Cluster ID')
        axes[1, 0].set_ylabel('Average Years of Experience')
        
        # Plot 4: Remote ratio
        remote_ratios = [cluster_profiles[cid]['remote_ratio'] for cid in cluster_ids]
        axes[1, 1].bar(cluster_ids, remote_ratios, color='gold')
        axes[1, 1].set_title('Average Remote Ratio by Cluster')
        axes[1, 1].set_xlabel('Cluster ID')
        axes[1, 1].set_ylabel('Remote Ratio (%)')
        
        plt.tight_layout()
        save_plot(fig, 'cluster_profiles.png')
        plt.close()
        
        return cluster_profiles, fig
    
    def save_best_model(self):
        """Save the best performing clustering model."""
        if self.best_model is None:
            raise ValueError("No best model available. Train models first.")
        
        save_model(self.best_model, 'job_clusterer.joblib')
        return 'job_clusterer.joblib'
    
    def generate_clustering_report(self, results, cluster_profiles):
        """Generate a comprehensive clustering report."""
        if self.best_model is None:
            raise ValueError("No best model available. Train models first.")
        
        report = []
        report.append("AI Job Market - Job Clustering Analysis Report")
        report.append("=" * 60)
        report.append("")
        
        report.append("1. CLUSTERING APPROACH")
        report.append("-" * 20)
        report.append("• Feature engineering: Label encoding for categorical variables")
        report.append("• Feature scaling: StandardScaler for numerical features")
        report.append("• Model selection: Multiple clustering algorithms")
        report.append("• Evaluation: Silhouette score, Calinski-Harabasz index, Davies-Bouldin index")
        report.append("")
        
        report.append("2. MODEL PERFORMANCE COMPARISON")
        report.append("-" * 30)
        
        valid_models = {k: v for k, v in results.items() if 'silhouette_score' in v}
        for name, metrics in valid_models.items():
            report.append(f"\n{name}:")
            report.append(f"  Silhouette Score: {metrics['silhouette_score']:.4f}")
            report.append(f"  Calinski-Harabasz Score: {metrics['calinski_score']:.4f}")
            report.append(f"  Davies-Bouldin Score: {metrics['davies_score']:.4f}")
            report.append(f"  Number of clusters: {metrics['n_clusters']}")
        
        report.append("")
        report.append("3. BEST MODEL")
        report.append("-" * 12)
        if valid_models:
            best_model_name = max(valid_models.keys(), key=lambda x: valid_models[x]['silhouette_score'])
            report.append(f"Model: {best_model_name}")
            report.append(f"Silhouette Score: {valid_models[best_model_name]['silhouette_score']:.4f}")
            report.append(f"Number of clusters: {valid_models[best_model_name]['n_clusters']}")
        
        report.append("")
        report.append("4. CLUSTER PROFILES")
        report.append("-" * 18)
        
        for cluster_id, profile in cluster_profiles.items():
            report.append(f"\nCluster {cluster_id}:")
            report.append(f"  Size: {profile['size']} jobs ({profile['percentage']:.1f}%)")
            report.append(f"  Average Salary: ${profile['avg_salary']:,.0f}")
            report.append(f"  Median Salary: ${profile['median_salary']:,.0f}")
            report.append(f"  Average Experience: {profile['avg_experience']:.1f} years")
            report.append(f"  Remote Ratio: {profile['remote_ratio']:.1f}%")
            report.append(f"  Top Employment Type: {profile['top_employment_type']}")
            report.append(f"  Top Company Size: {profile['top_company_size']}")
            report.append(f"  Top Education: {profile['top_education']}")
        
        report.append("")
        report.append("5. CLUSTER INTERPRETATION")
        report.append("-" * 25)
        report.append("• Clusters represent different job market segments")
        report.append("• Each cluster has distinct characteristics in terms of salary, experience, and work arrangements")
        report.append("• Understanding cluster profiles helps in targeted job search and recruitment")
        report.append("")
        
        report.append("6. RECOMMENDATIONS")
        report.append("-" * 15)
        report.append("• Use cluster analysis for market segmentation")
        report.append("• Apply insights for targeted recruitment strategies")
        report.append("• Consider cluster characteristics for salary benchmarking")
        report.append("• Regular re-clustering with new data")
        
        report_text = "\n".join(report)
        save_report(report_text, 'clustering_analysis_report.txt')
        
        return report_text

def run_job_clustering(input_file=None, n_clusters=None):
    """Convenience function to run complete job clustering analysis."""
    clusterer = JobClusterer()
    clusterer.load_data(input_file)
    clusterer.prepare_features()
    optimal_clusters = clusterer.find_optimal_clusters()
    results = clusterer.train_clustering_models(n_clusters)
    clusterer.create_cluster_visualization(results)
    cluster_profiles, _ = clusterer.create_cluster_profiles(results)
    clusterer.save_best_model()
    report = clusterer.generate_clustering_report(results, cluster_profiles)
    print("Job clustering analysis completed. Check outputs/plots/, outputs/models/, and outputs/reports/ for results.")
    return clusterer

if __name__ == "__main__":
    run_job_clustering() 