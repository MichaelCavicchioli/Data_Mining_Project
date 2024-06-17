from sklearn.metrics import silhouette_score
from .utilities import create_plot
import sklearn.cluster as skc
import pandas as pd
import numpy as np

class KMeans:
    
    # Algoritmo K-Means per trovare il miglior K
    def best_kmeans(self, df: pd.DataFrame, k_values: range) -> None:        
        sse_values = []
        silhouette_scores = []

        for k in k_values:
            kmeans = skc.KMeans(n_clusters=k, random_state=93, init='k-means++')
            kmeans.fit(df)
            print(f'K-Means clustering applied with {k} clusters.')
        
            sse = kmeans.inertia_  # SSE        
            sse_values.append(sse)

            score = silhouette_score(df, kmeans.labels_)
            silhouette_scores.append(score)

        create_plot(k_values, sse_values, 'Elbow method for optimal k', 'SSE')
        create_plot(k_values, silhouette_scores, 'Silhouette scores for optimal k', 'Silhouette scores')

    # Algoritmo K-Means
    def kmeans(self, df: pd.DataFrame) -> None:
        self.best_centroids, self.best_labels = self._kmeans_multiple_times(df, 6, 50)

        # Creazione del nuovo dataframe con le etichette dei cluster
        cols = df.columns.tolist()
        cols.append('Cluster ID')
        self.df_with_cluster_id = pd.DataFrame(
            np.concatenate((df, self.best_labels.reshape(-1, 1)), axis=1), 
            columns=cols
        )
        print('*** Dataframe with Cluster ID: ***\n', self.df_with_cluster_id, '\n')

        self.best_centroids = pd.DataFrame(self.best_centroids, columns=df.columns)
        self.best_centroids.to_csv('kmeans_dataset_clusters/centroids.csv', sep=',', mode='w', index=False)
        print('*** Centroids are: ***\n', self.best_centroids, '\n')

    # Restituisce i migliori centroidi
    def get_best_centroids(self) -> pd.DataFrame:
        return self.best_centroids

    # Restituisce le migliori etichette
    def get_best_labels(self) -> list:
        return self.best_labels
    
    # Restituisce il dataframe con la colonna Cluster ID
    def get_df_with_cluster_id(self) -> pd.DataFrame:
        return self.df_with_cluster_id

    # Trova i migliori centroidi ed etichette eseguendo l'algoritmo KMeans n_times
    def _kmeans_multiple_times(self, df: pd.DataFrame, n_clusters: int, n_times: int) -> list:
        best_see = np.inf
        best_centroids = None
        
        while n_times > 0:
            kmeans = skc.KMeans(n_clusters=n_clusters, random_state=93, init='k-means++')
            kmeans.fit(df)
            sse = kmeans.inertia_
            
            if sse < best_see:
                best_see = sse
                best_centroids = kmeans.cluster_centers_
                best_labels = kmeans.labels_

            n_times -= 1

        return best_centroids, best_labels