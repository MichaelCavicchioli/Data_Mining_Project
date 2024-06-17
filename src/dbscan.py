from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class DBscan:

    def dbscan(self, df: pd.DataFrame) -> None:
        
        # Trova min samples e epsilon con il più alto silhouette score
        min_samples = range(3, 9)
        eps_samples = np.arange(10, 50, 1)

        output = []
        for ms in min_samples:
            for eps in eps_samples:
                labels = DBSCAN(min_samples=ms, eps = eps).fit(df).labels_
                try:
                    score = silhouette_score(df, labels)
                    output.append((ms, eps, score))
                except:
                    pass
                
        # Prende i valori con il silhouette score migliore
        min_samples, eps, score = sorted(output, key=lambda x:x[2], reverse=True)[0]
        print(f'Best silhouette_score: {score}')
        print(f'min_samples: {min_samples}')
        print(f'eps: {eps}')

        # Trova il miglior epsilon
        k = 3
        nn = NearestNeighbors(n_neighbors=k, algorithm='brute').fit(df)
        distances = nn.kneighbors(df)[0]

        distances = distances[:,-1]
        distances = np.sort(distances)
        self._plot_distances(distances)

        # Prova con i parametri calcolati
        eps = 14
        ms = 3
        db = DBSCAN(eps=eps, min_samples=ms).fit(df)
        self._dbscan_plot(db, df, ['Age', 'Annual Income', 'Spending Score'])

        # Prova con un set di parametri differenti
        eps = 14
        ms = 7
        db = DBSCAN(eps=eps, min_samples=ms).fit(df)
        self._dbscan_plot(db, df, ['Age', 'Annual Income', 'Spending Score'])
        
        # Let's see the silhouette score
        score = silhouette_score(df, db.labels_)
        print(f'silhouette_score: {score}\n')

    # Crea i plot per le distanze
    def _plot_distances(self, distances) -> None:
        plt.figure(figsize=(10,8))
        plt.xlabel('Points')
        plt.ylabel('Distance') 
        plt.plot(distances)
        plt.show()

    # Crea i plot per DBscan
    def _dbscan_plot(self, db: DBSCAN, df: pd.DataFrame, columns: list) -> None:
        labels = pd.DataFrame(db.labels_, columns=['Cluster ID'])
        result = pd.concat((df, labels), axis=1)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(result[columns[0]], result[columns[1]], result[columns[2]], c=result['Cluster ID'], cmap='tab20b')
        legend = ax.legend(*scatter.legend_elements(), title='Clusters')
        ax.add_artist(legend)
        
        plt.show()