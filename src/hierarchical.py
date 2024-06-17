from .mall_customers_dataset_controller import MallCustomersDatasetController
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import fcluster
from sklearn.preprocessing import LabelEncoder
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt

class Hierarchical:

    def clustering(self, mdc: MallCustomersDatasetController, method: str) -> None:

        # Esegue il linkage dei dati
        clustering = hierarchy.linkage(mdc.get_df(), method, optimal_ordering = True)

        # Creazione del dendogramma
        hierarchy.dendrogram(clustering, labels = mdc.get_df_ids(), leaf_rotation=70, leaf_font_size=8)

        # Plotta il dendogramma
        plt.show()

        self._save_clustering_summaries(clustering, method)
        

    def _save_clustering_summaries(self, clustering: hierarchy.linkage, method: str) -> None:
        
        # Taglia il dendrogramma per ottenere esattamente 6 cluster
        cluster_labels = fcluster(clustering, t=6, criterion='maxclust')

        # Aggiunge le etichette dei cluster al DataFrame originale
        df = MallCustomersDatasetController().get_df()
        df['Cluster'] = cluster_labels
        
        # Crea una copia del DataFrame originale per il calcolo del Silhouette Score
        df_for_silhouette = df.copy()

        # Applica il LabelEncoder solo alle colonne di tipo 'object'
        for col in df_for_silhouette.columns:
            if df_for_silhouette[col].dtype == 'object':
                le = LabelEncoder()
                df_for_silhouette[col] = le.fit_transform(df_for_silhouette[col])

        # Aggiunge le etichette dei cluster al DataFrame
        df_for_silhouette['Cluster'] = cluster_labels

        # Calcola il Silhouette Score utilizzando il DataFrame con l'encoding
        silhouette_avg = silhouette_score(df_for_silhouette.drop(columns=['Cluster']), cluster_labels)
        print(f'Silhouette Score for {method} method: {silhouette_avg}')


        # Salva i cluster in un file CSV
        df.to_csv(f'hierarchical_clustering_summaries/{method}_clustering.csv', index=False)