from src import *
import warnings

def main():
    
    # Caricamento dataset
    mdc = MallCustomersDatasetController()

    # Dataset info
    mdc.print_column_info()

    ### Pre processing ###
    mdc.na()
    mdc.remove_duplicated()
    mdc.label_encoding()
    ######################

    # Creazione directories 
    utilities.create_directories_to_store_data()

    ### Plots ###
    utilities.save_all_attributes_combination(mdc.get_df())
    #############

    ### Kmeans ###
    kmn = KMeans()
    kmn.best_kmeans(
        mdc.get_df(), 
        range(2, 11)
        )

    kmn.kmeans(mdc.get_df())
    ##############

    ### Plot e salvataggio informazioni ###    
    utilities.clusters_centroids_plot(
        mdc.get_df(), 
        kmn.get_best_centroids(), 
        kmn.get_best_labels(), 
        ['Age', 'Annual Income', 'Spending Score']
        )
    
    utilities.save_clusters_as_csv(kmn.get_df_with_cluster_id())
    ################################   
    
    ### DBscan ###
    dbs = DBscan()
    dbs.dbscan(mdc.get_df())
    ##############

    ### Clustering gerarchico ###
    hier = Hierarchical()
    hier.clustering(mdc, 'single')
    hier.clustering(mdc, 'complete')
    hier.clustering(mdc, 'average')
    ####################

    ### Albero di decisione ###
    dec_tree = DecisionTree()
    dec_tree.best_decision_tree()
    #####################


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()