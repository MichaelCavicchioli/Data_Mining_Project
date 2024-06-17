from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Creazione delle directories dove salvare i dati
def create_directories_to_store_data() -> None:
    if not (
        os.path.exists('assets/images') and 
        os.path.exists('kmeans_clustering_summaries') and 
        os.path.exists('kmeans_dataset_clusters') and 
        os.path.exists('hierarchical_clustering_summaries')
    ):
        os.makedirs('assets/images')
        os.makedirs('kmeans_clustering_summaries')
        os.makedirs('kmeans_dataset_clusters')
        os.makedirs('hierarchical_clustering_summaries')

# Creazioni di tutti i possibili plot, combinando 1 ad 1 ogni attributo del dataset
def save_all_attributes_combination(df: pd.DataFrame) -> None:

    # Cicla su ogni colonna per creare una figura con scatterplot tra essa e tutte le altre
    for i in range(len(df.columns)):
        # Colonna corrente
        current_col = df.columns[i]

        # Crea una figura con 2 righe e 2 colonne (4 subplot)
        fig, axs = plt.subplots(2, 2, figsize=(15, 6))

        # Appiattisce la griglia di subplot per iterare facilmente
        axs = axs.flatten()

        # Aggiunge uno scatterplot per ogni altra colonna
        for j, ax in enumerate(axs):
            alt_col = df.columns[j]
            sns.scatterplot(x=current_col, y=alt_col, data=df, ax=ax)
            ax.set_title(f'{current_col} vs {alt_col}')

        # Regola il layout per evitare sovrapposizioni
        plt.tight_layout()

        # Salva il plot
        plt.savefig(f'assets/images/{current_col}.png')

        # Chiude la figura per risparmiare memoria
        plt.close(fig)
        
# Crea un plot con i valori passati come parametri, per vari valori di k
def create_plot(k_values: range, y_values: list, title: str, ylabel: str) -> None:
    plt.figure() 
    plt.plot(k_values, y_values, marker='o')
    plt.title(title)
    plt.xlabel('Number of clusters (k)')
    plt.ylabel(ylabel)
    plt.show()

# Plotta i cluster ed i centroidi ottenuti con l'algoritmo K-Means su due attributi
def clusters_centroids_plot(df: pd.DataFrame, best_centroids: pd.DataFrame, best_labels: list, attributes: list) -> None:
    # Mappa di colori per ogni cluster
    colors = ['r', 'g', 'c', 'blue', 'purple', 'orange']

    # Crea un grafico 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Visualizza ogni cluster con un colore diverso
    for i in range(6):
        points = df[best_labels == i]  # Seleziona i punti che appartengono a questo cluster
        ax.scatter(
            points.loc[:, attributes[0]], 
            points.loc[:, attributes[1]], 
            points.loc[:, attributes[2]], 
            color=colors[i], 
            label=f'Cluster {i + 1}'
        )

    # Visualizza i centroidi
    ax.scatter(
        best_centroids.loc[:, attributes[0]], 
        best_centroids.loc[:, attributes[1]], 
        best_centroids.loc[:, attributes[2]], 
        color='black', 
        s=800, 
        marker='*', 
        label='Centroidi'
    )

    # Aggiungi etichette agli assi
    ax.set_xlabel(attributes[0])
    ax.set_ylabel(attributes[1])
    ax.set_zlabel(attributes[2])

    # Aggiungi una legenda
    ax.legend()

    # Mostra il grafico
    plt.show()

# Creazione di K file csv, ognuno contenente gli elementi inerenti al k-esimo cluster
def save_clusters_as_csv(df_with_cluster_id: pd.DataFrame) -> None:
    for i in range(6):
        c_df = df_with_cluster_id.loc[df_with_cluster_id['Cluster ID'] == i]
        df_without_cluster_id = c_df.loc[:, c_df.columns != 'Cluster ID']

        min_max_mean_df = {
            'Categories' : df_without_cluster_id.columns,
            'Min': df_without_cluster_id.min().round(2),
            'Max': df_without_cluster_id.max().round(2),
            'Mean': df_without_cluster_id.mean().round(2)
        }

        pd.DataFrame(min_max_mean_df).to_csv(f'kmeans_clustering_summaries/cluster_{i}_summ.csv', sep=',', mode='w', index=False)
        c_df.to_csv(f'kmeans_dataset_clusters/dataset_cluster_{i}.csv', sep=',', mode='w', index=False)

# Splitta i dati in Train e Test
def split_data_in_train_and_test(i: int, df: pd.DataFrame) -> list:

    # Creazione dei range per la variabile obiettivo
    class_lim = ((100//i) + 1)
    df['Spending Score'] = df['Spending Score'] // class_lim

    # Suddivisione dei dati
    y = df.loc[:, 'Spending Score']
    X = df.drop(['Spending Score'], axis=1)

    return train_test_split(
        X, 
        y, 
        stratify=y,
        test_size=0.2,
        shuffle=True,
        random_state=93
    )

# Restituisce il miglior modello tramtie Grid Search usando la CV
def get_best_model_with_grid_search(dtc: DecisionTreeClassifier, param_grid: dict, cv: int, X_train: pd.DataFrame, y_train: pd.DataFrame) -> GridSearchCV:
    
    # Crea la GridSearchCV
    grid_search = GridSearchCV(estimator=dtc, param_grid=param_grid, cv=cv, scoring='accuracy')

    # Allena il modello con i dati di addestramento
    grid_search.fit(X_train, y_train)

    # Stampa i migliori parametri trovati e il miglior punteggio
    print(f'Best parameters found: {grid_search.best_params_}')
    print(f'Best cross-validation score: {grid_search.best_score_}')

    # Utilizza i migliori parametri per creare un nuovo modello
    return grid_search.best_estimator_

# Plot dell'albero
def plot_decision_tree(i: int, best_model: GridSearchCV, X_train: pd.DataFrame) -> None:
    class_names = [str(name) for name in range(0, i)]

    plt.figure(figsize=(20,10))
    plot_tree(best_model, feature_names=X_train.columns, class_names=class_names, filled=True)
    plt.show()

# Plotta la Matrice di confusione e restituisce l'accuratezza
def plot_confusion_matrix_and_get_accuracy(y_test, y_pred) -> float:
    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
    
    ax = plt.subplots(figsize=(10, 7))[1]
    sns.heatmap(conf_matrix, annot=True, linewidths=0.01, cmap='Greens', linecolor='gray', fmt='.1f', ax=ax)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    return np.trace(conf_matrix) / np.sum(conf_matrix)