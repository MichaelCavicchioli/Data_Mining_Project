from .utilities import split_data_in_train_and_test, get_best_model_with_grid_search, plot_decision_tree, plot_confusion_matrix_and_get_accuracy
from .mall_customers_dataset_controller import MallCustomersDatasetController
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn import tree

class DecisionTree:
    
    def best_decision_tree(self):

        best_accuracy = 0
        best_split = 0
        for i in range(2, 5):

            # Ogni volta si ha il dataset di partenza
            mdc = MallCustomersDatasetController()
            mdc.label_encoding()

            # Split dei dati 
            X_train, X_test, y_train, y_test = split_data_in_train_and_test(i, mdc.get_df())

            # Crea il modello DecisionTreeClassifier
            dtc = tree.DecisionTreeClassifier(random_state=93)

            # Griglia di parametri da testare usando la CV
            param_grid = {
                'criterion': ['gini', 'entropy'],
                'max_depth': [None, 3, 5, 7, 10]
            }

            # Ottiene il miglior modello trovato tramite Grid Search con CV
            best_model = get_best_model_with_grid_search(
                dtc, 
                param_grid, 
                5, 
                X_train, 
                y_train
            )

            # Plot dell'albero
            plot_decision_tree(i, best_model, X_train)

            # Predizioni
            y_pred = best_model.predict(X_test)

            # Plot della matrice di confusione
            accuracy = plot_confusion_matrix_and_get_accuracy(y_test, y_pred)
            print(f'Accuracy: {accuracy:.2f}')
            
            # Calcola le metriche
            self._metrics(y_test, y_pred)
            
            # Aggiorna il miglior split
            if best_accuracy < accuracy:
                best_accuracy = accuracy
                best_split = i

        print(f'Best split is: {best_split} \n')


    # Calcolo delle metriche: Precision, Recall e F-Measure
    def _metrics(self, y_test, y_pred) -> None:
        precision = precision_score(y_true=y_test, y_pred=y_pred, average='macro')
        recall = recall_score(y_true=y_test, y_pred=y_pred, average='macro')
        f1 = f1_score(y_true=y_test, y_pred=y_pred, average='macro')

        # Stampa i risultati
        print(f'Precision: {precision:.2f}')
        print(f'Recall: {recall:.2f}')
        print(f'F1 Score: {f1:.2f}\n')