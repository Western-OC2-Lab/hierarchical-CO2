from train_evaluate_models import *
from tensorflow.keras.models import load_model
import pickle

class TransferLearning(Train_Evaluate_Models):

    def __init__(self, h_f, root_dir, source_room, root_folder, runs, destination_room):
        super(TransferLearning, self).__init__(h_f, root_dir, source_room, root_folder, runs)
        self.destination_room = destination_room
        self.set_hfs = ["h-5_f-5", "h-10_f-10", "h-15_f-15", "h-20_f-20"]

        self.testing_dir = f"{self.root_dir}/data/generated/{self.destination_room}_test"

        eval_dt_test = EvaluationDt(self.testing_dir)
        self.all_dfs, _ = eval_dt_test.get_model_sets()

    def create_insights_folders(self, root_folder, room_name, h_f, method, type_op='test'):
        FIGURES_DIR =  f"{root_folder}/reports/figures/{room_name}/{type_op}/TL_{h_f}/"
        STATS_DIR = f"{root_folder}/reports/results/{room_name}/{type_op}/TL_{h_f}/"
        MODELS_DIR = f"{root_folder}/models/{self.room_name}/{h_f}/{method}"

        os.makedirs(FIGURES_DIR, exist_ok=True)
        os.makedirs(STATS_DIR, exist_ok=True)

        self.FIGURES_DIR = FIGURES_DIR
        self.STATS_DIR = STATS_DIR
        self.MODELS_DIR = MODELS_DIR

    def stacked_dataset(self, set_models, inputX):
        """
            This method produces the outputs of each individual model
        """
        stackX = None
        for idx, model in enumerate(set_models):
            data_input = inputX[idx]
            yhat = model.predict(data_input)
            yhat = yhat[:, 0] #Probability that the co2_var will decrease
            if stackX is None:
                stackX = yhat
            else:
                stackX = np.column_stack([stackX, yhat])

        return stackX

    def ensemble_prediction(self, set_models, model, generators, labels, aux_generators):
        stackedX = self.stacked_dataset(set_models, generators)
        aux_variables = aux_generators[0]
        for i in range(1, len(aux_generators)):
            aux_variables = np.column_stack([aux_variables, aux_generators[i]])

        
        stackedX = np.column_stack([stackedX, aux_variables])

        predictions = model.predict(stackedX)

        return labels, predictions

    def evaluate(self):
        results = pd.DataFrame()

        for h_f in self.set_hfs:
            testing_df = self.all_dfs[h_f]
            testing_df = self.change_df(testing_df)
            self.create_insights_folders(self.root_dir, self.destination_room,h_f, method = 'direction_var_transformation')
            encoder = LabelEncoder()
            encoder.fit(testing_df.direction.values)
            dict_results = {}
            dict_results['h_f'] = h_f
            set_models = []
            testing_generators, testing_generator_predictions, testing_aux = [], None, []

            for feature in self.set_features:
                print(f'Currently at {h_f} and {feature}')

                model = load_model(f"{self.MODELS_DIR}/{feature}.h5")
                set_models.append(model)
                X_test, y_test = testing_df.loc[:, feature].values, testing_df.loc[:, 'direction'].values
                y_test = encoding_retrieval(encoder, y_test)
                feature_aux_testing = testing_df.loc[:, "{}_h_pp".format(feature)].values
                testing_generator = retrieve_combined_generator(X_test, y_test)
                
                testing_p = retrieve_single_generator(testing_df.loc[:, 'co2_var'].values)
                testing_generator_predictions = retrieve_labels(testing_p)
                testing_generators.append(testing_generator)
                testing_aux.append(feature_aux_testing)
                dict_results[f'acc_{feature}'] = get_predictions(model, testing_generator, accuracy_score)

            ensemble_model = pickle.load(open(f"{self.MODELS_DIR}/dt.sav", 'rb'))

            start_time = time.time()
            labels, predictions = self.ensemble_prediction(set_models, ensemble_model, testing_generators, testing_generator_predictions, testing_aux)
            end_time = time.time()

            total_testing_time = round((end_time - start_time) / 60, 2)
            dict_results['test_time'] = total_testing_time
            dict_results['test_time_instance'] = total_testing_time / len(testing_generator_predictions)

            for threshold in [None, 5, 10, 20, 40, 50, 75, 100]:
                figures_filename = "{}/TestingEnsemble-H{}-T{}.png".format(self.FIGURES_DIR, h_f, str(threshold))
                dict_results['MAE_test-{}'.format(str(threshold))] = defining_plotting_thresholds(predictions, labels, figures_filename, threshold_values = threshold)
                print('Ensemble Test-Threshold -{}: {}'.format(str(threshold), str(dict_results['MAE_test-{}'.format(str(threshold))])))

            results = pd.concat([results, pd.DataFrame.from_dict([dict_results])])
            results.to_csv("{}/H{}.csv".format(self.STATS_DIR, h_f))
