""" """
from sklearn.model_selection import train_test_split

from memobuilder import mtrainer as memotrainer
from memobuilder.mdb import memomodel


class SurrogateModelTrainer(object):
    def __init__(self, test_split_ratio=0.2, num_cv_folds=5):
        self.name = None
        self.metamodel_trainer = Trainer(test_split_ratio, num_cv_folds)
        self.metamodels = []

    def fit(self, input_response_dataset):
        training_results = []
        for metamodel in self.metamodels:
            training_results.append(
                self.metamodel_trainer.fit(metamodel, input_response_dataset))

        result = memomodel.SurrogateModelTrainingResult()
        result.training_results = training_results
        result.surrogate_model_name = self.name
        return result

    def __repr__(self):
        return str(self.__dict__)


class Trainer(object):

    def __init__(self, test_split_ratio=0.2, num_cv_folds=5):
        self.test_split_ratio = test_split_ratio
        self.num_cross_val_folds = num_cv_folds

    def fit(self, metamodel, input_response_data):
        # select input dataset for this metamodel
        selected_data = input_response_data.select(
            selected_inputs=metamodel.input_names,
            selected_responses=metamodel.response_names)
        # split train and test data
        train_data, test_data = self.train_test_split(selected_data)

        # fit the metamodel to the training data
        metamodel.fit(train_data.inputs, train_data.responses)

        # compute different scores of the metamodel on the test data
        score_r2 = memotrainer.scores.r2_score(metamodel, test_data)
        score_mae = memotrainer.scores.mae_score(metamodel, test_data)
        score_hae = memotrainer.scores.hae_score(metamodel, test_data)
        score_mse = memotrainer.scores.mse_score(metamodel, test_data)

        # compose a result object and return it
        result = memomodel.TrainingResult()
        result.train_data = train_data
        result.test_data = test_data
        result.metamodel = metamodel

        result.score_r2 = score_r2
        result.score_mae = score_mae
        result.score_hae = score_hae
        result.score_mse = score_mse

        return result

    def train_test_split(self, input_response_data):
        split = train_test_split(
            input_response_data.inputs,
            input_response_data.responses,
            test_size=self.test_split_ratio, random_state=0)

        train_data = memomodel.InputResponseDataset()
        train_data.inputs = split[0]
        train_data.responses = split[2]

        test_data = memomodel.InputResponseDataset()
        test_data.inputs = split[1]
        test_data.responses = split[3]
        return train_data, test_data

    def __repr__(self):
        return '%s [%s]' % (self.__class__.__name__, str(self.__dict__))
