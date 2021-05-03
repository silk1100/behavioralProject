import DataDivisor
import FeatureSelection
import Classifers
import constants

class Experiment:
    def __int__(self):
        # Type of data to use
        self.data_repr = None

        # Data divisor parameters
        self.srs_type = None
        self.severity_group = None
        self.age_group = None
        self.gender = None
        self.divide_data = False

        # Featureselection parameters
        self.est = None
        self.cv = None
        self.scoring = None
        self.n_jobs = None
        self.verbose = None
        self.step = None
        self.min_feat_to_select = None


    def run(self):
        pass

    def save_results(self):
        pass

    def visualize_results(self):
        pass

if __name__ == "__main__":
    experiment_1 = Experiment()
    experiment_1.run()