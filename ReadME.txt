*** Structure

** Core/Motors/Algorithms/Models
	* Experiment (main_experiments.py): The main class that combines all other modules, classes, and functions to build an experiment.
	* DataDivisor (DataDivisor.py): The data handler class. It reads the data directory input as specified in constants.py, and outputs
	* FeatureSelector (FeatureSelection.py): Based upon recursive feature elimination using cross-validation algorithm. This class initialize and train various classifiers that supports feature_importance_ or feature_coef_
	* CustomClassifier (CLassifiers.py): The machine learning classifiers bag that initialize and train different machine learning classifiers
** Configuration/Variablenames/Directories
	* constansts.py: It contains all the keywords used to refer to each attribute in the behavioral report.
			 Contains the keywords used to refer to available classifiers for both ML and feature selection.
			 Contains all directories utilized in saving

** Utility functions
	* utils.py: Contains the functions used to give a time stamp for each result folder created. COntain the functions that is used for saving different FS and ML models.

** Starting point:
	* main.py: It contains only 1 line of code and a dictionary contains all the required experiment attributes.
		   Experimental attributes can be found in experiment_designer.py.


