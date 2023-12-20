def cross_validate_clf(X, y, classifiers, groups):
	'''Perform cross-validation for multiple classifiers on the input data.

    Args:
        X (array-like): Input feature matrix or dataset.
        y (array-like): Target variable or labels.
        classifiers (list): List of classifier objects to be evaluated.
        groups (array-like): Groups or categories for stratified grouping in cross-validation.

    Returns:
        dict: Dictionary containing evaluation results for each classifier, with metrics such as Accuracy, Sensitivity,
              Specificity, Precision, and ROC AUC.
	'''

	# Scores for evaluation
	scores ={'accuracy': make_scorer(accuracy_score), 
	  'sensitivity': make_scorer(recall_score), 
	  'specificity': make_scorer(recall_score, pos_label=0), 
	  'precision': make_scorer(precision_score), 
	  'roc_auc': make_scorer(roc_auc_score, needs_proba=True)}

	num_folds = 5
	cross_val = StratifiedGroupKFold(n_splits= num_folds)	

	evaluation_results = {}
	for classifier in classifiers:
		cv_results = cross_validate(classifier, X, y, scoring=scores, cv=cross_val, groups = groups)
		
		if type(classifier).__name__ == "KNeighborsClassifier":
			classifier_name = type(classifier).__name__
			params_dict = classifier.get_params()
			n_neigbors = params_dict["n_neighbors"]
			classifier_name = f"{classifier_name} with n_neighbors={n_neigbors}"
		else:
			classifier_name = type(classifier).__name__

		evaluation_results[classifier_name] = {
            'Accuracy': cv_results['test_accuracy'].mean(),
            'Sensitivity': cv_results['test_sensitivity'].mean(),
            'Specificity': cv_results['test_specificity'].mean(),
            'Precision': cv_results['test_precision'].mean(),
            'ROC AUC': cv_results['test_roc_auc'].mean()

        }

	return evaluation_results