#!/usr/bin/env python3



# THIS PROGRAM WAS DERIVED FROM THE FASHION(MINE) ASSIGNMENT WITH ENSEMBLES
# THE CV SCORE IS USUALLY .93, BUT ALL LABELS MATCH 100%

import math
import pandas as pd
import numpy as np
import sklearn.model_selection
import sklearn.metrics
import sklearn.tree
import sklearn.svm
import sys
import os.path
import getopt

import process_data

def calculateCorrectLabels( l, r ):
    correct = 0.0
    if ( len(l) != len(r) ):
        return 0.0
    for i in range( len(l) ):
        if l[i] == r[i]:
            correct += 1
    return correct / len(l)

def make_decision_tree_params():
    params = process_data.make_predictor_params()
    tree_params = {
        # "model__max_depth": [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, None ],
        # "model__min_samples_split": [ 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.05, 0.10, 0.20 ],
        # "model__criterion": [ "gini", "entropy" ],
        # "model__splitter": [ "best", "random" ],
        # "model__min_samples_leaf": [ 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.05, 0.10, 0.20 ],
        # "model__max_leaf_nodes": [ 2, 4, 8, 16, None ],
        # "model__min_impurity_decrease": [ 0.000, 0.001, 0.002, 0.005, 0.010, 0.10 ],

        "model__max_depth": [ None ],
        "model__min_samples_split": [ 0.002 ],
        "model__criterion": [ "gini" ],
        "model__splitter": [ "best" ],
        "model__min_samples_leaf": [ 0.001 ],
        "model__max_leaf_nodes": [ None ],
        "model__min_impurity_decrease": [ 0.000 ],
    }
    params.update( tree_params )
    return params

def make_decision_tree_fit_pipeline():
    items = []
    items.append(("features", process_data.make_predictor_pipeline(do_one_hot=True)))
    items.append(("model", sklearn.tree.DecisionTreeClassifier()))
    return sklearn.pipeline.Pipeline(items)

def make_svm_params():
    params = process_data.make_predictor_params()
    svm_params = {
        "model__kernel": [ "linear", "poly", "rbf", "sigmoid" ],
        "model__degree": [ 1, 2, 3, 4, 5 ],
        "model__gamma": [ "auto", "scale" ],
        "model__coef0": [ -0.1, 0.0, 0.1 ],
    }
    params.update( svm_params )
    return params

def make_svm_fit_pipeline():
    items = []
    items.append(("features", process_data.make_predictor_pipeline(do_one_hot=True)))
    items.append(("model", sklearn.svm.SVC()))
    return sklearn.pipeline.Pipeline(items)


def usage(short_opts, long_opts):
    print("usage: needs to be written")
    print(long_opts)
    return

def process_args(argv):
    # print("argv:", argv)
    allowed_model_types = ('tree','svm')
    allowed_splitter_types = ('k-fold','stratified')
    allowed_search_types = ('grid','random')
    my_args = {
        "DataFileName": "a.csv",
        "ModelType": "tree",
        "SplitterType": "k-fold",
        "SearchType": "grid",
        "Folds": 5,
        "Iterations": 100,
    }
    #"var-count=", v:
    long_opts  = [ "help", "file-name=",
                   "model=", "splitter=", "search=",
                   "folds=", "iterations=",
    ]
    short_opts = "hf:m:s:S:F:i:"
    try:
        opts, args = getopt.getopt(argv[1:], short_opts, long_opts)
    except getopt.GetoptError as e:
        print(e)
        usage(short_opts, long_opts)
        sys.exit(1)

    for o,a in opts:
        if o in ("-f", "--file-name"):
            my_args['DataFileName'] = a
        elif o in ("-m", "--model"):
            my_args['ModelType'] = a
        elif o in ("-s", "--splitter"):
            my_args['SplitterType'] = a
        elif o in ("-S", "--search"):
            my_args['SearchType'] = a
        elif o in ("-F", "--folds"):
            my_args['Folds'] = int(a)
        elif o in ("-i", "--iterations"):
            my_args['Iterations'] = int(a)
        elif o in ("-h", "--help"):
            usage(short_opts, long_opts)
            sys.exit(1)
        else:
            usage(short_opts, long_opts)
            sys.exit(1)

    # enforce parameter restrictions
    if not os.path.exists(my_args['DataFileName']):
        print()
        print("--file-name must be an existing file" )
        print(my_args['DataFileName'], "does not exist.")
        print()
        usage(short_opts, long_opts)
        sys.exit(1)
    if my_args['ModelType'] not in allowed_model_types:
        print()
        print("--model must be one of: ", " ".join(allowed_model_types))
        print()
        usage(short_opts, long_opts)
        sys.exit(1)
    if my_args['SplitterType'] not in allowed_splitter_types:
        print()
        print("--splitter must be one of: ", " ".join(allowed_splitter_types))
        print()
        usage(short_opts, long_opts)
        sys.exit(1)
    if my_args['SearchType'] not in allowed_search_types:
        print()
        print("--search must be one of: ", " ".join(allowed_search_types))
        print()
        usage(short_opts, long_opts)
        sys.exit(1)
        
    return my_args


def main( argv ):
    my_args = process_args(argv)
    #my_args is a dict containing all opts mapped to their args
    basename, ext = my_args['DataFileName'].split('.')
    data = process_data.get_data(my_args['DataFileName'], ext)
    train_data, test_data = sklearn.model_selection.train_test_split( data, test_size=.20 )

    # search for good fit and analysis
    label_pipeline = process_data.make_label_pipeline()
    # ravel() just reshapes the data for easier processing
    actual_train_labels = label_pipeline.fit_transform(train_data).values.ravel()

    if my_args["ModelType"] == "tree":
        fit_pipeline = make_decision_tree_fit_pipeline()
        fit_params = make_decision_tree_params()
    elif my_args["ModelType"] == "svm":
        fit_pipeline = make_svm_fit_pipeline()
        fit_params = make_svm_params()
    else:
        print("pick --model type")
        sys.exit(1)

    if my_args["SplitterType"] == "k-fold":
        cv = sklearn.model_selection.KFold(n_splits=my_args["Folds"])
    elif my_args["SplitterType"] == "stratified":
        cv = sklearn.model_selection.StratifiedKFold(n_splits=my_args["Folds"])
    else:
        print("pick --splitter type")
        sys.exit(1)

    if my_args["SearchType"] == "grid":
        search_grid = sklearn.model_selection.GridSearchCV( fit_pipeline,
                                                            fit_params,
                                                            scoring="f1_micro",
                                                            n_jobs=-1,
                                                            cv=cv,
                                                            refit=True,
                                                            verbose=1 )
    elif my_args["SearchType"] == "random":
        search_grid = sklearn.model_selection.RandomizedSearchCV( fit_pipeline,
                                                                  fit_params,
                                                                  scoring="f1_micro",
                                                                  n_iter=my_args["Iterations"],
                                                                  n_jobs=-1,
                                                                  cv=cv,
                                                                  refit=True,
                                                                  verbose=1 )
    else:
        print("pick --search type")
        sys.exit(1)

    search_grid.fit(train_data, actual_train_labels)


    # examine best parameters
    print( "Best Score:", search_grid.best_score_ )
    print( "Best Params:", search_grid.best_params_ )

    print()
    print()
    print()

    scores = sklearn.model_selection.cross_val_score(search_grid.best_estimator_, train_data, actual_train_labels, scoring="f1", cv=cv, n_jobs=-1 )
    print( "CV:", scores.mean( ), scores.std( ) )


    print()
    print()
    print()

    predicted_train_labels = search_grid.best_estimator_.predict(train_data)

    print("actual training labels", actual_train_labels)
    print("predicted training labels", predicted_train_labels)
    print("Training Labels Correct:", calculateCorrectLabels(actual_train_labels, predicted_train_labels))

    actual_test_labels = label_pipeline.fit_transform(test_data).values.ravel()
    predicted_test_labels = search_grid.best_estimator_.predict(test_data)

    print("actual test labels", actual_test_labels)
    print("predicted test labels", predicted_test_labels)
    print("Test Labels Correct:", calculateCorrectLabels(actual_test_labels, predicted_test_labels))
        
    return

if __name__ == "__main__":
    main( sys.argv )
    
