#!/usr/bin/env python3

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
import sklearn.base
import sklearn.pipeline
import sklearn.impute
import scipy
import sys, os.path

def get_data(filename, ext):
    # make sure to read the data in chunks, but all into one DataFrame
    if ext == "feather":
        data = pd.read_feather(filename)
    else:
        data = pd.read_csv(filename)
    return data


def get_feature_columns():
    return get_numerical_feature_columns() +  get_categorical_feature_columns() 

def get_numerical_feature_columns():
    # 'Finish', 'JVF', 'TotJ', 'JV%' 'Days' are all features that will reveal the winner, and are not known until the end of the game
    # Current Season Features: 'SurvSc', 'SurvAv', 'ChW', 'ChA', 'ChW%', 'SO', 'VFB', 'VAP', 'TotV', 'TCA' 'TC%', 'wTCR' 'MPF'
    # return ["Season", "SurvSc", "SurvAv", "Days", "Time", "ChW", "ChA", "ChW%", "SO", "MPF", "VAP", "TotV", "TCA", "TC%", "wTCR", "JVF", "TotJ", "JV%", "Idols found", "Idols played", "Votes voided", "Boot avoided", "Day found", "Day played" ]
    # the features below are the features we have throughout the entirety of the game. 
    return["ChW", "ChA", "ChW%", "SO", "MPF", "VAP", "TotV", "TCA", "TC%", "wTCR"  ]


def get_derived_numerical_feature_columns():
    return ["ChA*MPF", "Average", "Score"]
    # return ["Score"]
    # return ["ChA*MPF"]

def get_categorical_feature_columns():
    return []

def get_derived_categorical_feature_columns():
    return []

def get_all_feature_columns():
    return get_numerical_feature_columns() + get_derived_numerical_feature_columns() \
        + get_categorical_feature_columns() + get_derived_categorical_feature_columns()

def get_label_columns():
    return ["Winner"]



def get_column( data, i ):
    #
    # fetch the column of interest.  Important to know if this is a pandas.DataFrame or numpy.ndarray
    #
    if False:
        X = [ ]
    elif isinstance( data, pd.core.frame.DataFrame ):
        X = data.iloc[ :, i ]
    elif isinstance( data, np.ndarray ):
        X = data[ :, i ]
    elif isinstance( data, scipy.sparse.csr.csr_matrix ):
        X = data[:,i].todense()
        X = np.asarray(X)
    else:
        raise Exception( 'data is unexpected type: ' + str( type(data) ) )
    return X

#
# Pipeline member to display the data at this stage of the transformation.
#
class Printer( sklearn.base.BaseEstimator, sklearn.base.TransformerMixin ):
    
    def __init__( self ):
        return

    def fit( self, X, y=None ):
        return self

    def transform( self, X, y=None ):
        print("type",type(X))
        print("shape",X.shape)
        print("X[0]",X[0])
        print( X )
        return X


# Only for demonstration purposes
# I don't actually think these will be helpful
class DerivedNumericalAttributesAdder( sklearn.base.BaseEstimator, sklearn.base.TransformerMixin ):
    def __init__( self ):
        return

    def fit( self, X, y=None ):
        # do nothing
        return self

    def transform( self, X, y=None ):
        # print("Derivingg numerical features")
        # Deriving Challenge's Appearances * Mean % finish is a feature that is meant to weight MPF
        # by rewarding players if participating in more challenges. This is needed because a player 
        # might only participate in 2 challenges, winning both, and therefore having a perfect MPF
        # before getting voted out.
        X.insert(len(X.columns), "ChA*MPF", X.loc[:,'ChA'] * X.loc[:,'MPF'])

        # the features derived below are based off the data creators' formulas for trying to implement
        # a total score that includes 'social' factors. The creators' formula are much like below except that
        # here we don't use any sort of 'Jury' factors in the forumla. We do this because the feature seems useful
        # in calculating a winner, but in some cases when factoring in 'Jury' Features, the score can reveal a winner ( where their SurvAv > 2 ). 
        # so the purpose of the formulas below are to use the 'good' in the original forumlas and eliminate the 'bad'.
        X.insert(len(X.columns), "Average", X.loc[:,'ChW'] + X.loc[:,'wTCR'])
        X.insert(len(X.columns), "Score", X.loc[:,'ChW%'] + X.loc[:,'TC%'])
        # print(" X after deriving is: ")
        # print()
        # print(X)
        return X

# Only for demonstration purposes
# I don't actually think these will be helpful
class DerivedCategoricalAttributesAdder( sklearn.base.BaseEstimator, sklearn.base.TransformerMixin ):
    def __init__( self ):
        return

    def fit( self, X, y=None ):
        # do nothing
        return self

    def transform( self, X, y=None ):
        return X

    
    
# No outliers cut here.  Just a placeholder
class OutlierCuts( sklearn.base.BaseEstimator, sklearn.base.TransformerMixin ):

    def __init__( self ):
        return

    def fit( self, X, y=None ):
        # do nothing
        return self

    def transform( self, X, y=None ):
        # values = # the transformed values of X
        values = X
        # values = values[ ( values.Finish <= 25 ) ]
        # print( "-----------" )
        # print( "-----------" )
        # print( "-----------" )
        # print("VALUES: ", values)
        # print( "-----------" )
        # print( "-----------" )
        # print( "-----------" )
        return values

    

# keep predictors, remove labels
class DataFrameSelector( sklearn.base.BaseEstimator, sklearn.base.TransformerMixin ):
    
    def __init__( self, do_predictors=True, do_numerical=True ):
        self.do_predictors = do_predictors
        self.do_numerical = do_numerical

        self.mCategoricalPredictors = get_categorical_feature_columns()
        self.mNumericalPredictors = get_numerical_feature_columns()
        self.mLabels = get_label_columns()
        
        return

    def fit( self, X, y=None ):
        if self.do_predictors:
            if self.do_numerical:
                self.mAttributes = self.mNumericalPredictors
            else:
                self.mAttributes = self.mCategoricalPredictors                
        else:
            self.mAttributes = self.mLabels
        return self

    def transform( self, X, y=None ):
        # only keep columns selected
        values = X[ self.mAttributes ]
        return values


def make_numerical_predictor_params():
    params = { 
        "features__numerical__numerical-predictors-only__do_predictors" : [ True ],
        "features__numerical__numerical-predictors-only__do_numerical" : [ True ],
        "features__numerical__missing-values__strategy": [ 'median', 'mean', 'most_frequent' ],
    }
    return params

def make_categorical_predictor_params():
    params = { 
        "features__categorical__categorical-predictors-only__do_predictors" : [ True ],
        "features__categorical__categorical-predictors-only__do_numerical" : [ False ],
        "features__categorical__missing-data__strategy": [ 'most_frequent' ],
        "features__categorical__encode-category-bits__categories": [ 'auto' ],
    }
    return params

def make_predictor_params():
    p1 = make_numerical_predictor_params()
    # p2 = make_categorical_predictor_params()
    # p1.update(p2)
    return p1

def make_numerical_predictor_pipeline( ):
    # Numerical predictor pipeline
    items = [ ]
    items.append( ( "remove-outliers", OutlierCuts( ) ) )
    items.append( ( "numerical-predictors-only", DataFrameSelector( do_predictors=True, do_numerical=True ) ) )
    items.append( ( "derived-attributes", DerivedNumericalAttributesAdder( ) ) )
    items.append( ( "missing-values", sklearn.impute.SimpleImputer( missing_values=np.nan, strategy='constant', fill_value="NA" ) ) )
    # Note that StandardScaler will transform data from pandas.DataFrame to numpy.array
    items.append( ( "scaler", sklearn.preprocessing.StandardScaler( copy=False ) ) )
    #items.append( ("print-numbers",Printer()) )
    numerical_pipeline = sklearn.pipeline.Pipeline( items )
    return numerical_pipeline

def make_categorical_predictor_pipeline(do_one_hot):
    # Categorical predictor pipeline
    items = [ ]
    items.append( ( "remove-outliers", OutlierCuts( ) ) )
    items.append( ( "categorical-predictors-only", DataFrameSelector( do_predictors=True, do_numerical=False ) ) )
    items.append( ( "derived-attributes", DerivedCategoricalAttributesAdder( ) ) )
    items.append( ( "missing-data", sklearn.impute.SimpleImputer( strategy="most_frequent" ) ) )
    if do_one_hot:
        #items.append( ("print-categories-pre-onehot",Printer()) )
        items.append( ( "encode-category-bits", sklearn.preprocessing.OneHotEncoder(categories='auto',handle_unknown='ignore') )  )
        #items.append( ("print-categories-post-onehot",Printer()) )
    #items.append( ("print-categories",Printer()) )
    categorical_pipeline = sklearn.pipeline.Pipeline( items )
    return categorical_pipeline

def make_predictor_pipeline(do_one_hot):
    # union of the two pipelines
    items = [ ]
    items.append( ( "numerical", make_numerical_predictor_pipeline() ) )
    # items.append( ( "categorical", make_categorical_predictor_pipeline(do_one_hot=True) ) )
    pipeline = sklearn.pipeline.FeatureUnion( transformer_list=items )
    return pipeline

def make_label_pipeline():
    items = [ ]
    items.append( ( "labels-only", DataFrameSelector( do_predictors=False ) ) )
    pipeline = sklearn.pipeline.Pipeline( items )
    return pipeline




def main( argv ):
    if len( argv ) > 1:
        filename = argv[ 1 ]
    else:
        filename = "a.csv"

    if os.path.exists( filename ):
        basename, ext = filename.split( '.' )
        data = get_data( filename, ext )
        

        predictor_pipeline = make_predictor_pipeline(do_one_hot=False)
        label_pipeline = make_label_pipeline( )
        predictors_processed = predictor_pipeline.fit_transform( data )
        labels_processed = label_pipeline.fit_transform( data )
        print(predictors_processed)
        print(labels_processed)
    else:
        print( filename + " doesn't exist." )
    
    return

if __name__ == "__main__":
    main( sys.argv )
    
