import pandas as pd
import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator, clone

# Clase para transformar cualquier modelo en un clasificador ordinal.
class MultipleModelOrdinalClassifier(BaseEstimator,ClassifierMixin):
    # I inherit from these classes so that the class is compatible with the sklearn API.
    def __init__(self, classifier):
        """
        Parameters
        ----------
        :classifier: classifier from which to build the model
        """
        self.classifier = classifier
        self.fitted_classifiers = []

    def _fit(self,X,y):
        #We keep the labels and put them in order
        self.labels = np.sort(y.unique())

        #Iterate through all the labels except the last one
        for i,label in enumerate(self.labels[:-1]):
            smaller_labels = self.labels[:i+1] #We make the set of labels <= than the current one
            greater_labels = self.labels[i+1:] #Makes the set of labels > than the current one

            #We build two dictionaries to replace the label values in the dataset.
            smaller_replacements = { label:0 for label in smaller_labels} 
            greater_replacements = { label:1 for label in greater_labels} 
            smaller_replacements.update(greater_replacements)

            y_i = y.replace(smaller_replacements)

            #We create a classifier and fit it to the dataset on the replaced labels.
            classifier = clone(self.classifier)
            classifier.fit(X,y_i)
            #We put the classifier in the fitter_classifiers
            self.fitted_classifiers.append(classifier)
    
    def _predict(self,X):
        predictions_greater = []
        
        #For each classifier we draw predictions in the form of probabilities.
        for cl in self.fitted_classifiers:
            prediction = cl.predict_proba(X)[:,1] #We will stick with the second probability, i.e. with P(target > Vi)
            predictions_greater.append(prediction)

        #We compute the probabilities of the first and last class
        primera = 1 - predictions_greater[0] #first
        ultima = predictions_greater[-1] # last
        probabilidades_clase = [primera]
        #We calculate the remaining probabilities 
        for i,pred in enumerate(predictions_greater):
            if i != 0 and i != len(predictions_greater): #We avoid going through first and last
                prob_i = predictions_greater[i-1]*(1-predictions_greater[i])
                probabilidades_clase.append(prob_i)

        #We insert the last probability so that they are in order in the array.
        probabilidades_clase.append(ultima)

        #We take the index of the probability with the highest value
        predictions = np.argmax(probabilidades_clase,axis=0)
        #We replace the indexes with the actual value of the label
        final_preds = pd.Series(predictions).replace({i:label for i,label in enumerate(self.labels)})

        return final_preds

    def get_modelos(self):
        return self.fitted_classifiers

    def fit(self, X, y):
        """
        Fit the model to the data
        Parameters
        ----------
        X predictor variables
        y predictor variable
        """
        self._fit(X,y)
    def predict(self, X) -> pd.Series:
        """
        Makes a prediction on a set of predictor variables
        Parameters
        ----------
        X predictor variables

        Returns
        ----------
        labels labels predicted by the model
        """
        return self._predict(X)