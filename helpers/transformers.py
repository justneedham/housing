from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        rooms_per_household = x[:, self.rooms_ix] / x[:, self.household_ix]
        population_per_household = x[:, self.population_ix] / x[:, self.household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = x[:, self.bedrooms_ix] / x[:, self.rooms_ix]
            return np.c_[x, rooms_per_household, population_per_household, bedrooms_per_room]
        return np.c_[x, rooms_per_household, population_per_household]


class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return x[self.attribute_names].values
