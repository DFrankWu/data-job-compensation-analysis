import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

class BestForest:

    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def find_n_estimators(self, max_value, n_of_values):
        n_estimators = {}
        for n in np.linspace((max_value//n_of_values), max_value, n_of_values):
            n_estimator_rf = RandomForestRegressor(n_estimators=int(n), max_features='sqrt', random_state=42)
            n_estimator_rf.fit(self.X_train, self.y_train.ravel())
            n_rf_y_test_pred = n_estimator_rf.predict(self.X_test)
            n_estimators[n] = r2_score(self.y_test, n_rf_y_test_pred)
        plt.plot(np.linspace((max_value//n_of_values), max_value, n_of_values), list(n_estimators.values()))
        plt.xlabel('Number of estimators')
        plt.ylabel('R2 Score')
        plt.title('R2 Score vs. # of estimators')
        plt.show()
        plt.clf()
        best_r2 = max(n_estimators, key=n_estimators.get)
        print(f'Best R2 Score {n_estimators[best_r2]} happens with {best_r2} of estimators')

    def find_max_depth(self, max_value, n_of_values):
        max_depth = {}
        for n in np.linspace((max_value//n_of_values), max_value, n_of_values):
            max_depth_rf = RandomForestRegressor(max_depth=int(n), max_features='sqrt', random_state=42)
            max_depth_rf.fit(self.X_train, self.y_train.ravel())
            depth_rf_y_test_pred = max_depth_rf.predict(self.X_test)
            max_depth[n] = r2_score(self.y_test, depth_rf_y_test_pred)
        plt.plot(np.linspace((max_value//n_of_values), max_value, n_of_values), list(max_depth.values()))
        plt.xlabel('Max Depth')
        plt.ylabel('R2 Score')
        plt.title('R2 Score vs. Max Depth')
        plt.show()
        plt.clf()
        best_r2 = max(max_depth, key=max_depth.get)
        print(f'Best R2 Score {max_depth[best_r2]} happens with max_depth of {best_r2}')

    def find_min_sample_split(self, max_value, n_of_values):
        min_sample_split = {}
        for n in np.linspace((max_value//n_of_values), max_value, n_of_values):
            min_sample_split_rf = RandomForestRegressor(min_samples_split=int(n), max_features='sqrt', random_state=42)
            min_sample_split_rf.fit(self.X_train, self.y_train.ravel())
            min_split_rf_y_test_pred = min_sample_split_rf.predict(self.X_test)
            min_sample_split[n] = r2_score(self.y_test, min_split_rf_y_test_pred)
        plt.plot(np.linspace((max_value//n_of_values), max_value, n_of_values), list(min_sample_split.values()))
        plt.xlabel('Minimum Split')
        plt.ylabel('R2 Score')
        plt.title('R2 Score vs. Minimum Split')
        plt.show()
        plt.clf()
        best_r2 = max(min_sample_split, key=min_sample_split.get)
        print(f'Best R2 Score {min_sample_split[best_r2]} happens with min_sample_split of {best_r2}')

    def find_min_samples_leaf(self, max_value, n_of_values):
        min_samples_leaf = {}
        for n in np.linspace((max_value//n_of_values), max_value, n_of_values):
            min_samples_leaf_rf = RandomForestRegressor(min_samples_leaf=int(n), max_features='sqrt', random_state=42)
            min_samples_leaf_rf.fit(self.X_train, self.y_train.ravel())
            min_leaf_rf_y_test_pred = min_samples_leaf_rf.predict(self.X_test)
            min_samples_leaf[n] = r2_score(self.y_test, min_leaf_rf_y_test_pred)
        plt.plot(np.linspace((max_value//n_of_values), max_value, n_of_values), list(min_samples_leaf.values()))
        plt.xlabel('Minimum Leaf')
        plt.ylabel('R2 Score')
        plt.title('R2 Score vs. Minimum Leaf')
        plt.show()
        plt.clf()
        best_r2 = max(min_samples_leaf, key=min_samples_leaf.get)
        print(f'Best R2 Score {min_samples_leaf[best_r2]} happens with min_samples_leaf of {best_r2}')


