import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tabulate import tabulate

#Decision Tree Class
class DecisionTreeClassifier:
    def __init__(self, maximum_depth=None, criterion='gini', min_samples_split=2, min_samples_leaf=2): #defining intialization
        self.maximum_depth = maximum_depth
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        
    def gini(self, y):#defining gini
        unique_vals, frequency = np.unique(y, return_counts=True)
        gini_imputirty=(1 - sum([(count / len(y))**2 for count in frequency]))
        return gini_imputirty
    
    def misclassification_rate(self, y):#defining misclassification rate
        unique_vals, frequency = np.unique(y, return_counts=True)
        class_probabilities = frequency / len(y)
        misclassification = 1 - np.max(class_probabilities)
        return misclassification
   
    def entropy(self, y): #defining entropy
        unique_vals, frequency = np.unique(y, return_counts=True)
        class_probabilities = frequency / len(y)
        entrop = -np.sum([p * np.log2(p) if p > 0 else 0 for p in class_probabilities])
        return entrop
    
    def Tree_Traverse(self, x, node): #defining Tree Traversal
        if 'classified_as' in node:
            return node['classified_as']
        if x[node['best_feature']] >= node['best_limit']:
            return self.Tree_Traverse(x, node['right_leaf'])
        else:
            return self.Tree_Traverse(x, node['left_leaf'])

    def predict(self, X): #defining predict
        y_pred = np.array([self.Tree_Traverse(x, self.tree) for x in X])
        return y_pred

    def Tree_growth(self, X, y, depth=0): #defining Tree Shape
        number_of_samples, number_of_features = X.shape
        number_of_classes = len(np.unique(y))
        if (self.maximum_depth is not None and depth >= self.maximum_depth) or number_of_classes == 1 or number_of_samples < self.min_samples_split:
            return {'classified_as': np.bincount(y).argmax()}
        if self.criterion.lower() == 'gini':
            criterion_chosen = self.gini
        elif self.criterion.lower() == 'misclassification_rate':
            criterion_chosen = self.misclassification_rate
        elif self.criterion.lower() == 'entropy':
            criterion_chosen = self.entropy
        else:
            raise ValueError('Invalid criterion')
        indexes_of_features = np.arange(number_of_features)
        if number_of_features > 1:
            np.random.shuffle(indexes_of_features)
            indexes_of_features = indexes_of_features[:np.random.randint(1, number_of_features)]
        impurity = np.inf
        best_feature = None
        best_limit = None
        for feature in indexes_of_features:
            limits = np.unique(X[:, feature])
            for each_limit in limits:
                y_left, y_right = y[X[:, feature] < each_limit], y[X[:, feature] >= each_limit]
                if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                    continue
                current_impurity = (len(y_left)/number_of_samples) * criterion_chosen(y_left) + (len(y_right)/number_of_samples) * criterion_chosen(y_right)
                if current_impurity < impurity:
                    impurity, best_feature, best_limit = current_impurity, feature, each_limit
                    left_indexes, rigth_indexes = X[:, feature] < each_limit, X[:, feature] >= each_limit
        if impurity == np.inf:
            return {'classified_as': np.bincount(y).argmax()}
        left = self.Tree_growth(X[left_indexes], y[left_indexes], depth+1)
        right = self.Tree_growth(X[rigth_indexes], y[rigth_indexes], depth+1)
        Tree_shape = {'best_feature': best_feature, 'best_limit': best_limit, 'left_leaf': left, 'right_leaf': right}
        return Tree_shape
    
    def fit(self, X, y,weights=None): #defining fit function
        self.X = X
        self.y = y
        self.weights = weights
        self.tree = self.Tree_growth(X, y)

#Class for Random Forest Classifier
class RandomForestClassifier:
    def __init__(self,classifier, maximum_depth,criterion,min_samples_split=2, min_samples_leaf=2,num_trees=200,min_features=5):#defining Initialization
        self.classifier = classifier
        self.num_trees = num_trees
        self.min_features = min_features
        self.maximum_depth=maximum_depth
        self.criterion=criterion
        self.min_samples_split=min_samples_split
        self.min_samples_leaf=min_samples_leaf
        self.Jungle = []

    def predict(self, X):#defining predict function
        votes = np.zeros((X.shape[0],))
        votes += sum(model.predict(X[:, selected_features]) == 1 for model, selected_features in self.Jungle)
        y_pred = (votes >= (len(self.Jungle) / 2)).astype(int)
        return y_pred

    def fit(self, X, y,weights=None): #defining fit
        self.weights = weights
        number_of_features = X.shape[1]
        for _ in range(self.num_trees):
            indices = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
            X_train = X[indices]
            y_train = y[indices]
            num_selected_features = np.random.randint(self.min_features, number_of_features + 1)
            selected_features = np.random.choice(number_of_features, size=num_selected_features, replace=False)
            X_train = X_train[:, selected_features]
            model = self.classifier(maximum_depth=self.maximum_depth,criterion=self.criterion,min_samples_split=self.min_samples_split,min_samples_leaf=self.min_samples_split)
            model.fit(X_train, y_train)
            self.Jungle.append((model, selected_features))
    
class AdaBoostClassifier: #Class AdaBoost classifier
    def __init__(self, weak_learner, num_learners=100, learning_rate=1, maximum_depth=2,criterion='gini',min_samples_split=2,min_samples_leaf=2):#initialzation of Ada Boost
        self.weak_learner = weak_learner
        self.num_learners = num_learners
        self.learning_rate = learning_rate
        self.estimators = []
        self.weights_of_estimators = []
        self.maximum_depth=maximum_depth
        self.criterion=criterion
        self.min_samples_split=min_samples_split
        self.min_samples_leaf=min_samples_leaf

    def predict(self, X): #predict function of Ada Boost
        number_of_samples = X.shape[0]
        y_pred = np.zeros(number_of_samples)
        y_pred = np.sign(sum(weight * est.predict(X) for weight, est in zip(self.weights_of_estimators, self.estimators)))
        return y_pred
        
    def fit(self, X, y): #fit function of Ada Boost
        sample_weight = np.full(X.shape[0], 1 / X.shape[0]) 
        for i in range(self.num_learners):
            estimator = self.weak_learner(maximum_depth=self.maximum_depth,criterion=self.criterion,min_samples_split=self.min_samples_split,min_samples_leaf=self.min_samples_split)
            estimator.fit(X, y,weights=sample_weight)
            y_pred = estimator.predict(X)
            incorrect_classification = np.where(y_pred != y, 1, 0)
            error_in_estimator = np.dot(sample_weight, incorrect_classification) / np.sum(sample_weight)
            if error_in_estimator <= 0: self.weights_of_estimators.append(1.0); self.estimators.append(estimator); break
            estimator_weight = self.learning_rate * np.log((1 - error_in_estimator) / error_in_estimator)
            self.weights_of_estimators.append(estimator_weight)
            self.estimators.append(estimator)
            sample_weight *= np.exp(estimator_weight * incorrect_classification * ((sample_weight > 0) | (estimator_weight < 0)))
            sample_weight /= np.sum(sample_weight)

#preparing Data       
np.random.seed(2147483648)
T_data = sns.load_dataset('titanic')
T_data.dropna(inplace=True) 
T_data.drop(['deck', 'embark_town','alive'], axis=1, inplace=True)
T_data = pd.get_dummies(T_data, columns=['class', 'embarked']) 
T_data['alone'] = np.where(T_data['alone'] == True, 1, 0)
T_data['who'] = np.where(T_data['who'] == 'man', 1, 0)
#T_data['alive'] = np.where(T_data['alive'] == 'yes', 1, 0)
T_data['sex'] = np.where(T_data['sex'] == 'male', 1, 0)
T_data['adult_male'] = np.where(T_data['adult_male'] == 'True', 1, 0)
X = T_data.drop('survived', axis=1).values
y = T_data['survived'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0,shuffle=True)

#Model 1
model_1 = DecisionTreeClassifier(maximum_depth=3, criterion='gini')
model_1.fit(X_train, y_train)
y_pred = model_1.predict(X_test)
model_1_score=round(accuracy_score(y_test, y_pred)*100,ndigits=2)

#Model 2
model_2 = RandomForestClassifier(classifier=DecisionTreeClassifier,maximum_depth=3,criterion='gini')
model_2.fit(X_train, y_train)
y_pred = model_2.predict(X_test)
model_2_score = round(accuracy_score(y_test, y_pred)*100,ndigits=2)

#Model 3
model_3 = AdaBoostClassifier(weak_learner=DecisionTreeClassifier, num_learners=50, learning_rate=1,maximum_depth=3,criterion='gini')
model_3.fit(X_train, y_train)
y_pred = model_3.predict(X_test)
model_3_score = round(accuracy_score(y_test, y_pred)*100,ndigits=2)

#Tabulating the Result
headers = ['\033[33mModel Names\033[0m', '\033[33mAccuracy (%)\033[0m']
data = [["Decision Tree (Criterion:\033[36mGini\033[0m)", f'\033[32m{model_1_score} %\033[0m'],
        ["Random Forest (Classifier:\033[36mDecision Tree\033[0m & Criterion:\033[36mGini\033[0m)", f'\033[32m{model_2_score} %\033[0m'],
        ["AdaBoost (Classifier:\033[36mDecision Tree\033[0m & Criterion:\033[36mGini\033[0m)", f'\033[32m{model_3_score} %\033[0m']]
options = {"numalign": "center","stralign": "center","tablefmt": "fancy_grid"}
print(tabulate(data,headers=headers, **options))

##########  Accuracy will be displaying
## Note ##  same as seed and random state are given to it.
##########  To change remove seed and random state