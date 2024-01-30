import numpy as np 
import math

def euclidean_dist(x1, x2):
    distance = math.sqrt(np.sum((x1 - x2) ** 2))
    return distance


class KNNClassifier():
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors


    def fit(self, X, y):
        self.X = X
        self.y = y 
        return self 


    def predict(self, X):
        votes_for_all = [self._predict(x) for x in X]
        return votes_for_all


    def _predict(self, x):
        distances = [] 
        for sample in self.X:
            distances.append(euclidean_dist(x, sample))

        sorted_indices = np.argsort(np.array(distances))[:self.n_neighbors]
        classes = [self.y[index] for index in sorted_indices]
        pred_class = self._majority_vote(classes)

        return pred_class


    @staticmethod
    def _majority_vote(votes):
        counts = np.bincount(votes)
        return np.argmax(counts)



if __name__ == "__main__":
    import pandas as pd 
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    df = pd.read_csv("/Users/ammarahmed/My Files/Code Files/100_Days_ML_Code/ML Using Core Libraries/datasets/Social_Network_Ads.csv")
    X = df.iloc[:, [2, 3]].values 
    y = df.iloc[:, -1].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

    sc = StandardScaler()
    X_train, X_test = sc.fit_transform(X_train), sc.fit_transform(X_test)
    
    knn = KNNClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)


    print(f"KNN Accuracy: {np.sum(y_pred == y_test) / len(y_test)}")