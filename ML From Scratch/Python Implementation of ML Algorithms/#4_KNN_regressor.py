import numpy as np 
import math 


def euclidean_dist(x1, x2):
    distance = math.sqrt(np.sum((x1 - x2) ** 2))
    return distance


class KNNRegressor():
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors 


    def fit(self, X, y):
        self.X = X
        self.y = y 
        return self 
    

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        print(predictions)
        return predictions


    def _predict(self, x):
        distances = []
        for sample in self.X:
            distances.append(euclidean_dist(x, sample))
        
        sorted_indices = np.argsort(np.array(distances))[:self.n_neighbors]
        values = [self.y[index] for index in sorted_indices]
        return np.mean(values)



if __name__ == "__main__":
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    df = pd.read_csv("/Users/ammarahmed/My Files/Code Files/100_Days_ML_Code/ML Using Core Libraries/datasets/50_Startups.csv")
    X = df.iloc[:, :-1].values
    y = df["Profit"].values
    
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(missing_values=0.00, strategy="mean")
    X[:, [0, 2]] = imputer.fit_transform(X[:, [0, 2]])
    X

    le_X = LabelEncoder()
    X[:, -1] = le_X.fit_transform(X[:, -1])
    X

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

    sc = StandardScaler()
    X_train, X_test = sc.fit_transform(X_train), sc.fit_transform(X_test)
    
    knn = KNNRegressor(n_neighbors=7)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    print(np.mean(np.sum((y_test - y_pred)**2)))
