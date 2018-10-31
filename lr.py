from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np 
from sklearn.linear_model import LogisticRegression



np.random.seed(seed=0)
X_0 = np.random.multivariate_normal( [2,2],  [[2,0],[0,2]],  50 )
y_0 = np.zeros(len(X_0))
 
X_1 = np.random.multivariate_normal( [6,7],  [[3,0],[0,3]],  50 )
y_1 = np.ones(len(X_1))
 
X = np.vstack((X_0, X_1))
y = np.append(y_0, y_1)
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
 
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)


print(X_train_std,y_train)

 
lr = LogisticRegression()
lr.fit(X_train_std, y_train)

print (lr.predict(X_test_std))
print (lr.score(X_test_std, y_test))
print (lr.intercept_)
print (lr.coef_)


