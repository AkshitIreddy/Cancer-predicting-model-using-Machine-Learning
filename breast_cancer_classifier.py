import numpy as np
from sklearn import  svm
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

#print(plt.style.available)
style.use('Solarize_Light2')

df = pd.read_csv('breast-cancer-wisconsin.txt')
df.replace('?',-99999, inplace=True)
df.drop(['id'], axis = 1, inplace=True)

X = np.array(df.drop(['Class'], axis = 1))
y = np.array(df['Class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = svm.SVC(kernel = "linear")

clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)

example_measures = np.array([[4,2,1,1,1,2,3,2,1], [4,2,1,4,2,5,3,3,1],[6,2,1,1,1,2,3,2,1], [7,2,3,4,2,5,3,3,1],[8,2,1,4,1,2,3,2,1], [4,2,3,4,2,5,3,3,1],
[1,2,1,1,1,2,3,2,1], [7,2,3,5,2,5,3,3,1], [8,2,5,4,2,5,5,3,1],[8,2,4,1,6,2,3,2,1], [7,2,5,4,5,5,3,3,1],[8,5,6,4,4,2,6,2,1], [5,2,3,4,2,5,3,3,1]])

example_measures = example_measures.reshape(len(example_measures), -1)

prediction = clf.predict(example_measures)
print(prediction)

y = [i for i in range(len(example_measures))]
plt.scatter(prediction, y, label = "Benign - 2 \nMalignant - 4")
plt.title("Classification of cancer")
plt.legend()
plt.show()
