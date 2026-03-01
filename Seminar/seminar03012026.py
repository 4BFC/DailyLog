# ----------- 1. Supervised - Classification -----------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Study hours (X)
X = np.array([[1], [2], [3], [4], [5], [6]])

# Pass (1) or Fail (0)
y = np.array([0, 0, 0, 1, 1, 1])

# Create model
model = LogisticRegression()

# Train model
model.fit(X, y)

# Predict new student
new_student = np.array([[4.5]])
prediction = model.predict(new_student)

print("Prediction:", prediction[0])


# ----------- 2. Supervised - Regression -----------
import numpy as np
from sklearn.linear_model import LinearRegression

# 회사원의 데이터 

# Years of experience
X = np.array([[1], [2], [3], [4], [5]])

# Salary
y = np.array([30000, 40000, 50000, 60000, 70000])



# Study hours (X)
X = np.array([[1], [2], [3], [4], [5], [6]])

# Test score
y = np.array([10, 20, 30, 50, 60, 70])

# 학생이 공부하는시간에 따라 시험 score 뭐일까?


# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict salary for 6 years experience
prediction = model.predict([[6]])

print("Predicted Salary:", prediction[0])


# ----------- 3A. Unsupervised - Clustering -----------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Customer data (Spending Score, Income)
X = np.array([
    [200, 30],
    [250, 35],
    [300, 40],
    [1000, 80],
    [1100, 85],
    [1200, 90]
])

# Create model (2 clusters)
model = KMeans(n_clusters=2)

# Train model
model.fit(X)

# Get cluster labels
labels = model.labels_

print("Cluster assignments:", labels)

# Visualize
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.xlabel("Spending")
plt.ylabel("Income")
plt.show()

# ----------- 3B. Unsupervised - Clustering -----------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate synthetic customer data
X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

# random but same data 
# Create KMeans model
kmeans = KMeans(n_clusters=3, random_state=42)

# Train model
kmeans.fit(X)

# Get cluster labels
labels = kmeans.labels_

# Plot results
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.title("Customer Clusters")
plt.show()



# ----------- Hiring Example -----------


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Cosmos에서 지원한 사람들의 이력서, 판단 0/1
# Cosmos에서 새로운 사람올때 -> 이력서 -> hire/not hire
# Hiring Dataset
# 이력서 
data = pd.DataFrame({
    "GPA": [3.8, 3.0, 3.5, 2.8, 3.9, 3.2, 3.6, 2.9],
    "Experience": [2, 0, 1, 0, 3, 1, 2, 0],
    "InterviewScore": [85, 60, 75, 50, 90, 70, 80, 55],
    "Hired": [1, 0, 1, 0, 1, 0, 1, 0] #정답
})

data.head()

# Divide: X = input, Y = label (answer)
X = data[["GPA", "Experience", "InterviewScore"]] #이력서
y = data["Hired"] #판단
# What combination of GPA + Experience + InterviewScore leads to Hired = 1?

#divide into train and test groups
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

#train 학습 , test (새로운)
#100% = model이 외우는거잖아
#85% = train (6)
#25% = test (2)


model = RandomForestClassifier()

#decision tree -> multiple
#tree마다 pattern 찾는다 
## -> interviewscore > 70 -> hire: 1  /// pattern... (rule)
# tree rule pass, not pass, not pass (tree: 5개)
# final deicsion = majority vote (not pass: 4개) -> not pass
model.fit(X_train, y_train) #6

y_pred = model.predict(X_test) #2
# 1: 맞고, 1 틀렸어 

print("Accuracy:", accuracy_score(y_test, y_pred))
#Accuracy : 높을소록 좋음
#1미면 문제
#높게 해서


