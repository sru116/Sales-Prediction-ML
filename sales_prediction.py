# SALES PREDICTION USING MACHINE LEARNING

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error

# LOAD DATASET

df = pd.read_csv("advertising.csv")
print("Dataset Shape:",df.shape)

#DISPLAYING FIRST 5 ROWS

print("First 5 Rows:")
print(df.head())

#CORRELATION HEATMAP

plt.figure(figsize=(5,4))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="Blues"
)
plt.title("Correlation heatmap")
plt.tight_layout()
plt.savefig("heatmap.png",dpi=300)
plt.show()

#FEATURES AND TARGET

X = df[['TV','Radio','Newspaper']]
y = df['Sales']

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

#TRAIN MODEL

model = LinearRegression()
model.fit(X_train,y_train)
print("Model Trained Successfully!")

print("Intercept:")
print(model.intercept_)

feature_importance = pd.DataFrame({
    'Feature':X.columns,
    'Coefficient':model.coef_
})
print("Feature Importance:")
print(feature_importance)

plt.figure(figsize=(5,4))
plt.bar(
    feature_importance['Feature'],
    feature_importance['Coefficient']
)
plt.title("Feature Importance")
plt.xlabel("Advertising Medium")
plt.ylabel("Coefficient")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
plt.show()

y_pred=model.predict(X_test)

#COMPARE ACTUAL VS PREDICTED

comparison = pd.DataFrame({
    'Actual':y_test.values,
    'Predicted':y_pred
})
print("Actual vs Predicted:")
print(comparison.head(10))

#EVALUATION METRICS 

r2 = r2_score(y_test,y_pred)
mae = mean_absolute_error(y_test,y_pred)

plt.figure(figsize=(5,4))
plt.scatter(y_test,y_pred)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")

plt.title("Actual vs Predicted Sales")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=300)

plt.show()

# CUSTOM PREDICTION

tv=257
radio=46
newspaper=20

prediction = model.predict([[tv,radio,newspaper]])

print("custom prediction")
print("TV:",tv)
print("Radio:",radio)
print("Newspaper:",newspaper)
print("Predicted Sales:",prediction[0])