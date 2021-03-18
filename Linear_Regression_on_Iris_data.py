import pandas as pd

df = pd.read_csv("Iris.csv")
print(df)

df["Species_code"] = pd.factorize(df.Species)[0]
df

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df["Species_code_sklearn"] = le.fit_transform(df.Species)
print(df)

from sklearn.linear_model import LinearRegression
Model = LinearRegression()

Model.fit(df[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]],df.Species_code)

input1 = float(input("Enter Sepal Length in Cm :- ")) 
input2 = float(input("Enter Sepal Width in Cm :- "))
input3 = float(input("Enter  Petal Length in Cm :- "))
input4 = float(input("Enter  Petal width in Cm :-"))

a = Model.predict([[input1,input2,input3,input4]])

if a == 0 or a < 0:
    print("Iris-setosa")
elif a > 0.8:
    print("Versicolor")
elif a > 1.8:
    print("Virginica")

b = Model.score(df[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]],df.Species_code)
print(f"Accurary is :- {b}")

#6.4,3.2,4.5,1.5