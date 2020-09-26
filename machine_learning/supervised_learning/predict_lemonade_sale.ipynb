import tensorflow as tf
import pandas as pd


####get data

#get data from file
file='https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/lemonade.csv'
lemon=pd.read_csv(file)

#separate by independent and dependent variables
inde=lemon[['온도']]
de=lemon[['판매량']]


####make a model structure

#the num of inde(independent variable) variable is one, so shape=[1]
X=tf.keras.layers.Input(shape=[1])
#the num of de(dependent variable) variable is one, so Dense(1)
Y=tf.keras.layers.Dense(1)(X)
model=tf.keras.models.Model(X,Y)

#loss is mean of pow(predict-result, 2). As the loss nears zero, it is a well-learned model.
#모델이 학습할 방법 정의
#define how model will learn
model.compile(loss='mse')


####fit the model by data
#epochs : Decide how many times to repeat. In this code, 1000 times
#Repeat learning until loss is as low as desired.
#verbose=0 : Do not print on the screen during learning.

model.fit(inde, de, epochs=10000, verbose=0)

#Because of verbose=0, the loss value is unknown, so try to find the loss value by repeating 10 more times
model.fit(inde, de, epochs=10)

####use model(predict)
#온도가 15일때의 판매량 예측
#predict sales when tempurature is 15 degrees Celsius
print("Predictions : ", model.predict([[15]]))

