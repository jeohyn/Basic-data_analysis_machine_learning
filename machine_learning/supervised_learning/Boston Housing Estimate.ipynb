#####
#import library
import tensorflow as tf
import pandas as pd


#####
#get data
filename='https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/boston.csv'
boston=pd.read_csv(filename)
print(boston.columns)
boston.head()


#separate by independent and dependent variables
inde=boston[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax',
       'ptratio', 'b', 'lstat']]
de=boston[['medv']]
print(inde.shape, de.shape)


#####
#make a model structure
#swish : active function
#Dense(10, activation='swish) : 10nodes as a result
#h : hidden layer
#in this code, there are 3 hidden layers
#h=tf.keras.layers.Dense(10, activation='swish')(x) is same as
#h=tf.keras.layers.Dense(10)(x) h=tf.keras.layers.Activation('swish')(h)


x=tf.keras.layers.Input(shape=[13])
h=tf.keras.layers.Dense(10, activation='swish')(x)
h=tf.keras.layers.Dense(3, activation='swish')(h)
h=tf.keras.layers.Dense(3, activation='swish')(h)
y=tf.keras.layers.Dense(1)(h)
model=tf.keras.models.Model(x,y)
model.compile(loss='mse')

#check the model structure
model.summary()


#####
#fit the model by data
model.fit(inde, de, epochs=1000, verbose=0)
model.fit(inde, de, epochs=10)


#####
#use model(predict)
print(model.predict(inde[:5]))
print(de[:5])

