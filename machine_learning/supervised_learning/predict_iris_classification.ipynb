#####
#import library
import tensorflow as tf
import pandas as pd


#####
#get data
filename='https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/iris.csv'
iris=pd.read_csv(filename)
iris = pd.get_dummies(iris)
print(iris.columns)
iris.head()


#separate by independent and dependent variables(use the variable name in iris.head())
inde = iris[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']]
de = iris[['Species_setosa', 'Species_versicolor', 'Species_virginica']]
print(inde.shape, de.shape)


#####
#make a model structure
x = tf.keras.layers.Input(shape=[4])

h = tf.keras.layers.Dense(8)(x)
h=tf.keras.layers.BatchNormalization()(h)
h=tf.keras.layers.Activation('swish')(h)

h = tf.keras.layers.Dense(8)(x)
h=tf.keras.layers.BatchNormalization()(h)
h=tf.keras.layers.Activation('swish')(h)

h = tf.keras.layers.Dense(8)(x)
h=tf.keras.layers.BatchNormalization()(h)
h=tf.keras.layers.Activation('swish')(h)

y = tf.keras.layers.Dense(3, activation='softmax')(h)
model = tf.keras.models.Model(x, y)
model.compile(loss='categorical_crossentropy', metrics='accuracy')


#####
#fit the model by data
model.fit(inde, de, epochs=1000, verbose=0)

#####
#use model(predict)
model.predict(inde[0:5])
print(de[0:5])

#print weight from a result of learning
model.get_weights()

