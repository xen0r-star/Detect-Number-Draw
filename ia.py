from keras import *

model = Sequential()

model.add(layers.Dense(units=3, input_shape=[1]))

model.add(layers.Dense(units=64))
model.add(layers.Dense(units=64))
model.add(layers.Dense(units=64))

model.add(layers.Dense(units=1))


entree=[1,2,3,4,5]
sortie=[2,4,6,8,10]


model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(x=entree,y=sortie,epochs=500)


while True:
    x = int(input('Nombre : '))
    print('Prédiction = ' + str(model.predict([x])))
