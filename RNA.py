import pandas as pd
import tensorflow as tf

x_train = pd.read_csv("x_train_data.csv")
y_test = pd.read_csv("x_test_data.csv")
y_train = pd.read_csv("y_train_data.csv")
y_test = pd.read_csv("y_test_data.csv")

optimizer = tf.keras.optimizers.Adam(lr=0.01)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(3,input_shape=x_train.shape[1:],activation="sigmoid"))
model.add(tf.keras.layers.Dense(4, activation="sigmoid"))
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=250)
print("\nEvaluate:")
model.evaluate(x_test, y_test)