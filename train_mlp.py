import numpy as np 
import tensorflow as tf

from MLP import MLP_Model

# ****** parameters *********

n_epochs = 200 
batch_size = 16 
early_stopping_patience = 70 
loss_fcn =  "binary_crossentropy" 
optimizer  = "adam" 
metrics = "accuracy" 
shuffle = True

early_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss",min_delta=0,patience=early_stopping_patience,verbose=0,mode="auto",baseline=None,restore_best_weights=True)


# ****** Random example data  *********

x_train = np.random.rand(960, 192) # actual feature size as in publication 
x_val = np.random.rand(240, 192) # actual feature size as in publication 
x_test = np.random.rand(240, 192) # actual feature size as in publication 

y_train = np.random.randint(0, 2, size=(960, )) # binary classification 
y_val = np.random.randint(0, 2, size=(240, )) # binary classification 
y_test = np.random.randint(0, 2, size=(240, )) # binary classification 



# ****** Train model *********

# create MLP model
model = MLP_Model(x_train=x_train)

# compile the model 
model.compile(loss=loss_fcn, optimizer=optimizer, metrics=metrics)

# train the model
history = model.fit(x_train,
                    y_train,
                    epochs  = n_epochs,
                    batch_size= batch_size,
                    shuffle = shuffle,
                    validation_data = (x_val, y_val),
                    callbacks = [early_callback])


# store history of training process
history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]



