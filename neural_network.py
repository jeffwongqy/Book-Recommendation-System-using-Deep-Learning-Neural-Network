import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout
from tensorflow.keras.models import  Model
from tensorflow.keras.optimizers import Adam, RMSprop
from keras.regularizers import l2
from keras.metrics import RootMeanSquaredError
from keras.callbacks import EarlyStopping
from keras.constraints import non_neg


# define function for embedding path for both user and book
def embedding_path(latent_factor, num_unique_users, num_unique_books):
    
    # create the user embedding path 
    user_input = Input(shape = [1], name = "user_input")
    userEmbedding = Embedding(num_unique_users + 1, 
                              latent_factor, 
                              name = 'non_neg_user_embedding',
                              embeddings_initializer = 'he_normal',
                              embeddings_regularizer = l2(1e-4),
                              embeddings_constraint = non_neg())(user_input)
    user_vector = Flatten(name = 'user_flatten')(userEmbedding)
    
    # create the book embedding path 
    book_input = Input(shape = [1], name = 'book_input')
    bookEmbedding = Embedding(num_unique_books + 1, 
                              latent_factor, 
                              name = 'non_neg_book_embedding', 
                              embeddings_initializer = 'he_normal',
                              embeddings_regularizer = l2(1e-4),
                              embeddings_constraint = non_neg())(book_input)
    book_vector = Flatten(name = 'book_flatten')(bookEmbedding)
    
    return user_input, user_vector, book_input, book_vector



# define function for neural network A
def neural_network_A(bookVec, userVec, userInput, bookInput, learn_rate, momentum):
    # concat the user and book features
    concat = Concatenate()([bookVec, userVec])
    
    # add the fully-connected dense layers
    nn = Dense(512, activation = 'relu')(concat)
    nn = Dropout(0.25)(nn)
    nn = Dense(256, activation = 'relu')(nn)
    nn = Dropout(0.25)(nn)
    nn = Dense(128, activation = 'relu')(nn)
    nn = Dropout(0.25)(nn)
    nn = Dense(64, activation = 'relu')(nn)
    nn = Dropout(0.25)(nn)
    output = Dense(1, activation = 'relu')(nn)
    
    # create the model
    model = Model(inputs = [userInput, bookInput],
                  outputs = output)
    
    # compile the neural network
    model.compile(optimizer = RMSprop(learning_rate = learn_rate, momentum = 0.3),
                  loss = 'mean_squared_error',
                  metrics = ['mae', RootMeanSquaredError(name= 'rmse')])
    
    return model



# define function for neural network B 
def neural_network_B(bookVec, userVec, userInput, bookInput, learn_rate):
    # concat the user and book features
    concat = Concatenate()([bookVec, userVec])
    
    # add the fully-connected dense layers with dropout
    nn = Dense(128, activation = 'relu')(concat)
    nn = Dropout(0.5)(nn)
    nn = Dense(64, activation = 'relu')(nn)
    nn = Dropout(0.25)(nn)
    output = Dense(1, activation = 'relu')(nn)
    
    # create the model
    model = Model(inputs = [userInput, bookInput],
                  outputs = output)
    
    # compile the neural network
    model.compile(optimizer = Adam(learning_rate = learn_rate),
                  loss = 'mean_squared_error',
                  metrics = ['mae', RootMeanSquaredError(name= 'rmse')])
    
    return model



# define function for neural network C
def neural_network_C(bookVec, userVec, userInput, bookInput, learn_rate, momentum):
    # concat features
    concat = Concatenate()([bookVec, userVec])
    
    # add some neural network with dropout
    nn = Dense(100, activation = 'relu')(concat)
    nn = Dropout(0.2)(nn)
    nn = Dense(50, activation = 'relu')(nn)
    nn = Dropout(0.1)(nn)
    output = Dense(1, activation = 'relu')(nn)
    
    # create model
    model = Model(inputs = [userInput, bookInput],
                  outputs = output)
    
    # compile network
    model.compile(optimizer = RMSprop(learning_rate = learn_rate, momentum = momentum),
                  loss = 'mean_squared_error',
                  metrics = ['mae', RootMeanSquaredError(name= 'rmse')])
    
    return model

    

# define function for loss curve
def loss_curve(history_model, title):
    plt.figure(figsize = (10, 6))
    sns.set_style("darkgrid")
    plt.plot(history_model.history['loss'], label = "Training Loss")
    plt.plot(history_model.history['val_loss'], label = "Validation Loss")
    plt.title("Training and Validation Loss for " + title, fontweight = 'bold', fontsize = 14)
    plt.xlabel("Number of Epochs", fontweight = 'bold', fontsize = 12)
    plt.ylabel("Loss", fontsize = 12, fontweight = 'bold')
    plt.legend(shadow = True)
    plt.show()
    



# define function for MAE curve
def mae_curve(history_model, title):
    plt.figure(figsize = (10, 6))
    sns.set_style("darkgrid")
    plt.plot(history_model.history['mae'], label = "Training MAE Error")
    plt.plot(history_model.history['val_mae'], label = "Validation MAE Error")
    plt.title("Training and Validation MAE for " + title, fontweight = 'bold')
    plt.xlabel("Number of Epochs", fontweight = 'bold', fontsize = 12)
    plt.ylabel("MAE", fontsize = 12, fontweight = 'bold')
    plt.legend(shadow = True)
    plt.show()    



# define function for RMSE curve
def rmse_curve(history_model, title):
    plt.figure(figsize = (10, 6))
    sns.set_style("darkgrid")
    plt.plot(history_model.history['rmse'], label = "Training RMSE Error")
    plt.plot(history_model.history['val_rmse'], label = "Validation RMSE Error")
    plt.title("Training and Validation RMSE for " + title, fontweight = 'bold')
    plt.xlabel("Number of Epochs", fontweight = 'bold', fontsize = 12)
    plt.ylabel("RMSE", fontsize = 12, fontweight = 'bold')
    plt.legend(shadow = True)
    plt.show()    




# read the csv file 
preprocessed_df = pd.read_csv(r"C:/Users/jeffr/Desktop/Coursework Final/preprocessing.csv")

###################################### Pre-processing #################################################
# to extract the relevant columns necessary for deep training purpose
userItemRatings_df = preprocessed_df[['user_id', 'book_id', 'book_rating']]

# to normalize the book rating column between 0 and 1
x = userItemRatings_df.drop('book_rating', axis = 1)
minRatings = min(userItemRatings_df['book_rating'])
maxRatings = max(userItemRatings_df['book_rating'])
y = userItemRatings_df['book_rating'].apply(lambda x: (x - minRatings)/(maxRatings - minRatings)).values


# split the dataset into 70% training and 30% testing sets with random state of 42
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)



####################################### Embedding Path #################################################
# to count the number of unique users in the (i.e. userItemRatings_df['user_id'])
num_unique_users = userItemRatings_df['user_id'].nunique()
# to count the number of unique books in the (i.e. userItemRatings_df['book_id'])
num_unique_books = userItemRatings_df['book_id'].nunique()


# define the latent factor
latent_factor = 10

# call the function to create the embedding path for user and book features
userInput, userVector, bookInput, bookVector = embedding_path(latent_factor, num_unique_users, num_unique_books)



######################### Model Building for Respective Neural Networks #################################

# create an early stopping object
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 1, mode = 'auto')

# call the function to build the neural network for recommender system A
model_A = neural_network_A(bookVector, userVector, userInput, bookInput, 0.00001, 0.2)

# display the neural network summary for model A
model_A.summary()

# train and fit the model A using the training set. 
history_modelA = model_A.fit([X_train['user_id'].values, X_train['book_id'].values],
                             y_train,
                             epochs = 30, 
                             batch_size = 512,
                             validation_split = 0.2, 
                             callbacks = [early_stopping],
                             verbose = 1)



# call the function to build the network for recommender system B
model_B = neural_network_B(bookVector, userVector, userInput, bookInput, 0.00001)

# display the model summary for recommender system B
model_B.summary()

# train and fit recommender system B using the training set
history_modelB = model_B.fit([X_train['user_id'].values, X_train['book_id'].values],
                             y_train,
                             epochs = 30,
                             batch_size = 512,
                             validation_split = 0.2,
                             callbacks = [early_stopping],
                             verbose = 1)



# call the function to build the network for recommender system C
model_C = neural_network_C(bookVector, userVector, userInput, bookInput, 0.00001, 0.2)

# display the model summary for recommender system C
model_C.summary()

# train and fit recommender system C using the training set
history_modelC = model_C.fit([X_train['user_id'].values, X_train['book_id'].values],
                             y_train,
                             epochs = 30,
                             batch_size = 512,
                             validation_split = 0.2, 
                             callbacks = [early_stopping],
                             verbose = 1)



############################### Model Evaluation for Respective Neural Networks ############################

# call the function to plot the loss curve for Recommender System A
loss_curve(history_modelA, 'Recommender System A')

# call the function to plot the MAE curve for recommender system A
mae_curve(history_modelA, 'Recommender System A')

# call the function to plot the RMSE curve for recommender system A
rmse_curve(history_modelA, 'Recommender System A')

# to evaluate the performance of recommender system A based on loss, MSE, and MAE using testing set. 
res_modelA = model_A.evaluate((X_test['user_id'].values,
                            X_test['book_id'].values),
                           y_test,
                           batch_size = 512)



# call the function to plot the loss curve for recommender system B
loss_curve(history_modelB, 'Recommender System B')

# call the function to plot the MAE curve for recommender system B
mae_curve(history_modelB, 'Recommender System B')

# call the function to plot the RMSE curve for recommender system B
rmse_curve(history_modelB, 'Recommender System B')

# to evaluate the performance of recommender system B based on testing set
res_modelB = model_B.evaluate((X_test['user_id'].values,
                            X_test['book_id'].values),
                           y_test,
                           batch_size = 512)



# call the function to plot the loss curve for recommender system C
loss_curve(history_modelC, 'Recommender System C')

# call the function to plot the MAE curve for recommender system C
mae_curve(history_modelC, 'Recommender System C')

# call the function to plot the RMSE curve for recommender system C
rmse_curve(history_modelC, 'Recommender System C')

# evaluate the performance of recommender system C based on testing set
res_modelC = model_C.evaluate((X_test['user_id'].values,
                            X_test['book_id'].values),
                           y_test,
                           batch_size = 512)



################# Optimize the Latent Factor for Best Recommender System (Model B) ###################
# initialize the rmse score for grid-search process
best_rmse_score = 0.4000
# initialize the mae score for grid-search process
best_mae_score = 0.1600

# using a simple grid-search process via for-loops to find the best latent factor based on RMSE and MAE scores
for latent_factor in [5, 10, 20, 30, 40, 50]:

    # define the latent factor
    latent_factor = latent_factor
    
    # call the function to create the embedding path for user and book features
    userInput, userVector, bookInput, bookVector = embedding_path(latent_factor, num_unique_users, num_unique_books)

    # call the function to build the network for the best recommender system B
    model_B = neural_network_B(bookVector, userVector, userInput, bookInput, 0.00001)       
    
    # display the model summary for recommender system B
    model_B.summary()
    
    
    # train and fit the best recommender system B using the training set
    model_B.fit([X_train['user_id'].values, X_train['book_id'].values],
                                  y_train,
                                  epochs = 30,
                                  batch_size = 512,
                                  validation_split = 0.2,
                                  callbacks = [early_stopping],
                                  verbose = 1)
    
    
    # evaluate the bes recommender system B using the testing set
    rmse_score = model_B.evaluate([X_test['user_id'], X_test['book_id']], y_test)[2]
    mae_score = model_B.evaluate([X_test['user_id'], X_test['book_id']], y_test)[1]
    
    
    # if we got a better score, store the score and parameters
    if rmse_score < best_rmse_score and mae_score < best_mae_score:
        best_rmse_score = rmse_score
        best_mae_score = mae_score
        best_parameters = {'latent_factor': latent_factor}


# to display the best RMSE score after the grid-search process
print("\nThe best RMSE score is {:.3f}".format(best_rmse_score))
# to display the best MAE score after the grid-search process
print("\nThe best MAE score is {:.3f}".format(best_mae_score))
# to display the best parameters after the grid-search process
print("\nThe best latent factor is {}\n".format(best_parameters))

# store the best latent factor into Pandas Series.
best_params_df = pd.Series(best_parameters)


#################### Rebuild the Recommender System (Model B) with Best Latent Factor #########################

# define the best latent factor based on the grid-search result.
latent_factor = best_params_df['latent_factor']

# call the function to create the embedding path for user and book features
userInput, userVector, bookInput, bookVector = embedding_path(latent_factor, num_unique_users, num_unique_books)
    
# call the function to build the network for the best recommender system B
remodel_B = neural_network_B(bookVector, userVector, userInput, bookInput, 0.00001)       

# display the model summary for recommender system B
remodel_B.summary()

# retrain and fit the best recommender system B using the training set
history_remodelB = remodel_B.fit([X_train['user_id'].values, X_train['book_id'].values],
                                    y_train,
                                    epochs = 30,
                                    batch_size = 512,
                                    validation_split = 0.2,
                                    callbacks = [early_stopping],
                                    verbose = 1)



#################### Evaluate the Recommender System (Model B) with Best Latent Factor #######################

# call the function to plot the loss curve for recommender system B with the best latent factor
loss_curve(history_remodelB, 'Recommender System B (with best latent factor)')

# call the function to plot the MAE curve for recommender system B with the best latent factor
mae_curve(history_remodelB, 'Recommender System B (with best latent factor)')

# call the function to plot the RMSE curve for recommender system B with the best latent factor
rmse_curve(history_remodelB, 'Recommender System B (with best latent factor)')

# evaluate the performance of recommender system B with the best latent factor based on testing set
res_modelB = remodel_B.evaluate((X_test['user_id'].values,
                            X_test['book_id'].values),
                           y_test, batch_size =512)


################################ Save the Final Recommender System Model #################################################
remodel_B.save(r'C:\Users\jeffr\Desktop\Coursework Final\recomSysModelB.h5')


