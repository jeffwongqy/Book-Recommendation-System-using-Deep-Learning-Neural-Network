import pandas as pd
import numpy as np
from keras.models import load_model



# define function for user embedding learnt
def user_embedding_learnt(model, embedding_name):
    user_embedding_matrix = model.get_layer(name = embedding_name).get_weights()[0]
    return user_embedding_matrix
    
# define function for movie embedding learnt
def book_embedding_learnt(model, embedding_name):
    book_embedding_matrix = model.get_layer(name = embedding_name).get_weights()[0]
    return book_embedding_matrix
    
# define function to returns the top n most relevant books ids 
def books_recommendation_ids(userID, user_embeddingMatrix, book_embeddingMatrix, num_of_books):
    dot_prod = user_embeddingMatrix[userID]@book_embeddingMatrix.T
    books_ids = np.argpartition(dot_prod, -num_of_books)[-num_of_books: ]
    return books_ids
    


############################################# MAIN ##############################################
# read the preprocessed datafile
preprocessed_df = pd.read_csv(r"C:/Users/jeffr/Desktop/Coursework Final/preprocessing.csv")

# load the keras model
model = load_model(r"C:\Users\jeffr\Desktop\Coursework Final\recomSysModelB.h5")


# display the recommender system menu
print("###########################################################################")
print("#                                                                         #")
print("#            Welcome to the Book Recommender System                       #")
print("#                                                                         #")
print("###########################################################################")
print()
print("What's so special about this recommendation system?")
print("This recommendation system was built with Neural Network Embeddings in Keras.")
print()
print("Instruction to the users: ")
print("------------------------------")
print("1. You are required to key in your preferrence user ID from 0 to 34116 as ")
print("   we have a total of 34117 unique users in our databases. ")
print("2. You need to specified the number of books to be recommended by the system.")

# to initialize the continue parameter
cont = 'Y'


while(cont == 'Y' or cont == 'y'):
    # prompt the user to enter his/ her preference user ID
    userID = input("\nPlease enter your preference User ID (e.g. 2345): ")
    
    # check whether the input user ID is an integer or not
    while(userID.isnumeric() == False):
        userID = input("\nPlease re-enter your preferrence User ID(e.g. 2345): ")
        print()
    
    # check whether the input user ID is within the range of 0 to 34116
    while(int(userID) < 0 or int(userID) >= 34116):
        userID = input("\nPlease re-enter your preferrence User ID in the range of 0 to 34116: ")
        print()
    
    # convert the user input from string into integer
    userID = int(userID)
    
    # prompt the user to specified the number of books to be recommended by the system
    num_of_books = int(input("\nPlease enter the number of books to be recommended by the system (e.g. 5): "))
    
    # call the function to generate the embedding matrix for user 
    user_embeddingMatrix = user_embedding_learnt(model, 'non_neg_user_embedding')
    
    # call the function to generate the embedding matrix for books
    book_embeddingMatrix = book_embedding_learnt(model, 'non_neg_book_embedding')
    
    # call the function to perform dot products on user and book embedding matrices
    # and then get the top 10 most relevant books ids 
    books_IDs = books_recommendation_ids(userID, user_embeddingMatrix, book_embeddingMatrix, num_of_books)
    
    # get the book information from the dataset based on the relevant books IDs
    list_recommended_books = list()
    for i in range(num_of_books):
        book_title = preprocessed_df[preprocessed_df['book_id'] == books_IDs[i]][['book_id', 
                                                                                  'book_title',
                                                                                  'book_author',
                                                                                  'year_of_publication',
                                                                                  'publisher']]
        
        list_recommended_books.append(book_title.iloc[0])
    
    # save the list of recommended books into a new DataFrame
    recommended_books_df = pd.DataFrame(list_recommended_books).reset_index(drop = True)
    
    # display the top n most relevant books to the user interface
    print("\nBased on your specified User ID '{}', here are the top {} relevant recommended books: ".format(userID, num_of_books))
    print("--------------------------------------------------------------------------------------------")
    print()
    pd.set_option('display.max_columns', None)  
    print(recommended_books_df)
    
    # prompt the user whether would like to continue with the system
    cont = input("\nWould you like to continue to use the system? <Y/N>: ")

# display the end message 
print("\nThank you for using the book recommender system! Goodbye!")

