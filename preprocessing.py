import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate, Dropout
from tensorflow.keras.models import  Model
from tensorflow.keras.optimizers import Adam, Adamax, RMSprop, SGD
from keras.regularizers import l2
from keras.callbacks import EarlyStopping



######################################## Import Datasets #######################################
# read the books dataset
books_df = pd.read_excel(r"C:\Users\jeffr\Desktop\Coursework Final\datatsets\BX-Books.xlsx")
# read the users dataset
users_df = pd.read_csv(r"C:\Users\jeffr\Desktop\Coursework Final\datatsets\BX-Users.csv", delimiter = ';')
# read the book ratings dataset
book_ratings_df = pd.read_excel(r"C:\Users\jeffr\Desktop\Coursework Final\datatsets\BX-Book-Ratings.xlsx")
                    
   

################################### Display the Datasets ###############################
# display the top 5 rows of the books dataset
print("Top 5 Rows of the Books Dataset")
print("=================================================================================")
print(books_df.head(5))
print()

# display the top 5 rows of the users dataset
print("Top 5 Rows of the Users Dataset")
print("=================================================================================")
print(users_df.head(5))
print()

# display the top 5 rows of the book ratings dataset
print("Top 5 Rows of the Book Ratings Dataset")
print("=================================================================================")
print(book_ratings_df.head(5))
print()



################################ Data Preprocessing on Books ################################
# print the data info for books before preprocessing 
print(books_df.info())

# before changing the column names in the books dataframe
print(books_df.head())
# convert all the column name into lower cases
books_df.columns = books_df.columns.str.lower()
# convert all the column name with hyphen "-" into underscore "_" by using the replace () function
books_df.columns = books_df.columns.str.replace('-', '_')
# after changing the column names in the books dataframe
print(books_df.head())



# before drop down the irrelevant url columns
print(books_df.head())
# drop the irrelevant url-link columns
books_df.drop(columns = ['image_url_s', 'image_url_m', 'image_url_l'], inplace = True)
# after drop down the irrelevant url columns 
print(books_df.head())



# before changing the data type for year of publication column 
print(books_df.info())
#convert the year of publication column into float instead of object type
books_df['year_of_publication'] = pd.to_numeric(books_df['year_of_publication'], errors = 'coerce')
# after changing the data type for year of publication column
print(books_df.info())




# count the number of empty string, zeo, and null values in the year of publications column before replacing with mean values
num_of_zeros = books_df[books_df['year_of_publication'] == 0]['year_of_publication'].count()
num_of_empty = books_df[books_df['year_of_publication'] == ' ']['year_of_publication'].count()
num_of_nan = books_df['year_of_publication'].isnull().sum()

# display the count the number of empty string, zero, and null values in the year of publication column before replaicng with mean values
print("Number of empty strings in the Year of Publication column: {}".format(num_of_empty))
print("Number of zero values in the Year of Publication column: {}".format(num_of_zeros))
print("Number of NaN values in the Year of Publication column: {}".format(num_of_nan))
print()

# replace all the years with zero values with NaN with replace () function
books_df['year_of_publication'].replace(0, np.nan, inplace = True)
# replace all the years with NaN values with mean value using fillna() function 
books_df['year_of_publication'].fillna(books_df['year_of_publication'].mean(), inplace = True)

# count the number of empty string, zeo, and null values in the year of publications column after replacing with mean values
num_of_zeros = books_df[books_df['year_of_publication'] == 0]['year_of_publication'].count()
num_of_empty = books_df[books_df['year_of_publication'] == ' ']['year_of_publication'].count()
num_of_nan = books_df['year_of_publication'].isnull().sum()

# display the count the number of empty string, zero, and null values in the year of publication column after replacing with mean values
print("Number of empty strings in the Year of Publication column: {}".format(num_of_empty))
print("Number of zero values in the Year of Publication column: {}".format(num_of_zeros))
print("Number of NaN values in the Year of Publication column: {}".format(num_of_nan))
print()



# convert the year of publication data type from float to int
books_df['year_of_publication'] = books_df['year_of_publication'].astype('int')



# investigate the old books and future books based on year of publication
old_books = books_df[books_df['year_of_publication'] < 1900]
future_books = books_df[books_df['year_of_publication'] > 2021]

# display the list of the old books and future books
print("List of the Old Books based on the Year of Publication before 1900: ")
print("====================================================================================")
print(old_books)
print()
print("List of the Future Books based on the Year of Publication after 2021: ")
print("====================================================================================")
print(future_books.head())
print()

# old books tends to skew the model and seems to be irrelevant in this context, therefore drop the old books
print("Length of books before removal: {}".format(len(books_df)))
books_df = books_df.loc[~(books_df.isbn.isin(old_books.isbn))]
books_df = books_df.loc[~(books_df.isbn.isin(future_books.isbn))]
print("Length of books after removal: {}".format(len(books_df)))
print()



# replace the ampersand formmating in the Publisher data column with &
print("Before replacing the ampersand formatting: ")
print("====================================================================================")
print(books_df.head())
print()
books_df['publisher'] = books_df['publisher'].str.replace('&amp;', '&', regex = False)
print("After replacing the ampersand formatting: ")
print("====================================================================================")
print(books_df.head())
print()



# investigate the number of duplicate books records
num_of_unique_books = books_df.isbn.nunique()
num_of_books = books_df.isbn.count()




# count the number of empty strings, zero values and NaN values in the publisher column before removing the NaN
empty_values_publisher = books_df[books_df['publisher'] == '']['publisher'].count()
zero_values_publisher = books_df[books_df['publisher'] == 0]['publisher'].count()
null_values_publisher = books_df['publisher'].isnull().sum()

# display the number of empty strings, zero values and NaN values in the publisher column before removing the NaN 
print("Number of empty strings in the Publisher column: {}".format(empty_values_publisher))
print("Number of zero values in the Publisher column: {}".format(zero_values_publisher))
print("Number of NaN values in the Publisher column: {}".format(null_values_publisher))

# remove the rows that with NaN values that subset with publisher column
books_df = books_df.dropna(subset = ['publisher'])

# count the number of empty strings, zero values and NaN values in the publisher column after removing the NaN
empty_values_publisher = books_df[books_df['publisher'] == '']['publisher'].count()
zero_values_publisher = books_df[books_df['publisher'] == 0]['publisher'].count()
null_values_publisher = books_df['publisher'].isnull().sum()

# display the number of empty strings, zero values and NaN values in the publisher column after removing the NaN 
print("Number of empty strings in the Publisher column: {}".format(empty_values_publisher))
print("Number of zero values in the Publisher column: {}".format(zero_values_publisher))
print("Number of NaN values in the Publisher column: {}".format(null_values_publisher))





# count the number of empty strings, zero values, and NaN values in the book-author column before removing the NaN
empty_values_book_author = books_df[books_df['book_author'] == '']['book_author'].count()
zero_values_book_author = books_df[books_df['book_author'] == 0]['book_author'].count()
null_values_book_author = books_df['book_author'].isnull().sum()

# count the number of empty strings, zero values, and NaN values in the book-author column before removing the NaN
print("Number of empty strings in the Book Author column: {}".format(empty_values_book_author))
print("Number of zero values in the Book Author column: {}".format(zero_values_book_author))
print("Number of NaN values in the Book Author column: {}".format(null_values_book_author))
print()

# remove the rows that with NaN values that subset with book-author column
books_df = books_df.dropna(subset = ['book_author'])

# count the number of empty strings, zero values, and NaN values in the book-author column  after removing the NaN
empty_values_book_author = books_df[books_df['book_author'] == '']['book_author'].count()
zero_values_book_author = books_df[books_df['book_author'] == 0]['book_author'].count()
null_values_book_author = books_df['book_author'].isnull().sum()

# count the number of empty strings, zero values, and NaN values in the book-author column after removing the NaN
print("Number of empty strings in the Book Author column: {}".format(empty_values_book_author))
print("Number of zero values in the Book Author column: {}".format(zero_values_book_author))
print("Number of NaN values in the Book Author column: {}".format(null_values_book_author))
print()




# display the data info and dataframe for book after pre-processing
print(books_df.info())
print()

# create a new copy of the book dataframe after pre-processing
book_df_preprocessed = books_df




#################################  Data Preprocessing for Users ##############################
# print the data info for users before preprocessing 
print(users_df.info())
print()



# before changing the column names in the users dataframe 
print(users_df.head())
# convert all the column name into lower cases
users_df.columns = users_df.columns.str.lower()
# convert all the column name with dash "-" into underscore "_" by using the replace () function
users_df.columns = users_df.columns.str.replace('-', '_')
# after changing the column names in the users dataframe
print(users_df.head())



# investigate the number of duplicate books records
num_of_unique_users = users_df.user_id.nunique()
num_of_users = users_df.user_id.count()




# set null values for age group before 5 years old and above 100 years old 
print("Before setting null values for age group before 5 years old and above 100 years old")
print("==========================================================================================")
print(users_df.age.unique())
print()
print("After setting null values for age group before 5 years old and above 100 years old")
print("==========================================================================================")
users_df.loc[(users_df['age'] < 5) | (users_df['age'] > 100)] = np.nan
print(users_df.age.unique())
print()




# create a plot to investigate the distribution of the age group
sns.distplot(users_df['age'], bins = 50, color = 'crimson', kde = False)
plt.title("Distribution of the User Age Group", fontsize = 14, fontweight = 'bold')
plt.xlabel("Age", fontsize = 12, fontweight = 'bold')
plt.ylabel("Number of Entries", fontsize = 12, fontweight = 'bold')
plt.show()




# count the number of empty strings, zero values, and NaN values before removing the NaN
empty_values_age = users_df[users_df['age'] == '']['age'].count()
zero_values_age = users_df[users_df['age'] == 0]['age'].count()
null_values_age = users_df['age'].isnull().sum()

# display the total number of empty strings, zero values, and NaN values before removing the NaN
print("Number of empty strings in the User Age column: {}".format(empty_values_age))
print("Number of zero values in the User Age column: {}".format(zero_values_age))
print("Number of NaN values in the User Age column: {}".format(null_values_age))
print()

# remove the rows with NaN that subset with the age columns 
users_df = users_df.dropna(subset = ['age'])

# count the number of empty strings, zero values, and NaN values after removing the NaN
empty_values_age = users_df[users_df['age'] == '']['age'].count()
zero_values_age = users_df[users_df['age'] == 0]['age'].count()
null_values_age = users_df['age'].isnull().sum()

# display the total number of empty strings, zero values, and NaN values after removing the NaN
print("Number of empty strings in the User Age column: {}".format(empty_values_age))
print("Number of zero values in the User Age column: {}".format(zero_values_age))
print("Number of NaN values in the User Age column: {}".format(null_values_age))
print()




# before expanding the location column into city, state, and country columns
print("Before expanding the location column into city, state, and country columns ")
print("=========================================================================================")
print(users_df.head())
print()

users_df_location = users_df['location'].str.split(',', 2, expand = True)
users_df_location.columns = ['city', 'state', 'country']
users_df = users_df.join(users_df_location)

print("After expanding the location column into city, state, and country columns ")
print("=========================================================================================")
print(users_df.head())
print()




# count the number of empty string, zero values and null values in the city column before removing NaN
empty_values_city = users_df[users_df['city'] == '']['city'].count()
zero_values_city = users_df[users_df['city'] == 0]['city'].count()
null_values_city = users_df['city'].isnull().sum()

# display the number of empty string, zero values and null values in the city column before removing NaN
print("Number of empty strings in the City column: {}".format(empty_values_city))
print("Number of zero values in the City column: {}".format(zero_values_city))
print("Number of NaN values in the City column: {}".format(null_values_city))
print()

# replace the empty strings with NaN values 
users_df['city'].replace('', np.nan, inplace = True)
# remove the rows consists of NaN that subsets with city column 
users_df = users_df.dropna(subset = ['city'])

# count the number of empty string, zero values and null values in the city column after removing NaN
empty_values_city = users_df[users_df['city'] == '']['city'].count()
zero_values_city = users_df[users_df['city'] == 0]['city'].count()
null_values_city = users_df['city'].isnull().sum()

# display the number of empty string, zero values and null values in the city column after removing NaN
print("Number of empty strings in the City column: {}".format(empty_values_city))
print("Number of zero values in the City column: {}".format(zero_values_city))
print("Number of NaN values in the City column: {}".format(null_values_city))
print()





# count the number of empty string, zero values, and null values in the state column before removing NaN
empty_values_state = users_df[users_df['state'] == '']['state'].count()
zero_values_state = users_df[users_df['state'] == 0]['state'].count()
null_values_state = users_df['state'].isnull().sum()

# display the total number of empty string, zero values, and null values in the state column before removing NaN
print("Number of empty strings in the State column: {}".format(empty_values_state))
print("Number of zero values in the State column: {}".format(zero_values_state))
print("Number of NaN values in the State column: {}".format(null_values_state))
print()

# replace the empty strings with NaN values 
users_df['state'].replace('', np.nan, inplace = True)
# remove the rows consists of NaN that subsets with state column 
users_df = users_df.dropna(subset = ['state'])

# count the number of empty string, zero values, and null values in the state column after removing NaN
empty_values_state = users_df[users_df['state'] == '']['state'].count()
zero_values_state = users_df[users_df['state'] == 0]['state'].count()
null_values_state = users_df['state'].isnull().sum()

# display the total number of empty string, zero values, and null values in the state column after removing NaN
print("Number of empty strings in the State column: {}".format(empty_values_state))
print("Number of zero values in the State column: {}".format(zero_values_state))
print("Number of NaN values in the State column: {}".format(null_values_state))
print()




# count the number of empty string, zero values, and null values in the country column before removing NaN
empty_values_country = users_df[users_df['country'] == '']['country'].count()
zero_values_country = users_df[users_df['country'] == 0]['country'].count()
null_values_country = users_df['country'].isnull().sum()

# display the total number of empty string, zero values, and null values in the country column before removing NaN
print("Number of empty strings in the Country column: {}".format(empty_values_country))
print("Number of zero values in the Country column: {}".format(zero_values_country))
print("Number of null values in the Country column: {}".format(null_values_country))
print()

# replace the empty strings with NaN values 
users_df['country'].replace('', np.nan, inplace = True)
# remove the rows consists of NaN that subsets with country column 
users_df = users_df.dropna(subset = ['country'])

# count the number of empty string, zero values, and null values in the country column after removing NaN
empty_values_country = users_df[users_df['country'] == '']['country'].count()
zero_values_country = users_df[users_df['country'] == 0]['country'].count()
null_values_country = users_df['country'].isnull().sum()

# display the total number of empty string, zero values, and null values in the country column after removing NaN
print("Number of empty strings in the Country column: {}".format(empty_values_country))
print("Number of zero values in the Country column: {}".format(zero_values_country))
print("Number of null values in the Country column: {}".format(null_values_country))
print()




# investigate the number of empty , 0 or NaN values in the location column
empty_values_location = users_df[users_df['location'] == '']['location'].count()
zero_values_location = users_df[users_df['location'] == 0]['location'].count()
null_values_location = users_df['location'].isnull().sum()
print("Number of Empty String in the Location column: {}".format(empty_values_location))
print("Number of Zero Values in the Location column: {}".format(zero_values_location))
print("Number of Null Values in the Location column: {}".format(null_values_location))
print()



# before changing the data type for age column 
print(users_df.info())
# convert the age column from float into int
users_df['age'] = users_df['age'].astype('int')
# after changing the data type for age column 
print(users_df.info())


# create a new copy of the users dataframe after pre-processing
users_df_preprocessed = users_df.copy()




#################################  Data Preprocessing for Book Ratings ##############################
# print the data info for book ratings before preprocessing 
print(book_ratings_df.info())
print()



# before changing the column names in the users dataframe 
print(book_ratings_df.head())
# convert all the column name into lower cases
book_ratings_df.columns = book_ratings_df.columns.str.lower()
# convert all the column name with hyphens "-" into underscore "_" by using the replace () function
book_ratings_df.columns = book_ratings_df.columns.str.replace('-', '_')
# before changing the column names in the users dataframe 
print(book_ratings_df.head())




# display the distribution of the users with less than 100 book ratings
most_users = book_ratings_df.groupby('user_id').isbn.count().sort_values(ascending = False)
user_dist_less_than_100 = most_users.where(most_users < 100)
sns.set_style('darkgrid')
sns.histplot(data = user_dist_less_than_100, bins = 10, color = 'navy')
plt.title("Distribution of the User with Less Than 100 Book Ratings", fontsize = 14, fontweight = 'bold')
plt.xlabel("Number of Ratings", fontsize = 12, fontweight = 'bold')
plt.ylabel("Number of Count", fontsize = 12, fontweight = 'bold')
plt.show()



# display the distribution of the users with more than 1000 book ratings
most_users = book_ratings_df.groupby('user_id').isbn.count().sort_values(ascending = False)
user_dist_more_than_1000 = most_users.where(most_users >= 1000)
sns.set_style('darkgrid')
sns.histplot(data = user_dist_more_than_1000, bins = 10,  color = 'crimson')
plt.title("Distribution of the User with More Than 1000 Book Ratings", fontsize = 14, fontweight = 'bold')
plt.xlabel("Number of Ratings", fontsize = 12, fontweight = 'bold')
plt.ylabel("Number of Count", fontsize = 12, fontweight = 'bold')
plt.show()



# display the distribution of book ratings (before removing the implicit rating)
ratings = book_ratings_df['book_rating'].value_counts().sort_index()
sns.barplot(x = ratings.index, y = ratings.values)
plt.title("Distribution of the Book Ratings (Before removing implicit rating)", fontsize = 14, fontweight = 'bold')
plt.xlabel("Rating", fontsize = 12, fontweight = 'bold')
plt.ylabel("Number of Count", fontsize = 12, fontweight = 'bold')
plt.show()

# As zero indicates an implicit rating, therefore it will be removes from the data.
# As such, it will be focusing more on the explicit ratings instead.
print("Number of rows in book ratings before removing zero ratings: {}".format(len(book_ratings_df)))
# remove the implicit ratings from the dataframe
book_ratings_df = book_ratings_df[book_ratings_df['book_rating'] != 0]
print("Number of rows in book ratings after removing zero ratings: {}".format(len(book_ratings_df)))

# display the distribution of book ratings (after removing the implicit rating)
ratings = book_ratings_df['book_rating'].value_counts().sort_index()
sns.barplot(x = ratings.index, y = ratings.values)
plt.title("Distribution of the Book Ratings (After removing explicit rating)", fontsize = 14, fontweight = 'bold')
plt.xlabel("Rating", fontsize = 12, fontweight = 'bold')
plt.ylabel("Number of Count", fontsize = 12, fontweight = 'bold')
plt.show()



# create a new book ratings dataframe after preprocessing
book_ratings_df_preprocessed = book_ratings_df.copy()




########################  Data Preprocessing on Combining Users, Books, Book Ratings ##############################

##### Joining the Preprocessed Books DataFrame with Preprocessed Book Ratings
# print the number of rows in the preprocessed book dataframe 
print("Number of rows in the Preprocessed Book DataFrame: {}".format(len(book_df_preprocessed)))
# print the number of rows in the preprocessed book ratings dataframe
print("Number of rows in the Preprocessed Book Ratings DataFrame: {}".format(len(book_ratings_df_preprocessed)))
# join the preprocssed book table to the preprocessed book rating table based on the ISBN
bk_with_ratings_df = book_ratings_df_preprocessed.join(book_df_preprocessed.set_index('isbn'), on = 'isbn')
# print the number of rows in the combination of books and book ratings dataframe
print("Number of rows in the New DataFrame (Books + Books-Ratings): {}".format(len(bk_with_ratings_df)))
# display the new book with rating dataframe
print(bk_with_ratings_df.head())



# count the number of book title with NaN values before removing NaN
print("Before removing all the NaN values, the number of Book Title with NaN values is {}".format(bk_with_ratings_df['book_title'].isnull().sum()))
# remove the rows consists of NaN that subset with the book title column 
bk_with_ratings_df.dropna(subset = ['book_title'], inplace = True)
# count the number of book title with NaN values removed
print("After removing all the NaN values, the number of Book Title with NaN values is {}".format(bk_with_ratings_df['book_title'].isnull().sum()))




### to display the top 20 books title with the highest average ratings
val_count = bk_with_ratings_df['book_title'].value_counts()
# to compute find the books with at least 100 ratings received 
avg_ratings = bk_with_ratings_df[bk_with_ratings_df['book_title'].isin(val_count[val_count>50].index)].groupby('book_title')['book_rating'].mean()
print("The top 20 books with the highest average ratings are: ")
print("====================================================================================")
print(avg_ratings.sort_values(ascending = False)[:20])


### to display the bottom 20 books title with the lowest average ratings
print("The bottom 20 books with the lowest average ratings are: ")
print("====================================================================================")
print(avg_ratings.sort_values(ascending = False)[-20:])




##### Joining the Preprocessed User DataFrame with New DataFrame (bk_with_ratings_df)
# print the number of rows in the preprocessed user dataframe 
print("Number of rows in the Preprocessed User DataFrame: {}".format(len(users_df_preprocessed)))
# print the number of rows in the new table (books + book_ratings)
print("Number of Rows in the New DataFrame (bk_with_ratings_df): {}".format(len(bk_with_ratings_df)))
# print the number of rows in the new table (books + book_ratings + users)
bookUserRatings_df = bk_with_ratings_df.join(users_df_preprocessed.set_index('user_id'), on = 'user_id')
print("Number of Rows in the New DataFrame (User + bk_with_ratings_df): {}".format(len(bookUserRatings_df)))




# count the number of empty string, zero values, and null values in the location column before removing NaN
empty_values_location = bookUserRatings_df[bookUserRatings_df['location'] == '']['location'].count()
zero_values_location = bookUserRatings_df[bookUserRatings_df['location'] == 0]['location'].count()
null_values_location = bookUserRatings_df['location'].isnull().sum()

# display the total number of empty string, zero values, and null values in the location column before removing NaN
print("Number of empty strings in the Location column: {}".format(empty_values_location))
print("Number of zero values in the Location column: {}".format(zero_values_location))
print("Number of null values in the Location column: {}".format(null_values_location))
print()

# remove the rows consists of NaN that subsets with country column 
bookUserRatings_df = bookUserRatings_df.dropna(subset = ['location'])

# count the number of empty string, zero values, and null values in the location column after removing NaN
empty_values_location = bookUserRatings_df[bookUserRatings_df['location'] == '']['location'].count()
zero_values_location = bookUserRatings_df[bookUserRatings_df['location'] == 0]['location'].count()
null_values_location = bookUserRatings_df['location'].isnull().sum()

# display the total number of empty string, zero values, and null values in the location column after removing NaN
print("Number of empty strings in the Location column: {}".format(empty_values_location))
print("Number of zero values in the Location column: {}".format(zero_values_location))
print("Number of null values in the Location column: {}".format(null_values_location))
print()


# before changing the data type for age column 
print(bookUserRatings_df.info())
# convert the data type for age column from float into int
bookUserRatings_df['age'] = bookUserRatings_df['age'].astype('int')
# after changing the data type for age column 
print(bookUserRatings_df.info())


##### create a new columns for encoded user_id and isbn 
bookUserRatings_df['user_id'] = bookUserRatings_df['user_id'].astype('category').cat.codes.values
bookUserRatings_df['book_id'] = bookUserRatings_df['isbn'].astype('category').cat.codes.values


# display the new table (bookUserRatings_df)
print(bookUserRatings_df.info())

# export the preprocessed bookUserRatings dataframe into csv file 
bookUserRatings_df.to_csv(r"C:\Users\jeffr\Desktop\Coursework Final\preprocessing.csv", header = True, index = False)
















