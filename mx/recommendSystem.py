import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


df = pd.read_csv('dataset.csv')
listTitle = df['sortedTitle'].tolist()
countVectorizer = CountVectorizer(dtype=np.uint8)                               #Convert a collection of text documents to a matrix of token counts#
transformed_m = countVectorizer.fit_transform(df['genres']).toarray()

#-1 in reshape function is used when you dont know or want to explicitly tell the dimension of that axis.
matrix = np.concatenate((transformed_m, np.array(df['averageRating']).reshape(-1, 1)), axis=1)

#A way to normalize the input features/variables is the Min-Max scaler. By doing so, all features will be transformed into the range [0,1]
minMaxScaler = MinMaxScaler()

no_of_votes = np.array(df['numVotes'])
no_of_votes = no_of_votes.reshape(-1, 1)
no_of_votes = minMaxScaler.fit_transform(no_of_votes)
matrix = np.concatenate((matrix, no_of_votes), axis=1)

#Cosine similarity is the cosine of the angle between two n-dimensional vectors in an n-dimensional space. 
#It is the dot product of the two vectors divided by the product of the two vectors' lengths (or magnitudes).

#Cosine similarity, computes similarity as the normalized dot product of features of the provided matrix:
sim = cosine_similarity(matrix, dense_output=False)


def recommendationBuild(movieTitle):
    #try:
        movieTitle = movieTitle.lower()
        

        #checking whether the movie title matches with any of the movies in our dataset
        foundSortedTitle = True in [True for T in df['sortedTitle'] if T.lower() == movieTitle]  
        if foundSortedTitle:                                            
            index = df[df['sortedTitle'].apply(lambda X: X.lower()) == movieTitle].index[0]         #matching the movie title with the movie title in dataset if found
        else:
            index = df[df['primaryTitle'].apply(lambda X: X.lower()) == movieTitle].index[0]
            

        #numpy.argsort() function is used to perform an indirect sort along the given axis using the algorithm specified by the kind keyword.
        #It returns an array of indices of the same shape as arr that that would sort the array.
        #taking all movies that can be recommended based on the similarity matrix and arranging them in descending order
        recommendation = df['sortedTitle'].iloc[sim[index].argsort()[::-1]]

        
        #selecting the top 10 movies and returning the dataframe with 2 columns (tconst and sortedTitle)
        recommendMovie = {r: [df['tconst'].iloc[r], df['sortedTitle'].iloc[r]] for r in
                                     recommendation.index }
        return pd.DataFrame(recommendMovie).transpose().iloc[1:11]
    #except:
     #   return None


def recommendGET(movieTitle):
    recommendation = recommendationBuild(movieTitle)
    if recommendation is None:
        return recommendation
    else:
        recommendation.rename(columns={0: 'tconst', 1: 'title'}, inplace=True)
        recommendation.reset_index(drop=True, inplace=True)
        #tconst (string) - alphanumeric unique identifier of the title.
        recommendation['urls'] = [title_id for title_id in recommendation['tconst']]
        return recommendation.drop('tconst', axis=1)


def get_movie_data():
    return listTitle
