#import libaries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import top_k_accuracy_score
from sklearn.metrics.pairwise import cosine_similarity


#load dataset
def load_data():
   df = pd.read_csv('netflix_titles.csv')
   movies = df[df['type']=='Movie'].reset_index()
   movies.drop(['index'], axis=1, inplace=True)
   movies['title'] = movies['title'].str.lower()
   movies['show_id'] = [x for x in range(1,6132)]
   return movies


# to remove commas if contain
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(",", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(",", ""))
        else:
            return ''

# create combine features to use in the content-based recommendation process
def preprocess_data(movies_df):
    movies_df.fillna(value = '',inplace = True)
    movies_df['combined_features']= movies_df['director']+" "+movies_df['cast']+ " "+ movies_df['rating']+ " "+ movies_df['listed_in']+ " "+ movies_df['description']
    movies_df['combined_features'] = movies_df['combined_features'].apply(clean_data)
    return movies_df

# calculate cosine_similarity
def transform_data(movies_df):
    cv = CountVectorizer(stop_words='english')
    count_matrix = cv.fit_transform(movies_df['combined_features'])
    cosine_sim = cosine_similarity(count_matrix)
    return cosine_sim

# top5cast = []
# def cast_split(castlist,index):
#     castlist = castlist.fillna(' ')
#     for i in index:
#         if castlist[i]==' ':
#             top5cast.append('No information available')
#         else:
#             top5cast.append(castlist[i].split(',')[:5])
#     return top5cast
    

def recommend_movies(title, movies_df, cosine_sim):
    #create series to get the indices of similar movies
    indices = pd.Series(movies_df.index, index=movies_df['title']).drop_duplicates()

    #get the indices of similar movies
    idx = indices[title]

    # calculate the similarity score
    sim_scores = list(enumerate(cosine_sim[idx]))

    # sorted in decending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # get top 10 highest scores
    sim_scores = sim_scores[1:11]
    
    # get the indices of top 10 similar movies
    movie_indices = [i[0] for i in sim_scores]

    # get the titles
    titles = movies_df['title'].iloc[movie_indices]

    # get the produced countries 
    countries = movies_df['country'].iloc[movie_indices]


    # casts = movies_df['cast'].iloc[movie_indices]
    # get release_years 
    release_years = movies_df['release_year'].iloc[movie_indices]

    # create empty df to put result data
    mov_list = pd.DataFrame()
    # mov_list['id'] = [x for x in range(1,10)] 
    mov_list['Title'] = titles.str.capitalize()
    mov_list['Country'] = countries
    # mov_list['Casts'] = cast_split(casts,index=casts.index)
    mov_list['Release_year'] = release_years

    # return result as a df
    return mov_list