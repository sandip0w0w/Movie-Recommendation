import numpy as np
import pandas as pd
import streamlit as st
from rapidfuzz import process
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

#load and preprocess
@st.cache_data
def load_and_preprocess():

    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    ratings = ratings.drop(columns = ['timestamp'])

    return movies, ratings


@st.cache_data
def user_item_matrix(df):

    pivot = df.pivot(index = ['movieId'], columns = ['userId'], values = 'rating').fillna(0)

    #compressing the sparse matrix

    matrix = csr_matrix(pivot.values)

    return pivot, matrix


def model_fitting(matrix):

    #using nearest neighbor  model with cosine similarity

    nn_model = NearestNeighbors(metric = 'cosine', algorithm = 'brute', n_neighbors = 25, n_jobs = -1)

    fitted_model = nn_model.fit(matrix)

    return fitted_model


def movie_recommender_engine(movie_name, pivot, matrix, model, n_recs, movies_df):

    #extract the movie Id
    match = process.extractOne(movie_name, movies_df['title'])
    matched_title = match[0]
    matched_id = movies_df[movies_df['title'] == matched_title]['movieId'].values[0]

    pivot_index = pivot.index.get_loc(matched_id)
    

    # #calculating the neighbor distance
    distances, indices = model.kneighbors(matrix[pivot_index], n_neighbors = n_recs)


    # #list to store the recommendations 

    recommendations = []

    for i in range(1, len(distances.flatten())):

        rec_movie_id = pivot.index[indices.flatten()[i]]

        title = movies_df[movies_df['movieId'] == rec_movie_id]['title'].values[0]

        recommendations.append({'Title': title, 'Distance' : distances.flatten()[i]})
    
    return pd.DataFrame(recommendations), matched_title



movies,rating = load_and_preprocess()
pivot, matrix = user_item_matrix(rating)
model = model_fitting(matrix)
n_recs = 10


st.set_page_config( 
    page_title = "Movie Recommendation System",
    page_icon = "🍿"
)

st.title("Movie Recommendation System - Using Collaborative Filtering")
st.image('https://i.pinimg.com/1200x/3f/84/6b/3f846b82a348a4bb718e88ccd7888dd0.jpg')

st.markdown("""
This project implements a Movie Recommendation System using collaborative filtering techniques.
The system leverages user–item interactions (such as ratings, watch history, or preferences)
to predict and recommend movies that a user is likely to enjoy..
            """)

st.divider()

user_input = st.text_input("movies you watched...")

if st.button("Suggest"):
    new_df, input = movie_recommender_engine(user_input, pivot, matrix, model,n_recs, movies)
    
    st.write(f"Since you watched {user_input}. Here are some recommendations.")
    st.dataframe(new_df)









    
