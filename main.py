import streamlit as st
import pickle
import pandas as pd

movie_list = pickle.load(open('movies_dict.pkl', 'rb'))
movie=pd.DataFrame(movie_list)

st.title("Movie Recommender System")
similarity = pickle.load(open('similarity.pkl', 'rb'))

def recommend(movie):
    movie_index=movie[movie['title'] == select_movie_name].index[0]
    distance=similarity[movie_index]
    movie_list= sorted(enumerate(distance), key=lambda x: x[1], reverse=True)[1:6]
    
    recommended_movies=[]
    for i in movie_list:
        recommended_movies.append(movie.iloc[i[0]].title)
    return recommended_movies

select_movie_name=st.selectbox(
    "Select movies",movie['title'].values)
if st.button("Recommend"):
    recommendations=recommend(movie)
    for rec in recommendations:
        st.write(rec)