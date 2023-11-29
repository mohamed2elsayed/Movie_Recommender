from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process

class MovieRecommender:
    def __init__(self):
        self.df1 = pd.read_csv('tmdb_5000_credits.csv')
        self.df2 = pd.read_csv('tmdb_5000_movies.csv')
        self.df1.columns = ['id', 'tittle', 'cast', 'crew']
        self.df2 = self.df2.merge(self.df1, on='id')
        self.indices = pd.Series(self.df2.index, index=self.df2['title']).drop_duplicates()
        self.features = ['cast', 'crew', 'keywords', 'genres']
        for feature in self.features:
            self.df2[feature] = self.df2[feature].apply(literal_eval)
        self.df2['director'] = self.df2['crew'].apply(self.get_director)
        self.features = ['cast', 'keywords', 'genres']
        for feature in self.features:
            self.df2[feature] = self.df2[feature].apply(self.get_list)
        self.features = ['cast', 'keywords', 'director', 'genres']
        for feature in self.features:
            self.df2[feature] = self.df2[feature].apply(self.clean_data)
        self.df2['soup'] = self.df2.apply(self.create_soup, axis=1)
        count = CountVectorizer(stop_words='english')
        count_matrix = count.fit_transform(self.df2['soup'])
        self.cosine_sim = cosine_similarity(count_matrix, count_matrix)
        self.df2 = self.df2.reset_index()
        self.indices = pd.Series(self.df2.index, index=self.df2['title'])

    def get_director(self, x):
        for i in x:
            if i['job'] == 'Director':
                return i['name']
        return np.nan

    def get_list(self, x):
        if isinstance(x, list):
            names = [i['name'] for i in x]
            if len(names) > 3:
                names = names[:3]
            return names
        return []

    def clean_data(self, x):
        if isinstance(x, list):
            return [str.lower(i.replace(" ", "")) for i in x]
        else:
            if isinstance(x, str):
                return str.lower(x.replace(" ", ""))
            else:
                return ''

    def create_soup(self, x):
        return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

    def get_recommendations(self, title, cosine_sim=None):
        if cosine_sim is None:
            cosine_sim = self.cosine_sim
        idx = self.indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:201]
        movie_indices = [i[0] for i in sim_scores]
        return self.df2[['title', 'release_date', 'vote_average', 'genres']].iloc[movie_indices].reset_index().drop(
            columns=['index'])

    def user_input_recommendations_web(self, title, num_recommendations):
        try:
            if len(title) <= 3:
                return "Please enter a title correctly"

            all_titles = self.df2['title'].tolist()
            closest_match = process.extractOne(title, all_titles)
            recommendations = self.get_recommendations(closest_match[0], self.cosine_sim)

            return recommendations.head(num_recommendations).to_dict('records')
        except Exception as e:
            return f"An error occurred: {e}"


app = Flask(__name__)
movie_recommender = MovieRecommender()

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/recommendations')
def get_recommendations():
    title = request.args.get('title')
    num_recommendations = 100  # You can adjust this based on your preference
    recommendations = movie_recommender.user_input_recommendations_web(title, num_recommendations)
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
