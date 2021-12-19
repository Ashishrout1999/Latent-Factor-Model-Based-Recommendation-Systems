# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 14:32:42 2021

@author: 91960
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv


# Read the learned parameters of the best model
#mean_rating = np.float(pd.read_csv('../Weights/Mean_Rating/Weight_lr0.01_reg0.2_factor40_epoch100.csv', index_col = 0).to_numpy())
#user_bias = pd.read_csv('../Weights/User_Bias/Weight_lr0.01_reg0.2_factor40_epoch100.csv', index_col = 0).to_numpy().flatten()
#item_bias = pd.read_csv('../Weights/Item_Bias/Weight_lr0.01_reg0.2_factor40_epoch100.csv', index_col = 0).to_numpy().flatten()
user_factor = pd.read_csv('../Weights/User_Factor/Weight_lr0.01_reg0.2_factor10_epoch100.csv', index_col = 0).to_numpy()
item_factor = pd.read_csv('../Weights/Item_Factor/Weight_lr0.01_reg0.2_factor10_epoch100.csv', index_col = 0).to_numpy()


# Data used during training of these models
data = pd.read_csv('../Data/u.data', delimiter = '\t', names = ['User', 'Item', 'Rating', 'Time'])

num_users = len(pd.unique(data['User']))
num_items = len(pd.unique(data['Item']))

train_data = pd.read_csv('../Data_Split/train.csv').to_numpy()[:, 1 :]
val_data = pd.read_csv('../Data_Split/val.csv').to_numpy()[:, 1 :]
test_data = pd.read_csv('../Data_Split/test.csv').to_numpy()[:, 1:]


# Movie Index - Name Mapping
movie_indices = []
movie_name = []
movie_file = open('../Data/u.item')
csv_file = csv.reader(movie_file, delimiter = '|')

for row in csv_file:
    movie_indices.append(int(row[0]) - 1)
    movie_name.append(row[1])

movie_name = np.array(movie_name)


# Explanation for a particular prediction
user_index = 900
num_recos = 10
movie_indices_train_user = train_data[train_data[:, 0] == user_index][:, 1]

best_movie_indices_train_user = train_data[(train_data[:, 0] == user_index) & (train_data[:, 2] >= 5)]
best_movies_training = movie_name[best_movie_indices_train_user[:, 1]]


threshold_num_train_ratings = 10
train_movie_indices, num_train_movie_count = np.unique(train_data[:, 1], return_counts = True)
train_movies_below_threshold = train_movie_indices[np.argwhere(num_train_movie_count <= threshold_num_train_ratings).flatten()]

items_to_del_test = np.array(list(set(np.concatenate((movie_indices_train_user, train_movies_below_threshold)))))

movie_indices_test_user = np.delete(np.arange(num_items), items_to_del_test)


#movie_indices_test_user = np.delete(np.arange(num_items), movie_indices_train_user)
movie_rating_predict_user = np.dot(item_factor, user_factor[user_index])
movie_rating_test_predict_user = movie_rating_predict_user[movie_indices_test_user]
top_movies_user_local = np.argsort(movie_rating_test_predict_user)[-num_recos : ][::-1]
top_movies_user = movie_indices_test_user[top_movies_user_local]

top_movies_name_user = movie_name[top_movies_user]
top_rating_user = movie_rating_test_predict_user[top_movies_user_local]

print('Movies rated 5/5 by User {}'.format(user_index))
print('\n')
print(best_movies_training)
print('\n\n')
print('Movies Suggested By Recommender System')
print('\n')
print(top_movies_name_user)



