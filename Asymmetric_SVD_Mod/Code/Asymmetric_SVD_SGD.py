# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 14:11:07 2021

@author: ashish
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections
import time



import os
if not os.path.exists('../Weights'):
    os.makedirs('../Weights')
    os.makedirs('../Weights/Mean_Rating')
    os.makedirs('../Weights/User_Bias')
    os.makedirs('../Weights/Item_Bias')
    os.makedirs('../Weights/Explicit_Factor')
    os.makedirs('../Weights/Item_Factor')
    os.makedirs('../Weights/Implicit_Factor')
    
    
if not os.path.exists('../Results'):
    os.makedirs('../Results/Test_Error')
    os.makedirs('../Results/Train')
    os.makedirs('../Results/Validate')
    os.makedirs('../Results/Figure')




data = pd.read_csv('../Data/u.data', delimiter = '\t', names = ['User', 'Item', 'Rating', 'Time'])

# Timestamp is not required for our purpose.
data = data.drop(columns = ['Time'])

# Subtracting 1 from all entries so that user id and item id can be used as index.
data['User'] = data['User'] - 1
data['Item'] = data['Item'] - 1
#print(data['User'].min(), data['User'].max(), data['Item'].min(), data['Item'].max())

num_users = len(pd.unique(data['User']))
num_items = len(pd.unique(data['Item']))
'''
# Splitting data into train, validation and test set
def split_data(data, splits):
    data = np.array(data)
    splits = np.array(splits)
    num_data = data.shape[0]
    rng = np.random.default_rng(seed = 4)
    rng.shuffle(data)
    num_train, num_val, num_test = (splits * num_data).astype(int)
    train_data = data[ : num_train]
    val_data = data[num_train : num_train + num_val]
    test_data = data[num_train + num_val : ]
    return train_data, val_data, test_data


splits = [0.7, 0.15, 0.15]
train_data, val_data, test_data = split_data(data, splits)
'''

train_data = pd.read_csv('../Data_Split/train.csv').to_numpy()[:, 1 :]
val_data = pd.read_csv('../Data_Split/val.csv').to_numpy()[:, 1 :]
test_data = pd.read_csv('../Data_Split/test.csv').to_numpy()[:, 1:]

implicit_dict = pd.DataFrame(train_data, columns = ['User', 'Item', 'Rating']).groupby(['User'])['Item'].apply(lambda grp: list(grp.value_counts().index)).to_dict()


# Predicting movie ratings with trained weights
def pred_results(data, mean_rating, user_bias, item_bias, explicit_factor, item_factor, implicit_factor, implicit_dict, train_data):
    prediction = []
    num_data = data.shape[0]
    user_item_bias_matrix = user_bias.reshape(-1, 1) + item_bias.reshape(1, -1) + mean_rating
    
    for i in range(num_data):
        user_id = data[i, 0]
        item_id = data[i, 1]
        curr_items = implicit_dict[user_id]
        len_curr_items = len(curr_items)
        num_factors = item_factor.shape[1]
        curr_user_item_vector = np.zeros(num_factors)
        num_items = item_factor.shape[0]
        mask = np.zeros(num_items)
        mask[curr_items] = 1
        mask = np.reshape(mask, (-1, 1))
        
        curr_user_item_vector = np.sum(np.multiply(mask, implicit_factor), axis = 0)
        
        #Do from here.
        curr_user_train = train_data[train_data[:, 0] == user_id]
        curr_user_true_rating = np.zeros(num_items)
        curr_user_true_rating[curr_user_train[:, 1]] = curr_user_train[:, 2]
        curr_user_bias = user_item_bias_matrix[user_id, :]
        
        r_b = (curr_user_true_rating - curr_user_bias).reshape(-1, 1)
        curr_explicit_vector = np.sum(np.multiply(mask, np.multiply(r_b, explicit_factor)), axis = 0)
        
        pred_curr = mean_rating + user_bias[user_id] + item_bias[item_id] + np.dot((curr_explicit_vector / (len_curr_items ** 0.5)) + (curr_user_item_vector / (len_curr_items ** 0.5)), item_factor[item_id, :])
        prediction.append(pred_curr)
    
    prediction = np.array(prediction)
    
    return prediction

# Root Mean Squared Error Calculation
def calc_rmse(y_true, y_pred):
    return np.sqrt(np.sum(np.square(y_true - y_pred))/len(y_true))


# Stochastic Gradient Descent to update weights
def sgd_asvd(train_data, val_data, implicit_dict, num_users, num_items, num_factors, num_epochs, learning_rate, reg_param):
    mean_rating = np.mean(train_data[:, 2])
    user_bias = np.zeros(num_users)
    item_bias = np.zeros(num_items)
    #user_factor = np.random.rand(num_users, num_factors)
    explicit_factor = np.zeros((num_items, num_factors))
    item_factor = np.random.rand(num_items, num_factors)
    implicit_factor = np.zeros((num_items, num_factors))
    
    user_item_bias_matrix = user_bias.reshape(-1, 1) + item_bias.reshape(1, -1) + mean_rating
    
    
    train_error_list = []
    val_error_list = []
    
    start_time = time.time()
    for i in range(num_epochs):
        train_ratings = train_data[ : , 2]
        pred_train_ratings = []
        
        start_time = time.time()
        for j in range(train_data.shape[0]): 
            
            user_id = train_data[j, 0]
            item_id = train_data[j, 1]
            
            #temp_user_factor = user_factor[user_id, :]
            temp_item_factor = item_factor[item_id, :]
            
            curr_items = implicit_dict[user_id]
            len_curr_items = len(curr_items)
            
            curr_user_item_vector = np.zeros(num_factors)
            mask = np.zeros(num_items)
            mask[curr_items] = 1
            mask = np.reshape(mask, (-1, 1))
        
            curr_user_item_vector = np.sum(np.multiply(mask, implicit_factor), axis = 0)
            
            curr_user_train = train_data[train_data[:, 0] == user_id]
            curr_user_true_rating = np.zeros(num_items)
            curr_user_true_rating[curr_user_train[:, 1]] = curr_user_train[:, 2]
            curr_user_bias = user_item_bias_matrix[user_id, :]
            
            r_b = (curr_user_true_rating - curr_user_bias).reshape(-1, 1)
            curr_explicit_vector = np.sum(np.multiply(mask, np.multiply(r_b, explicit_factor)), axis = 0)
            
            pred_curr = mean_rating + user_bias[user_id] + item_bias[item_id] + np.dot((curr_explicit_vector / (len_curr_items ** 0.5)) + (curr_user_item_vector / (len_curr_items ** 0.5)), item_factor[item_id, :])
                
            #pred_curr = mean_rating + user_bias[user_id] + item_bias[item_id] + np.dot(user_factor[user_id, :] + (curr_user_item_vector / (len_curr_items ** 0.5)), item_factor[item_id, :])
            
            pred_train_ratings.append(pred_curr)
            curr_error = train_data[j, 2] - pred_curr
            #print(curr_error)
            
            user_bias[user_id] += learning_rate * (curr_error - reg_param*user_bias[user_id])
            item_bias[item_id] += learning_rate * (curr_error - reg_param*item_bias[item_id])
            #user_factor[user_id, :] += learning_rate * (curr_error * temp_item_factor - reg_param * temp_user_factor)
            item_factor[item_id, :] += learning_rate * (curr_error * ((curr_explicit_vector / (len_curr_items ** 0.5)) + (curr_user_item_vector / (len_curr_items ** 0.5))) - reg_param * temp_item_factor)
            explicit_factor += np.multiply(mask, learning_rate * (curr_error * (np.multiply(r_b, temp_item_factor.reshape(1, -1)) / (len_curr_items ** 0.5)) - reg_param * explicit_factor))
            implicit_factor += np.multiply(mask, learning_rate * (curr_error * (temp_item_factor.reshape(1, -1) / (len_curr_items ** 0.5)) - reg_param * implicit_factor))
            
            user_item_bias_matrix[user_id, item_id] = user_bias[user_id] + item_bias[item_id] + mean_rating
        
        #end_time = time.time()
        #print(end_time - start_time)
                                           
        #print(implicit_factor.shape)   
        pred_train_ratings = np.array(pred_train_ratings)
        train_error = np.sqrt(np.sum(np.square(train_ratings - pred_train_ratings))/len(train_ratings))
        train_error_list.append(train_error)
        
        #print('Training error after {}th epoch is {}'.format(i+1, train_error))
        
        val_ratings = val_data[ : , 2]
        pred_val_ratings = pred_results(val_data, mean_rating, user_bias, item_bias, explicit_factor, item_factor, implicit_factor, implicit_dict, train_data)
        val_error = calc_rmse(val_ratings, pred_val_ratings)
        val_error_list.append(val_error)
        
        print('Validation error after {}th epoch is {}'.format(i+1, val_error))
        
        rng = np.random.default_rng(seed = 5)
        rng.shuffle(train_data)

    end_time = time.time()
    time_taken = round(end_time - start_time, 2)
    
    print('Time required for {} epochs is {} second'.format(num_epochs, time_taken))
    print('Training Error is {0:.6f}'.format(train_error))
    print('Validation Error is {0:.6f}'.format(val_error))
    
    
    error_df = pd.DataFrame()
    error_df['Training Error'] = train_error_list
    error_df.to_csv('../Results/Train/Error_Train_lr{}_reg{}_factor{}_epoch{}.csv'.format(learning_rate, reg_param, num_factors, num_epochs))

    error_df = pd.DataFrame()
    error_df['Validation Error'] = val_error_list
    error_df.to_csv('../Results/Validate/Error_Train_lr{}_reg{}_factor{}_epoch{}.csv'.format(learning_rate, reg_param, num_factors, num_epochs))
    
    
    plt.figure()
    plt.plot(np.arange(len(train_error_list)) + 1, train_error_list, c = 'red', label = 'Training Error')
    plt.plot(np.arange(len(val_error_list)) + 1, val_error_list, c = 'blue', label = 'Validation Error')
    plt.xlabel('Epochs')
    plt.ylabel('Root Mean Squared Error')
    plt.title('RMSE vs Epochs During Training and Validation')
    plt.legend()
    plt.savefig('../Results/Figure/Error_Plot_lr{}_reg{}_factor{}_epoch{}.png'.format(learning_rate, reg_param, num_factors, num_epochs))
    
    return mean_rating, user_bias, item_bias, explicit_factor, item_factor, implicit_factor
    


# Hyperparameters to tune
lr_list = [0.01]
factors_list = [40]
reg_list = [0.05]
num_epochs = 100

for learning_rate in lr_list:
    for reg_param in reg_list:
        for num_factors in factors_list:
            mean_rating, user_bias, item_bias, explicit_factor, item_factor, implicit_factor = sgd_asvd(train_data, val_data, implicit_dict, num_users, num_items, num_factors, num_epochs, learning_rate, reg_param)


            weights = pd.DataFrame([mean_rating])
            #weights['Mean Rating'] = [mean_rating]
            weights.to_csv('../Weights/Mean_Rating/Weight_lr{}_reg{}_factor{}_epoch{}.csv'.format(learning_rate, reg_param, num_factors, num_epochs))
            weights = pd.DataFrame(user_bias)
            #weights['User Bias'] = user_bias
            weights.to_csv('../Weights/User_Bias/Weight_lr{}_reg{}_factor{}_epoch{}.csv'.format(learning_rate, reg_param, num_factors, num_epochs))
            weights = pd.DataFrame(item_bias)
            #weights['Item Bias'] = [item_bias]
            weights.to_csv('../Weights/Item_Bias/Weight_lr{}_reg{}_factor{}_epoch{}.csv'.format(learning_rate, reg_param, num_factors, num_epochs))
            weights = pd.DataFrame(explicit_factor)
            #weights['User Factor'] = [user_factor]
            weights.to_csv('../Weights/Explicit_Factor/Weight_lr{}_reg{}_factor{}_epoch{}.csv'.format(learning_rate, reg_param, num_factors, num_epochs))
            weights = pd.DataFrame(item_factor)
            #weights['Item Factor'] = [item_factor]
            weights.to_csv('../Weights/Item_Factor/Weight_lr{}_reg{}_factor{}_epoch{}.csv'.format(learning_rate, reg_param, num_factors, num_epochs))
            weights = pd.DataFrame(implicit_factor)
            weights.to_csv('../Weights/Implicit_Factor/Weight_lr{}_reg{}_factor{}_epoch{}.csv'.format(learning_rate, reg_param, num_factors, num_epochs))
            
            
            test_ratings = test_data[ : , 2]
            pred_test_ratings = pred_results(test_data, mean_rating, user_bias, item_bias, explicit_factor, item_factor, implicit_factor, implicit_dict, train_data)
            test_error = calc_rmse(test_ratings, pred_test_ratings)
            print('Test Error is {0:.6f}'.format(test_error))
            
            error = pd.DataFrame()
            error['Test Error'] = [test_error]
            error.to_csv('../Results/Test_Error/Test_error_lr{}_reg{}_factor{}_epoch{}.csv'.format(learning_rate, reg_param, num_factors, num_epochs))
            