import numpy as np
import pandas as pd
from numpy import linalg as LA

# initialize data
def Random_Initialize(features_dimension, users_dimension, products_dimension):
    return np.random.rand(users_dimension, features_dimension), np.random.rand(products_dimension, features_dimension)

# calculate the pian dao shu
def user_derivatives_function(data, user, product, lambdaa):
    nan_index = np.isnan(data)
    data[nan_index] = 0  
    temp = np.dot(user, product.T)  - data 
    temp[nan_index] = 0 
    data[nan_index] = np.nan 
    user_dervatives = np.random.rand(user.shape[0], user.shape[1] ) 
    for i in range(user.shape[0]):
        for j in range(user.shape[1]):
            user_dervatives[i][j] = np.dot(temp[i], product[:,j]) + lambdaa * user[i][j]
    return user_dervatives

# calculate the pian dao shu
def product_derivatives_function(data, user, product, lambdaa):
    nan_index = np.isnan(data) 
    data[nan_index] = 0 
    temp = np.dot(user, product.T) - data 
    temp[nan_index] = 0 
    data[nan_index] = np.nan 
    product_dervatives = np.random.rand(product.shape[0], product.shape[1]) 
    for i in range(product.shape[0]):
        for j in range(product.shape[1]):
            product_dervatives[i][j] = np.dot(temp[:,i], user[:,j]) + lambdaa * product[i][j]
    return product_dervatives

# calculate the cost
def cost_function(data, user, product, lambdaa):
    nan_index = np.isnan(data)
    data[nan_index] = 0 
    temp = np.dot(user, product.T) - data 
    temp[nan_index] = 0 
    cost = 0.5*np.sum(temp**2) + 0.5*lambdaa*(np.sum(user**2)+np.sum(product**2))
    data[nan_index] = np.nan 
    return cost
    
# Collaboration_Filtering function:iteration to reduce cost
def Collaboration_Filtering(data, user, product, iterate_num, learning_rate, lambdaa):
    for i in range(iterate_num):
        cost = cost_function(data, user, product, lambdaa) 
        user_derivatives = user_derivatives_function(data, user, product, lambdaa) 
        product_derivates = product_derivatives_function(data, user, product, lambdaa) 
        user = user - learning_rate * user_derivatives 
        product = product - learning_rate * product_derivates
        print 'the cost of :',i,'th iteration is ', cost
    return user, product

# predict the movie score
def Score(user_matrix, product_matrix):
    return np.dot(product_matrix,user_matrix.T)

if __name__ == '__main__':


    data = np.array([[4,5,np.nan,5,1,1,np.nan],[4,5,4,np.nan,np.nan,np.nan,1],
    [np.nan,np.nan,1,2,5,5,np.nan],[np.nan,1,np.nan,5,4,np.nan,5],
    [1,np.nan,np.nan,np.nan,5,np.nan,np.nan],[1,np.nan,1,1,np.nan,4,5],
    [5,np.nan,5,2,np.nan,np.nan,1],[np.nan,np.nan,4,np.nan,1,np.nan,np.nan]])
    
    user, movie = Random_Initialize(4,data.shape[0],data.shape[1])
# 正则化参数λ = 0.1，特征向量维数n = 4， 学习率α = 0.01
    user, movie = Collaboration_Filtering(data, user, movie, iterate_num=50000, learning_rate=0.01, lambdaa=0.1)
    score_matrix = Score(user, movie)
    for i in range(8):
        for j in range(4):
            user[i][j] = float('%.4f' %user[i][j])
    print 'user_feature:'
    print user

    for i in range(7):
        for j in range(4):
            movie[i][j] = float('%.4f' %movie[i][j])
    print 'movie_feature:'
    print movie

    for i in range(7):
        for j in range(8):
            score_matrix[i][j] = float('%.1f' %score_matrix[i][j])
    print 'score_matrix:'
    print score_matrix 

    ori_data=data.T
    Sigma = 0
    for i in range(7):
        for j in range(8):
            if((np.isnan(ori_data[i][j]))== False):
                Sigma = Sigma +((score_matrix[i][j]-ori_data[i][j])**2)
    print 'Sigma:',Sigma
# 计算电影的相似性计算（计算第二范数）    
    distance1_1 = np.array([0,0,0,0])
    distance1_2 = np.array([0,0,0,0])
    distance1_3 = np.array([0,0,0,0])
    distance1_4 = np.array([0,0,0,0])
    distance1_5 = np.array([0,0,0,0])
    distance1_6 = np.array([0,0,0,0])
    for j in range(4):
        distance1_1[j]=movie[0][j]-movie[1][j]
        distance1_2[j]=movie[0][j]-movie[2][j]
        distance1_3[j]=movie[0][j]-movie[3][j]
        distance1_4[j]=movie[0][j]-movie[4][j]
        distance1_5[j]=movie[0][j]-movie[5][j]
        distance1_6[j]=movie[0][j]-movie[6][j]
    print 'The difference between HP1 and HP2 is',LA.norm(distance1_1)
    print 'The difference between HP1 and HP3 is',LA.norm(distance1_2)
    print 'The difference between HP1 and TW is',LA.norm(distance1_3)
    print 'The difference between HP1 and SW1 is',LA.norm(distance1_4)
    print 'The difference between HP1 and SW2 is',LA.norm(distance1_5)
    print 'The difference between HP1 and SW3 is',LA.norm(distance1_6)
        
    distance2_1 = np.array([0,0,0,0])
    distance2_2 = np.array([0,0,0,0])
    distance2_3 = np.array([0,0,0,0])
    distance2_4 = np.array([0,0,0,0])
    distance2_5 = np.array([0,0,0,0])
    distance2_6 = np.array([0,0,0,0])
    for j in range(4):
        distance2_1[j]=movie[4][j]-movie[0][j]
        distance2_2[j]=movie[4][j]-movie[1][j]
        distance2_3[j]=movie[4][j]-movie[2][j]
        distance2_4[j]=movie[4][j]-movie[3][j]
        distance2_5[j]=movie[4][j]-movie[5][j]
        distance2_6[j]=movie[4][j]-movie[6][j]
    print 'The difference between SW1 and HP1 is',LA.norm(distance2_1)
    print 'The difference between SW1 and HP2 is',LA.norm(distance2_2)
    print 'The difference between SW1 and HP3 is',LA.norm(distance2_3)
    print 'The difference between SW1 and TW is',LA.norm(distance2_4)
    print 'The difference between SW1 and SW2 is',LA.norm(distance2_5)
    print 'The difference between SW1 and SW3 is',LA.norm(distance2_6)
  
