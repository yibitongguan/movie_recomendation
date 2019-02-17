# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 09:33:02 2019

@author: Dcm
"""
"""基于Tensorflow的电影推荐"""
import pandas as pd
import numpy as np
import tensorflow as tf
# -----------导入数据-------------------
ratings_df = pd.read_csv('./ml-latest-small/ratings.csv')
movies_df = pd.read_csv('./ml-latest-small/movies.csv')

movies_df['movieRow'] = movies_df.index
movies_df = movies_df[['movieRow','movieId','title']]
movies_df.to_csv('./ml-latest-small/moviesProcessed.csv', index=False,
                 header=True, encoding='utf-8')

# 合并
ratings_df = pd.merge(ratings_df, movies_df, on='movieId')
ratings_df = ratings_df[['userId','movieRow','rating']]
ratings_df.to_csv('./ml-latest-small/ratingsProcessed.csv', index=False,
                  header=True, encoding='utf-8')

# ---------创建电影评分矩阵rating和评分纪录矩阵record---------
userNo = ratings_df['userId'].max() + 1
movieNo = ratings_df['movieRow'].max() + 1
rating = np.zeros((movieNo,userNo))

for index,row in ratings_df.iterrows():
    #将ratings_df表里的'movieRow'和'userId'列，填上row的‘评分’
    rating[int(row['movieRow']),int(row['userId'])] = row['rating']
    
record = rating > 0
record = np.array(record, dtype = int)

# ----------构建模型--------------
def normalizeRatings(rating, record):
    #m代表电影数量，n代表用户数量
    m, n =rating.shape
    #每部电影的平均得分
    rating_mean = np.zeros((m,1))
    rating_norm = np.zeros((m,n))

    for i in range(m):
        idx = record[i,:] !=0
        rating_mean[i] = np.mean(rating[i,idx])
        #rating_norm = 原始得分-平均得分
        rating_norm[i,idx] -= rating_mean[i]
    return rating_norm, rating_mean

rating_norm, rating_mean = normalizeRatings(rating, record)
#对值为NaNN进行处理，改成数值0
rating_norm =np.nan_to_num(rating_norm)
rating_mean =np.nan_to_num(rating_mean)

# 创建模型
num_features = 10
X_parameters = tf.Variable(tf.random_normal([movieNo, num_features],stddev = 0.35))
Theta_parameters = tf.Variable(tf.random_normal([userNo, num_features],stddev = 0.35))
#基于内容的推荐算法模型
loss = 1/2 * tf.reduce_sum(((tf.matmul(X_parameters, Theta_parameters, transpose_b = True) - rating_norm) * record) ** 2) + 1/2 * (tf.reduce_sum(X_parameters ** 2) + tf.reduce_sum(Theta_parameters ** 2))

train = tf.train.AdamOptimizer(1e-4).minimize(loss)

tf.summary.scalar('loss',loss)
#merge_all 可以将所有summary全部保存到磁盘，以便tensorboard显示。
summaryMerged = tf.summary.merge_all()
filename = './movie_tensorborad'
writer = tf.summary.FileWriter(filename)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
#运行
for i in range(5000):
    # 把训练的结果summaryMerged存在movie里
    _, movie_summary = sess.run([train, summaryMerged])
    # 把训练的结果保存下来
    writer.add_summary(movie_summary, i)

#--------------评估模型------------------
Current_X_parameters, Current_Theta_parameters = sess.run([X_parameters, Theta_parameters])
# 计算所有的预测得分
predicts = np.dot(Current_X_parameters,Current_Theta_parameters.T) + rating_mean
errors = np.sqrt(np.sum((predicts - rating)**2))

#---------------测试电影推荐系统-------------------
def test():
    user_id = input('您要想哪位用户进行推荐？请输入用户编号：')
    sortedResult = predicts[:, int(user_id)].argsort()[::-1]
    idx = 0
    print('为该用户推荐的评分最高的20部电影是：'.center(80,'='))
    for i in sortedResult:
        print('评分: %.2f, 电影名: %s' % (predicts[i,int(user_id)],movies_df.iloc[i]['title']))
        idx += 1
        if idx == 20:break
    
if __name__ == '__main__':
    test()
