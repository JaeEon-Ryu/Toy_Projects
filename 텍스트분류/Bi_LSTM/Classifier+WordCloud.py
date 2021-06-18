# -*- coding: utf-8 -*-


import os
import tensorflow as tf
import Bi_LSTM as Bi_LSTM
import Word2Vec as Word2Vec
import gensim
import numpy as np
import csv

from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import matplotlib
from matplotlib import rc
rc('font', family='NanumBarunGothic')
import warnings
warnings.filterwarnings(action='ignore')


def Convert2Vec(model_name, sentence):

    word_vec = []
    sub = []
    model = gensim.models.word2vec.Word2Vec.load(model_name)

    for word in sentence:
        if (word in model.wv.vocab):
            sub.append(model.wv[word])
        else:
            sub.append(np.random.uniform(-0.25, 0.25, 300))  # used for OOV words
    word_vec.append(sub)
    return word_vec

def Grade(sentence):
    tokens = W2V.tokenize(sentence)

    embedding = Convert2Vec('Data\\post.embedding', tokens)
    zero_pad = W2V.Zero_padding(embedding, Batch_size, Maxseq_length, Vector_size)
    global sess
    result = sess.run(prediction, feed_dict={X: zero_pad, seq_len: [len(tokens)]}) # tf.argmax(prediction, 1)이 여러 prediction 값중 max 값 1개만 가져옴
    point = result.ravel().tolist()
    Tag = ["IT과학", "경제", "정치", "e스포츠", "골프", "농구", "배구", "야구", "일반 스포츠", "축구", "사회", "생활문화"]

    predict = "IT과학"
    predict_point = 0
    index =-1
    temp_li = [3,4,5,6,7,8,9]
    for t, i in zip(Tag, point):
        index += 1
        if index in temp_li : continue
        if i >= predict_point:
            predict = t
            predict_point = i
        print(t, round(i * 100, 2),"%")
        # percent = t + str(round(i * 100, 2)) + "%"

        #text.write(percent)
        #text.write("\n")
    #text.write("\n")
    # print(predict)
    # return predict

W2V = Word2Vec.Word2Vec()

Batch_size = 1
Vector_size = 300
Maxseq_length = 800  # Max length of training data
learning_rate = 0.001
lstm_units = 128
num_class = 12
keep_prob = 1.0

X = tf.placeholder(tf.float32, shape = [None, Maxseq_length, Vector_size], name = 'X')
Y = tf.placeholder(tf.float32, shape = [None, num_class], name = 'Y')
seq_len = tf.placeholder(tf.int32, shape = [None])

BiLSTM = Bi_LSTM.Bi_LSTM(lstm_units, num_class, keep_prob)

with tf.variable_scope("loss", reuse = tf.AUTO_REUSE):
    logits = BiLSTM.logits(X, BiLSTM.W, BiLSTM.b, seq_len)
    loss, optimizer = BiLSTM.model_build(logits, Y, learning_rate)

prediction = tf.nn.softmax(logits)  # softmax

saver = tf.train.Saver()
init = tf.global_variables_initializer()
modelName = "Data\\Bi_LSTM.model"

sess = tf.Session()
sess.run(init)
saver.restore(sess, modelName)

def print_wordCloud(sentence):
    noun_list = []
    tokens = W2V.tokenize(sentence)
    for word in tokens:
        if word.endswith("Noun"):
            noun_list.append(word[:word.find('/')])

    count = Counter(noun_list)
    words = dict(count.most_common())
    wordcloud = WordCloud(
        font_path='/Library/Fonts/NanumBarunGothic.ttf',  # 맥에선 한글폰트 설정 잘해야함.
        background_color='white',  # 배경 색깔 정하기
        colormap='Accent_r',  # 폰트 색깔 정하기
        width=800,
        height=800
    )
    wordcloud_words = wordcloud.generate_from_frequencies(words)
    array = wordcloud.to_array()
    print(type(array))  # numpy.ndarray
    print(array.shape)  # (800, 800, 3)

    fig = plt.figure(figsize=(10, 10))
    plt.imshow(array, interpolation="bilinear")
    plt.axis('off')
    plt.show()


while(True):
    #try:
    s = input("문장을 입력하세요 : ")
    Grade(s)
    print_wordCloud(s)
    #except:
    #     pass