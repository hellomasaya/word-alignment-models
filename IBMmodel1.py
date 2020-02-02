#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import math
from nltk.tokenize import word_tokenize
from collections import defaultdict


# In[2]:


#DATA
file = open('./train.en', 'r')
train_en = file.read()
sentences_train_en = train_en.split("\n")

file = open('./train.hi', 'r')
train_hi = file.read()
sentences_train_hi = train_hi.split("\n")

file = open('./test.en', 'r')
test_en = file.read()
sentences_test_en = test_en.split("\n")

file = open('./test.hi', 'r')
test_hi = file.read()
sentences_test_hi = test_hi.split("\n")

file = open('./dev.en.txt', 'r')
dev_en = file.read()
sentences_dev_en = dev_en.split("\n")

file = open('./dev.hi', 'r')
dev_hi = file.read()
sentences_dev_hi = dev_hi.split("\n")


# In[59]:


def is_converged(new, old, epoch):
    epsilone = 0.0000001
    new = list(new.values())
    old = list(old.values())
    print("here")
    for i in range(len(old)):
        print("s",epoch)
        if math.fabs(new[i]-old[i]) > epsilone: 
            return False
        return True


# In[60]:

def perform_EM(en_sentences, hi_sentences):
    
    translation_prob = defaultdict(float)
    translation_prob_prev = defaultdict(float)
    uni_ini = 0.00001
    
    epoch = 0
    
    while not is_converged(translation_prob, translation_prob_prev, epoch):
    # while(epoch<2):
        
        translation_prob_prev = translation_prob
        
        print(len(translation_prob_prev))
        print(len(translation_prob))
        
        epoch += 1
        print("epoch num:", epoch,"\n")
        count = defaultdict(float)
        total = defaultdict(float)
        for index_sen, hin_sen in enumerate(hi_sentences):
            #compute normalization
            hin_sen_words = hin_sen.split(" ")
            s_total = defaultdict(float)
            for hin_word in hin_sen_words:
                s_total[hin_word] = 0
                eng_sen_words = en_sentences[index_sen].split(" ")
                for eng_word in eng_sen_words:
                    if epoch == 1:
                        s_total[hin_word] += uni_ini
                        translation_prob[(hin_word, eng_word)] = uni_ini
                    else:
                        s_total[hin_word] += translation_prob[(hin_word, eng_word)]
            
            #collect counts
            for hin_word in hin_sen_words:
                eng_sen_words = en_sentences[index_sen].split(" ")
                for eng_word in eng_sen_words:
                    if epoch == 1:
                        translation_prob[(hin_word, eng_word)] = uni_ini
                        count[(hin_word, eng_word)] += uni_ini/s_total[hin_word]
                        total[eng_word] += uni_ini/s_total[hin_word]
                    else:
                        count[(hin_word, eng_word)] += translation_prob[(hin_word, eng_word)]/s_total[hin_word]
                        total[eng_word] += translation_prob[(hin_word, eng_word)]/s_total[hin_word]                   

        #estimate probabilities
        for (hin_word, eng_word) in translation_prob.keys():
                translation_prob[(hin_word, eng_word)] = count[(hin_word, eng_word)]/total[eng_word]

        print(len(translation_prob_prev))
        print(len(translation_prob))
                

    return translation_prob


# In[7]:


def train_model(sentences_train_en, sentences_train_hi):
    
    translation_prob = perform_EM(sentences_train_en, sentences_train_hi)
    return translation_prob


# In[20]:


def test_model(dataset, tef):
#     tef = np.load('./models/IBMmodel1tef.npy')
    for sentence in dataset:
        translate_sentence(sentence, tef)


# In[9]:


def translate_sentence(sentence, tef):
    
    
    tokens = sentence.split(" ")
#     for token in tokens:
#         print(list(tef.items())[0])

    
#     max_score = -1
#     max_sentence = ""
    
#     prob = get_translation_prob(end_sentence, tef)
    
#     if prob > max_score:
#         max_score = prob
#         max_sentence = poss_sentence


# In[61]:


tef = train_model(sentences_train_en, sentences_train_hi)


# In[ ]:


# np.save("./models/IBMmodel1tef_3", tef)


# In[ ]:


# test_model(test_en_tokenised_sentence, tef)


# In[10]:





# In[11]:





# In[ ]: