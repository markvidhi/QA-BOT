
"""This python file creates pickle file for training vector and testing vector"""

#python packages
from tkinter import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import ngrams
import pickle
from nltk import sent_tokenize
from nltk.stem.lancaster import LancasterStemmer
# importing csv module
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from scipy.sparse.csr import csr_matrix 

import pickle  
import csv     
import timeit  
import random
import configparser
from flask import Flask,render_template,request



#function to create pickle file
def later():

            tfidf_vectorizer_pikle_path = "tfidf_vectorizer.pickle"  #pickle file for tfidf vectorizer
            tfidf_matrix_train_pikle_path ="tfidf_matrix_train.pickle"

            config = configparser.RawConfigParser()
            config.read('conf.ini', encoding='utf-8-sig') #reading config file
            th=config.getfloat('d', 'threshold')  #getting value of threshold
            path=config.get('d', 'csv_file_path') #getting dataset path

            start = timeit.default_timer() #timer start,this timer is to find exact time taken to train the dataset 

            i =0
            with open(path, "r", errors='ignore') as sentences_file: #open the dataset
                reader = csv.reader(sentences_file, delimiter=',')
                sentences = []
                for row in reader:
                    sentences.append(row[0])#put all Questions which are the first column of the dataset in 'sentence' list  
                    i += 1


                p = [elem.strip().split(',') for elem in sentences]# seperate Questions by comma 
                processed = [] #to save stemmed words 
                #pr = []
                ngram = [] #to save bigrams of all documents

                #cleaning the dataset
                for document in p:
                    tokenizer = nltk.RegexpTokenizer(r'\w+') 
                    intermediate = tokenizer.tokenize(str(document)) #tokenize each document into words
                    stop = stopwords.words('english') 
                    intermediate = [i for i in intermediate if i not in stop] #remove stopwords
                    lanste = LancasterStemmer()
                    intermediate = [lanste.stem(i) for i in intermediate] #Stemming
                    processed.append(intermediate) #save stemmed words in 'processed' list
                for document in processed:
                    bigrams = ngrams(document, 2) #create tokens/bigrams
                for grams in bigrams:
                    ngram.append(grams) #save bigrams of all documents in 'ngram' list
               
            #Apply tfidf vectorizer to training set with our vocabulary which is stored in 'ngram' list
                    
            tfidf_vectorizer = TfidfVectorizer(min_df=1, sublinear_tf=True, use_idf =True,vocabulary="ngram")
            tfidf_matrix_train = tfidf_vectorizer.fit_transform(sentences) #finds the tfidf score with normalization
           
           
            stop = timeit.default_timer()
            print ("training time took was : ")
            print (stop - start)  #finds training time
           
            f = open(tfidf_vectorizer_pikle_path, 'wb')  #creates pickle file for tfidf vectorizer
            pickle.dump(tfidf_vectorizer, f)
            f.close()

           
            f = open(tfidf_matrix_train_pikle_path, 'wb') #creates pickle file for tfidf training matrix
            pickle.dump(tfidf_matrix_train, f)
            f.close()

            return(tfidf_vectorizer_pikle_path ,tfidf_matrix_train_pikle_path)  #return pickle file to orignal python file'sopra_main'


if ( __name__ == "__main__"):
    later()
