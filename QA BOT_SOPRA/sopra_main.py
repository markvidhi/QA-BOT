"""This is the main file from whch execution begins."""

#All Packages

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
from sopra_pickle import later


#Frontend implemented in Flask starts from here
app = Flask(__name__)

#send is a function which executes when an user inputs its Query
@app.route("/send",methods=["GET","POST"])
def send():
    if request.method=="POST":
        textques=request.form["age"]
        config = configparser.RawConfigParser()#for Config file
        config.read('conf.ini', encoding='utf-8-sig')
        th=config.getfloat('d', 'threshold')#getting values from config file
        path=config.get('d', 'csv_file_path')


        #'Quest' function evalues the Answer/output
        q1,ans1,r1,q2,ans2,r2 = Quest(textques,th,path)
        print(q2)
        return(render_template("age.html",ques1=q1,ans1=ans1,ques2=q2,ans2=ans2))#passing values to age.html for showing Answer to an user
        
    return(render_template("index1.html"))


#index mathod is for handling an exception
@app.route('/')
def index():
    raise Exception("Can't connect to database")


@app.errorhandler(Exception)
def exception_handler(error):
    return "!!!!"  + repr(error)

 


#This is the method 'Quest' which finds best possible answer to the user's Query
def Quest(test_set_sentence,th,path):
    
    csv_file_path = path  #dataset path 
    tfidf_vectorizer_pikle_path = "tfidf_vectorizer.pickle"  #pickle file for tfidf vectorizer
    tfidf_matrix_train_pikle_path ="tfidf_matrix_train.pickle" #pickle file for training vector

    
    i = 0
    sentences = []
    test_set = (test_set_sentence, "")


    #Pickle file is created only once. So, if pickle file exists, 'try' will run
    try:  
       
        f = open(tfidf_vectorizer_pikle_path, 'rb')
        tfidf_vectorizer = pickle.load(f)
        f.close()
        
        f = open(tfidf_matrix_train_pikle_path, 'rb')
        tfidf_matrix_train = pickle.load(f)
        f.close()
        
       
    #if pickle file does not exist, it will create pickle file by calling a function later which is in another python file named sopra_pickle
    
    except:
        tfidf_vectorizer_pikle_path,tfidf_matrix_train_pikle_path=later()



    tfidf_matrix_test = tfidf_vectorizer.transform(test_set) #create testing vector
    print(tfidf_matrix_test)

    # calculate cosine similarity between training vector and testing vector
    
    cosine = cosine_similarity(tfidf_matrix_test, tfidf_matrix_train)
    cosine = np.delete(cosine, 0)#if value of cosine is 0, remove that value 
    
    max = cosine.max()    #finds max of cosine value
    response_index1 = 0   #for storing index of output 1
    response_index2 = 0   #for storing index of output 2

    s_cos=sorted(cosine, reverse=True)  #sort cosine values in reverse order

    

    smax = s_cos[1]  # finds second most relevant document

    if (max > th): #comparing cosine score with threshold value which is 0.7 and if more than 1 document has score greater than 0.7, then it will run

        new_max = max - 0.01  #if value is greater than threshold,reduce it by 0.01
        
        list1 = np.where(cosine > new_max) # load them to a list1
         
        response_index1 = random.choice(list1) # choose a random one to return to the user 
        
        response_index1 = list1
        
    else:

        response_index1 = np.where(cosine == max)[0][0] + 2 # else simply return the highest score

        
    #same for second highest cosine score's document
    if (smax > 0.7): 
        new_max = smax - 0.01 #if more than 1 document has score greater than 0.7, then it will run
        
        list2 = np.where(cosine >= new_max and cosine < max )# load them to a list if it is second highest value
        
        response_index2 = random.choice(list2) # choose a random one to return to the user 
        
        response_index2 = list2
        
    else:
        response_index2 = np.where(cosine == smax)[0][0] + 2  # else simply return the second highest score
       
    j = 0
    q1=["popoNot Foundpopo"]  #if relevant document doesn't found, then print 'NOT FOUND'
    ans1=["\nNot Found\n"]
    r1=0

    # loop to return the output
    with open(csv_file_path, "r",encoding="utf-8") as sentences_file:
        reader = csv.reader(sentences_file, delimiter=',')
        
        for row in reader:
            j += 1  # we begin with 1 not 0 and j is initialized by 0
            if j == response_index1: #if value of j is equal to first output's index
                print ("\n")
                print ("\n")
                q1=row[0] #save Output's Ques to q1
                ans1=row[1] #save Output's Ans to ans1
                r1=response_index1 #save Output's response/index to r1
                break
            
                 
        j = 0
        q2=["popoNot Foundpopo"]  #if relevant document doesn't found, then print 'NOT FOUND'
        ans2=["\nNot Found\n"]
        r2=0
        for row in reader:
            j += 1  # we begin with 1 not 0 and j is initialized by 0
            if j == response_index2:#if value of j is equal to second output's index
                print ("\n")
                print ("\n")
                q2=row[0] #save Output's Ques to q1
                ans2=row[1] #save Output's Ans to ans1
                r2=response_index2 #save Output's response/index to r2
                break

        #Replace symbols
        q1=str(q1).replace('[','').replace(']','').replace('\\n', 'popo').replace('\\r', ' ').replace('&gt', '>').replace('&lt', '<') 
        a1=str(ans1).replace('[',' ').replace(']','').replace('\\n', '\n').replace('\\r', ' ').replace('&gt', '>').replace('&lt', '<')
        q2=str(q2).replace('[',' ').replace(']','').replace('\\n', 'popo').replace('\\r', ' ').replace('&gt', '>').replace('&lt', '<')
        a2=str(ans2).replace('[',' ').replace(']','').replace('\\n', '\n').replace('\\r', ' ').replace('&gt', '>').replace('&lt', '<')
        
        return q1,ans1,r1,q2,ans2,r2 #Return Ques, Ans and its response

    

#Main Functon
if __name__ == "__main__":
    app.run(debug=True)
