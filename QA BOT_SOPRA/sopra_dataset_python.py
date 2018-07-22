"""This file is for creating the dataset"""

from stackapi import StackAPI
import csv
import simplejson as json

SITE = StackAPI('stackoverflow')#stack overflow API
x = 'questions/'
p = '/answers'

data=1227000 #QUESTION ID

filename = 'sopra_dataset.csv' #Dataset file name


#open the file
with open(filename, 'a',encoding='utf8',newline='') as f:
    n=0   
    w = csv.writer(f)
    while(n<=80): #loop to write into file
        qn1=[]
        qn2=[]
        y= ('%s%d'%(x,data))  
        q= ('%s%d%s'%(x,data,p))
        ques = SITE.fetch(y,body='true',filter='!9YdnSJ*_T') #fetching questions
        comments = SITE.fetch(q,body='true', is_accepted= 'true',filter = '!-*f(6rktpIY5') #fetching answers
        
        j1 = json.dumps(ques) 
        n1 = json.loads(j1)
        
        j2 = json.dumps(comments)
        n2 = json.loads(j2)

        #append questions to qn1 list
        for body in n1['items']:     
            if(body['body_markdown']==" "):
                qn1=[]
            else:
                qn1.append([body['body_markdown']])
            



        #append answers to qn1 list
        for body in n2['items']:
            if(body['body_markdown']==" "):
                qn2=[]
            else:
                #qn2.append(data)
                #qn2.append('A')
                qn1.append([body['body_markdown']])
                #qn1.append([body['body']])


        if qn1:  #write qn1 list to dataset
            w.writerow(qn1)
            n=n+1

        data=data+1   #increment Question ID


