import NLPpipeline as myNLP
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import f1_score,precision_score,recall_score,confusion_matrix,classification_report
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE 
from collections import Counter
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import os
from nltk.corpus import stopwords 
from sklearn.metrics import jaccard_similarity_score
from nltk.tokenize import word_tokenize 
import random
import itertools



enhancements = "AllData.csv" #data with just the enhancements of a project already preprocessed and wordcount>3
path = "../genData/"
df_enhancements = pd.read_csv(path+enhancements)
print("enhancements: ", len(df_enhancements))

outputPath = "../processedGenData/"
fileName = "All_requires_Pos_Pairs.csv"  #this has been created using Analyze and Combine preprocessed and wordcount>3
outputFile = "All_requires_Pos_Neg_Pairs.csv"

used_pairs = set()

def generatePair(instance1, instance2):
    req1 = instance1.iloc[0]['summary']
    req1Id = instance1.iloc[0]['id']
    req1Priority = instance1.iloc[0]['priority']
    req1Severity = instance1.iloc[0]['severity']
    req1Type = instance1.iloc[0]['type']
    req1Version = instance1.iloc[0]['version']
    req1Release = instance1.iloc[0]['target_milestone']
    req1Product = instance1.iloc[0]['product']

    req2 = instance2.iloc[0]['summary']
    req2Id = instance2.iloc[0]['id']
    req2Priority = instance2.iloc[0]['priority']
    req2Severity = instance2.iloc[0]['severity']
    req2Type = instance2.iloc[0]['type']
    req2Version = instance2.iloc[0]['version']
    req2Release = instance2.iloc[0]['target_milestone']
    req2Product = instance2.iloc[0]['product']

    pair = {'req1':req1,'req1Id':req1Id, 
            'req1Type':req1Type,
            'req1Priority':req1Priority, 
            'req1Severity':req1Severity,
            'req1Ver':req1Version,
            'req1Release': req1Release, 
            'req1Product':req1Product,  

            'req2':req2,'req2Id':req2Id,
            'req2Type':req2Type,
            'req2Priority':req2Priority, 
            'req2Severity':req2Severity,
            'req2Ver':req2Version, 
            'req2Release': req2Release, 
            'req2Product':req2Product,
            'Label':'independent'}
    
    return pair


#Thanks to https://www.quora.com/How-do-you-create-random-nonrepetitive-pairs-from-a-list-in-Python

'''Return an iterator of random pairs from a list of numbers.'''
#Keep track of already generated pairs
#used_pairs = set()
def pair_generator(numbers):
    global used_pairs
    while True:
        pair = random.sample(numbers, 2)
        #Avoid generating both (1, 2) and (2, 1)
        pair = tuple(pair)
        if pair not in used_pairs:
            used_pairs.add(pair)
            yield pair
    #return pair
    
from sklearn.metrics.pairwise import cosine_similarity
def getSim(df):
    df['cosine'] = simi(df['req1'].astype(str) ,df['req2'].astype(str))
    return df

def simi(X, Y):
    # Program to measure similarity between  
    # two sentences using cosine similarity. 
   
    #X = input("Enter first string: ").lower() 
    #Y = input("Enter second string: ").lower() 
    #X ="I love horror movies"
    #Y ="Lights out is a horror movie"
    cos = []
    X = X.tolist()
    Y = Y.tolist()
    #print(X[0],"--------",Y[0])
    print(type(X))
    #input("Hit enter")
    # tokenization 
    for idx in range(len(X)):

        X_list = word_tokenize(X[idx])  
        Y_list = word_tokenize(Y[idx]) 
    
        # sw contains the list of stopwords 
        sw = stopwords.words('english')  
        l1 =[];l2 =[] 
    
        # remove stop words from string 
        X_set = {w for w in X_list if not w in sw}  
        Y_set = {w for w in Y_list if not w in sw} 
    
        # form a set containing keywords of both strings  
        rvector = X_set.union(Y_set)  
        for w in rvector: 
            if w in X_set: l1.append(1) # create a vector 
            else: l1.append(0) 
            if w in Y_set: l2.append(1) 
            else: l2.append(0) 
        c = 0
        
        # cosine formula  
        for i in range(len(rvector)): 
                c+= l1[i]*l2[i] 
        cosine = c / float((sum(l1)*sum(l2))**0.5) 
        #print("similarity: ", cosine)
        #print("Jaccard: ", jaccard_similarity_score(l1,l2)) 
        cos.append(cosine)
    return cos

def GenNegPairs(negPairs):
    df = pd.DataFrame()
    tempList = []
    for i in negPairs:
        req1Id = i[0]
        req2Id = i[1]
        req1Instance = df_enhancements.loc[df_enhancements['id']==req1Id]
        req2Instance = df_enhancements.loc[df_enhancements['id']==req2Id]
        pair = generatePair(req1Instance, req2Instance)
        tempList.append(pair)
    df = pd.DataFrame(tempList)
    return df

def main():
    '''
    read the Main all enhancement file for a project
    read the pairs file and then generate the negative samples almost 5 times the size
    '''
    global used_pairs
    df = pd.read_csv(path+fileName)#, usecols=['req1','req1Product','req2Product','req2','BinaryClass','MultiClass','req1Id', 'req2Id'], encoding='utf-8')
        
    print (df['Label'].value_counts(ascending=True))  
    
    #find similarity
    df = getSim(df)
    print ("after cosine: ", df['Label'].value_counts(ascending=True))  

    #generate negative pairs
    ''' generate positive number of negative pairs which are generated randomly'''
    used_pairs = list(zip(df.req1Id, df.req2Id))
    used_pairs = set(used_pairs)
    print("Used pairs are: ", len(used_pairs), set(itertools.islice(used_pairs, 5)))
    # A relatively long list
    AllIds = list(df_enhancements['id'])
    
    gen = pair_generator(AllIds)
    # Get len(df) number of pairs:
    negPairs = []
    #for i in xrange(int(len(df))):
    for i in range(int(len(df)*5)): #generating 1.5 times as the unspecified type are more.
        pair = next(gen)
        negPairs.append(pair)
    #print(len(negPairs), negPairs)

    #now iterate through the negPairs to generate the pairs with data
    df_neg = GenNegPairs(negPairs)
    print("generated :",len(df_neg)," Negative pairs")
    
    #compute similarity
    df_neg = getSim(df_neg) 

    #concatenate the pos and the neg
    df_alto = pd.concat([df,df_neg]) 

    print("-"*80)
    print("Total pos samples: ", len(df), "\n", "Total Negative samples: ", len(df_neg))
    print("Removing Duplicates if any ")
    df_alto.drop_duplicates(subset=['req1Id','req2Id'], inplace=True)
    print("New length is: ", len(df_alto))
    print("-"*80)
    print("Now saving to a file", outputPath+outputFile)
    df_alto.to_csv(outputPath+outputFile)
    #df.to_csv(outputPath+outputFile, encoding='utf-8', index = True)

main()