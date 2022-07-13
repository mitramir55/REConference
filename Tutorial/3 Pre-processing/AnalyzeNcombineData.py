
import pandas as pd
import os

import matplotlib.pyplot as plt
from NLPpipeline import NLPpipeLine_onestring
import requests

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1500)
pd.set_option('display.width', 2000)

Mycolumns = ['id', 'type', 'priority', 'severity', 'version', 'target_milestone', 'product', 'summary', 'depends_on', 'blocks']
pairsColumns = ['req1','req1Id','req1Type','req1Priority','req1Severity','req1Ver','req1Release', 'req1Product', 'req2','req2Id','req2Type','req2Priority', 'req2Severity','req2Ver', 'req2Release','req2Product','Label']
REST_API = "https://bugzilla.mozilla.org/rest/bug/"

def generatePair(item, instance2):
    req1 = item['summary']
    req1Id = item['id']
    req1Priority = item['priority']
    req1Severity = item['severity']
    req1Type = item['type']
    req1Version = item['version']
    req1Release = item['target_milestone']
    req1Product = item['product']

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
            'Label':'requires'}
    
    return pair

def ChangeStringListstoInts(Lst):
    ilist = []
    # print(Lst, type(Lst))
    # input("hit enter to proceed")
    if (Lst!='NaN') or (Lst!='') or (Lst!='nan'):
        Lst = str(Lst).split(',')
        if Lst: #if list not empty
            for i in Lst:
            #print(str(row[col]).split(','))
                ilist.append(int(float(i)))
        #print(ilist)
    return ilist

def convert(lst):
    lst = [str(a) for a in lst]
    lst = ', '.join(lst)
    return lst

def FetItemInfo(id):
    new_url = REST_API+str(id)
    try:
        resp = requests.get(new_url)
        ele = resp.json()
        if ele['bugs'][0]['type'] in ('enhancement'):
            ele['bugs'][0]['depends_on'] = convert(ele['bugs'][0]['depends_on'])
            return ele['bugs'][0]
        else:
            return 0
    except Exception:
        return 0

def CountDepends_On(df):
    '''
    For every element in the df 
        dependsList = get the depends_on column values
            for every item in dependsList:
                check that id exists in the list
                else get it from the REST API and 
                    if found 
                    generate the pair and append to a dataframe for later use
                    increment the counter
            store the counter as df['#newDependencies']
    return
    '''
    temp_pairs = []
    print(df.columns)
    print("There are: ", df['depends_on'].isnull().sum(), " with NULL depends_on field, out of ", len(df))
    #input("hit enter")
    for index, row in df.iterrows():
        lst = row['depends_on']
        #lst = ChangeStringListstoInts(lst)
        # if int(len(lst))>100:
        #     print(row['id'], lst)
        #     input("hit enter")
        count = 0
        if not pd.isnull(row['depends_on']):
            lst = ChangeStringListstoInts(lst)
            for i in lst:
                if not (df.loc[df['id']==i]).empty: #if that particular id is found in our database
                    #generate a pair and add create a new DB
                    pair = generatePair(row, df.loc[df['id']==i])
                    temp_pairs.append(pair)
                    count = count+1
                else:
                    #download it using REST API
                    inst = FetItemInfo(i)
                    #if found or not empty
                    if inst!=0:
                        # make sure to check the word count part and add the cosine to data
                        df_temp = pd.DataFrame([inst])
                        df_temp = NLPpipeLine_onestring(df_temp, 'summary')
                        df_temp['wordCount'] = df_temp['summary'].str.split().str.len()
                        counts = 3 # after looking at the sentence length 
                        df_temp = df_temp[df_temp['wordCount']>counts]
                        print(len(df_temp), inst)
                        if(len(df_temp)>0):
                            # append to Enhancement list
                            df = pd.concat([df,df_temp])
                            # create a pair
                            pair = generatePair(row, df.loc[df['id']==i])
                            temp_pairs.append(pair)
                            count = count+1
                    else:
                        print("Could not find ",i," in Bugzilla, so ignoring for now")        
            #print (len(lst), count)
        df.loc[index, '#newDependencies'] = count
    
    return df, temp_pairs

def process(df_ALL):
    print("Before: ", len(df_ALL))    
    df_ALL.drop_duplicates(subset=['id'],inplace=True)
    
    #working with just the enhancements this point onwards
    df_ALL = df_ALL[df_ALL['type']=='enhancement']
    print("After: ", len(df_ALL))    

    #pre process the data through NLP pipeline
    df_ALL = NLPpipeLine_onestring(df_ALL, 'summary')

    df_ALL['wordCount'] = df_ALL['summary'].str.split().str.len()
    # plt.hist(df_ALL['wordCount'], bins=10, rwidth=1.5, fc=(.2, .5, 0, 1))
    # plt.show()
    
    counts = 3 # after looking at the sentence length 
    df_ALL = df_ALL[df_ALL['wordCount']>counts]
    print("After selecting sentences above 3 words: ", len(df_ALL))    

    return df_ALL

def main():
    '''Read all the files from dir and extract the required columns
        Then concatenate the files to create one big DB
        Find the stats for them using Table
    '''
    path = '../rawDataDepends_on/'
    files = os.listdir(path)
    print(files)
    
    #initialize
    df_ALL = pd.DataFrame(columns=Mycolumns)
    df_pairs = pd.DataFrame(columns=pairsColumns)

    for i in files:
        df = pd.read_csv(path+i, usecols=Mycolumns)
        df_ALL = df_ALL.append(df)#, sort=True)
        #print(len(df_ALL), ":", len(df))
        
    df_ALL = process(df_ALL)
    
    # while counts:
    #     counts = int(input("enter the cutoff"))
    #     df_ALL = df_ALL[df_ALL['wordCount']>counts]
    #     plt.hist(df_ALL['wordCount'], bins=10, rwidth=1.5, fc=(.2, .5, 0, 1))
    #     plt.show()
    
    #print ("after more than 5 words: ",df['Label'].value_counts(ascending=True)) 

    #working on just a few of the projects
    #df_ALL = df_ALL[df_ALL['product'].isin(['Core'])]#(['Core','Firefox','Toolkit','Testing','DevTools','ThunderBird'])]    
   
    df_ALL, lst_pairs = CountDepends_On(df_ALL)
    df_pairs = pd.DataFrame(lst_pairs)
    df_ALL.to_csv("../genData/AllData.csv")
    df_pairs.to_csv("../genData/ALL_requires_Pos_Pairs.csv")
    pass


main()
