import re, string, unicodedata
import nltk
#import contractions
import inflect
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
import pandas as pd
from html.parser import HTMLParser
html_parser = HTMLParser()
import glob


frequency = 2500  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second

# def replace_contractions(text):
#     """Replace contractions in string of text"""
#     return contractions.fix(text)

def remove_URL(sample):
    """Remove URLs from a sample string"""
    sample =  re.sub(r"http\S+", "", sample)
    #sample = re.sub(r'[^\w\s]', '', sample)  #return puctuation
    sample = re.sub(r"[\[\],@\'?\.$%_:()\-\"&;<>{}|+!*#]", " ", sample, flags=re.I)
    sample = re.sub(r"\s+"," ", sample, flags = re.I) #remove empty spaces
    sample = re.sub(r"\s+[a-zA-Z]\s+", " ", sample) #remove single characters
    sample = ' '.join(w for w in sample.split() if not any(x.isdigit() for x in w)) #remove of the type e70bae07664def86aefd11c86dac818ab7ea64ea
    return sample.lower()

# def remove_non_ascii(words):
#     """Remove non-ASCII characters from list of tokenized words"""
#     new_words = []
#     for word in words:
#         new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
#         new_words.append(new_word)
#     return new_words

# def to_lowercase(words):
#     """Convert all characters to lowercase from list of tokenized words"""
#     #new_words = []
#     words = words.split()
#     for word in words:
#         word = word.lower()
#         #new_words.append(new_word)
#     word = " ".join(word)
#     return word

# def remove_punctuation(words):
#     """Remove punctuation from list of tokenized words"""
#     new_words = []
#     for word in words:
#         new_word = re.sub(r'[^\w\s]', ' ', word)
#         if new_word != '':
#             new_words.append(new_word)
#     return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    words = words.split() 
    noise_free_words = [word for word in words if not word.isdigit()] 
    noise_free_text = " ".join(noise_free_words) 
    return noise_free_text

# def _remove_noise(input_text):
#     words = input_text.split() 
#     noise_free_words = [word for word in words if word not in noise_list] 
#     noise_free_text = " ".join(noise_free_words) 
#     return noise_free_text


def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    #new_words = []
    words = words.split()
    noise_free_words = [word for word in words if word not in stopwords.words('english')] 
    noise_free_words = [word for word in words if word not in ["meta","META",'a','the','an']] 
    noise_free_text = " ".join(noise_free_words) 
    return noise_free_text

# def remove_punctuation(s):
#     s = ''.join([i if i not in frozenset(string.punctuation) else ' ' for i in s])
#     return s.lower()


# def stem_words(words):
#     """Stem words in list of tokenized words"""
#     stemmer = LancasterStemmer()
#     stems = []
#     for word in words:
#         stem = stemmer.stem(word)
#         stems.append(stem)
#     return stems

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    words = words.split()
    lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in words])
    # for word in words:
    #     word = lemmatizer.lemmatize(word, pos='v')
    #     #lemmas.append(word)
    #     #word = word.lemmatize()
    # words = " ".join(words) 
    return lemmatized_output

# def apoLookup(words):
#     APPOSTOPHES = {"'s" : " is", "'re" : " are", "'t":" not"} ## Need a huge dictionary
#     words = words.split()
#     reformed = [APPOSTOPHES[word] if word in APPOSTOPHES else word for word in words]
#     reformed = " ".join(reformed)
#     return reformed


def normalize(words):
    #words = remove_non_ascii(words)
    #print(words)
    words = remove_URL(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    words = lemmatize_verbs(words)
    #print("--------"+words+"-------\n")
    return words

# def preprocess(sample):
#     sample = remove_URL(sample)
#     sample = replace_contractions(sample)
#     # Tokenize
#     words = nltk.word_tokenize(sample)

#     # Normalize
#     return normalize(words)


# if __name__ == "__main__":
#     sample = "Blood test for Down's syndrome hailed http://bbc.in/1BO3eWQ"               
    
#     sample = remove_URL(sample)
#     sample = replace_contractions(sample)

#     # Tokenize
#     words = nltk.word_tokenize(sample)
#     print(words)

#     # Normalize
#     words = normalize(words)
#     print(words)

def NLPpipeLine_onestring(df_data,col1):
    df_data[col1] = df_data[col1].apply(normalize)
    return df_data


def NLPpipeLine(df_data,col1,col2):
    #Tokenize
    #df_data[col1] = df_data[col1].apply(nltk.word_tokenize)
    #df_data[col2] = df_data[col2].apply(nltk.word_tokenize)
    
    # Normalize
    #words = normalize(words)
    df_data[col1] = df_data[col1].apply(normalize)
    df_data[col2] = df_data[col2].apply(normalize)
    #print(df_data['req1'].head())
    #input("hit enter")

    #print(words)

    return df_data


def Test():
    # load CSV
    df_data = pd.read_csv("Requires_enhancement_2_enhancementData.csv")
    #print(df_data['req1'].head(5))
    df_data = NLPpipeLine(df_data,'req1','req2')
    
    df_data.to_csv("Processed_Requires_enhancement_2_enhancementData.csv")

import os
def processAllFile():
    entries = os.listdir('Data/')
    print(entries)
    df_data = pd.read_csv("Data/"+entries[0])
    print(df_data['req1'].head(5))

    input("hit enter")
    for filename in entries:
        print("Processing ", filename)
        # load CSV
        df_data = pd.read_csv("Data/"+filename)
        #print(df_data['req1'].head(5))
        df_data = NLPpipeLine(df_data,'req1','req2')
        df_data.to_csv("Processed"+filename)
        print("Done")
        
    #print(entries)

def processAllFileListLines():
    entries = os.listdir('Data/')
    print(entries)
    size = 0
    df_data = pd.read_csv("Processed_Requires_enhancement_2_enhancementData.csv")
    print(len(df_data))
    for i in entries:
        df_data = pd.read_csv("Data/"+i)
        print(i,len(df_data))
        size = size +len(df_data)
    print(size)
    
def verify():
    c = "This must not b3 delet3d, e70bae07664def86aefd11c86dac818ab7ea64ea but the number at the end yes 134411"
    c = ' '.join(w for w in c.split() if not any(x.isdigit() for x in w))
    print (c)

#processAllFileListLines()
#verify()

#Test()
#processAllFile()

