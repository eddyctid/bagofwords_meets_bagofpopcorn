## bagofwords_meets_bagofpopcorn




#Well lets do this.
#At first import what we need.

import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier


#panda does the reading  and saves as DataFrame
train = pd.read_csv("F:/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
train.head()

print(train.shape)
#Column names
print(train.columns.values)

    (25000, 3)
    ['id' 'sentiment' 'review']
    


#Raw Unclean Review Text
print(train.review[0])

    "With all this stuff going down at the moment with MJ i've started listening to his music, watching the odd documentary here and there, watched The Wiz and watched Moonwalker again. Maybe i just want to get a certain insight into this guy who i thought was really cool in the eighties just to maybe make up my mind whether he is guilty or innocent. Moonwalker is part biography, part feature film which i remember going to see at the cinema when it was originally released. Some of it has subtle messages about MJ's feeling towards the press and also the obvious message of drugs are bad m'kay.<br /><br />Visually impressive but of course this is all about Michael Jackson so unless you remotely like MJ in anyway then you are going to hate this and find it boring. Some may call MJ an egotist for consenting to the making of this movie BUT MJ and most of his fans would say that he made it for the fans which if true is really nice of him.<br /><br />The actual feature film bit when it finally starts is only on for 20 minutes or so excluding the Smooth Criminal sequence and Joe Pesci is convincing as a psychopathic all powerful drug lord. Why he wants MJ dead so bad is beyond me. Because MJ overheard his plans? Nah, Joe Pesci's character ranted that he wanted people to know it is he who is supplying drugs etc so i dunno, maybe he just hates MJ's music.<br /><br />Lots of cool things in this like MJ turning into a car and a robot and the whole Speed Demon sequence. Also, the director must have had the patience of a saint when it came to filming the kiddy Bad sequence as usually directors hate working with one kid let alone a whole bunch of them performing a complex dance scene.<br /><br />Bottom line, this movie is for people who like MJ on one level or another (which i think is most people). If not, then stay away. It does try and give off a wholesome message and ironically MJ's bestest buddy in this movie is a girl! Michael Jackson is truly one of the most talented people ever to grace this planet but is he guilty? Well, with all the attention i've gave this subject....hmmm well i don't know because people can be different behind closed doors, i know this for a fact. He is either an extremely nice but stupid guy or one of the most sickest liars. I hope he is not the latter."
    


```python
#Using BeautifulSoup to clean data initially & remove html tags and comments 


parsedRev = BeautifulSoup(train.review[0],"html.parser")


#Print the result to compare with Unclean data 
print(parsedRev.get_text())
```

    "With all this stuff going down at the moment with MJ i've started listening to his music, watching the odd documentary here and there, watched The Wiz and watched Moonwalker again. Maybe i just want to get a certain insight into this guy who i thought was really cool in the eighties just to maybe make up my mind whether he is guilty or innocent. Moonwalker is part biography, part feature film which i remember going to see at the cinema when it was originally released. Some of it has subtle messages about MJ's feeling towards the press and also the obvious message of drugs are bad m'kay.Visually impressive but of course this is all about Michael Jackson so unless you remotely like MJ in anyway then you are going to hate this and find it boring. Some may call MJ an egotist for consenting to the making of this movie BUT MJ and most of his fans would say that he made it for the fans which if true is really nice of him.The actual feature film bit when it finally starts is only on for 20 minutes or so excluding the Smooth Criminal sequence and Joe Pesci is convincing as a psychopathic all powerful drug lord. Why he wants MJ dead so bad is beyond me. Because MJ overheard his plans? Nah, Joe Pesci's character ranted that he wanted people to know it is he who is supplying drugs etc so i dunno, maybe he just hates MJ's music.Lots of cool things in this like MJ turning into a car and a robot and the whole Speed Demon sequence. Also, the director must have had the patience of a saint when it came to filming the kiddy Bad sequence as usually directors hate working with one kid let alone a whole bunch of them performing a complex dance scene.Bottom line, this movie is for people who like MJ on one level or another (which i think is most people). If not, then stay away. It does try and give off a wholesome message and ironically MJ's bestest buddy in this movie is a girl! Michael Jackson is truly one of the most talented people ever to grace this planet but is he guilty? Well, with all the attention i've gave this subject....hmmm well i don't know because people can be different behind closed doors, i know this for a fact. He is either an extremely nice but stupid guy or one of the most sickest liars. I hope he is not the latter."
    


```python
#We can see the result has Numbers and Symbols in it. Not good for "Bag of Words". Lets start removing.



cleanRev = re.sub("[^a-zA-Z]"," ",parsedRev.get_text())
print(cleanRev)
```

     With all this stuff going down at the moment with MJ i ve started listening to his music  watching the odd documentary here and there  watched The Wiz and watched Moonwalker again  Maybe i just want to get a certain insight into this guy who i thought was really cool in the eighties just to maybe make up my mind whether he is guilty or innocent  Moonwalker is part biography  part feature film which i remember going to see at the cinema when it was originally released  Some of it has subtle messages about MJ s feeling towards the press and also the obvious message of drugs are bad m kay Visually impressive but of course this is all about Michael Jackson so unless you remotely like MJ in anyway then you are going to hate this and find it boring  Some may call MJ an egotist for consenting to the making of this movie BUT MJ and most of his fans would say that he made it for the fans which if true is really nice of him The actual feature film bit when it finally starts is only on for    minutes or so excluding the Smooth Criminal sequence and Joe Pesci is convincing as a psychopathic all powerful drug lord  Why he wants MJ dead so bad is beyond me  Because MJ overheard his plans  Nah  Joe Pesci s character ranted that he wanted people to know it is he who is supplying drugs etc so i dunno  maybe he just hates MJ s music Lots of cool things in this like MJ turning into a car and a robot and the whole Speed Demon sequence  Also  the director must have had the patience of a saint when it came to filming the kiddy Bad sequence as usually directors hate working with one kid let alone a whole bunch of them performing a complex dance scene Bottom line  this movie is for people who like MJ on one level or another  which i think is most people   If not  then stay away  It does try and give off a wholesome message and ironically MJ s bestest buddy in this movie is a girl  Michael Jackson is truly one of the most talented people ever to grace this planet but is he guilty  Well  with all the attention i ve gave this subject    hmmm well i don t know because people can be different behind closed doors  i know this for a fact  He is either an extremely nice but stupid guy or one of the most sickest liars  I hope he is not the latter  
    


```python
# changing all the words to lowercase to create a "bag of words"
lcCleanRev = cleanRev.lower()

# Split to create an array from which  "stop words" will be removed
words = lcCleanRev.split()
```


```python
# Stopwords from nltk are used in this phase
#some stopwords in english language are
## print(stopwords.words("english"))
```

    ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    


```python
#removing most common words from split array
bow = [w for w in words if w not in stopwords.words("english")]

#Bag of Words
print(bow)
```

    ['stuff', 'going', 'moment', 'mj', 'started', 'listening', 'music', 'watching', 'odd', 'documentary', 'watched', 'wiz', 'watched', 'moonwalker', 'maybe', 'want', 'get', 'certain', 'insight', 'guy', 'thought', 'really', 'cool', 'eighties', 'maybe', 'make', 'mind', 'whether', 'guilty', 'innocent', 'moonwalker', 'part', 'biography', 'part', 'feature', 'film', 'remember', 'going', 'see', 'cinema', 'originally', 'released', 'subtle', 'messages', 'mj', 'feeling', 'towards', 'press', 'also', 'obvious', 'message', 'drugs', 'bad', 'kay', 'visually', 'impressive', 'course', 'michael', 'jackson', 'unless', 'remotely', 'like', 'mj', 'anyway', 'going', 'hate', 'find', 'boring', 'may', 'call', 'mj', 'egotist', 'consenting', 'making', 'movie', 'mj', 'fans', 'would', 'say', 'made', 'fans', 'true', 'really', 'nice', 'actual', 'feature', 'film', 'bit', 'finally', 'starts', 'minutes', 'excluding', 'smooth', 'criminal', 'sequence', 'joe', 'pesci', 'convincing', 'psychopathic', 'powerful', 'drug', 'lord', 'wants', 'mj', 'dead', 'bad', 'beyond', 'mj', 'overheard', 'plans', 'nah', 'joe', 'pesci', 'character', 'ranted', 'wanted', 'people', 'know', 'supplying', 'drugs', 'etc', 'dunno', 'maybe', 'hates', 'mj', 'music', 'lots', 'cool', 'things', 'like', 'mj', 'turning', 'car', 'robot', 'whole', 'speed', 'demon', 'sequence', 'also', 'director', 'must', 'patience', 'saint', 'came', 'filming', 'kiddy', 'bad', 'sequence', 'usually', 'directors', 'hate', 'working', 'one', 'kid', 'let', 'alone', 'whole', 'bunch', 'performing', 'complex', 'dance', 'scene', 'bottom', 'line', 'movie', 'people', 'like', 'mj', 'one', 'level', 'another', 'think', 'people', 'stay', 'away', 'try', 'give', 'wholesome', 'message', 'ironically', 'mj', 'bestest', 'buddy', 'movie', 'girl', 'michael', 'jackson', 'truly', 'one', 'talented', 'people', 'ever', 'grace', 'planet', 'guilty', 'well', 'attention', 'gave', 'subject', 'hmmm', 'well', 'know', 'people', 'different', 'behind', 'closed', 'doors', 'know', 'fact', 'either', 'extremely', 'nice', 'stupid', 'guy', 'one', 'sickest', 'liars', 'hope', 'latter']
    


```python
# Function to do the make collection of cleaned text using all the reviews

def review_to_words(raw_review):
    review_text = BeautifulSoup(raw_review,"html.parser").get_text()
    letters_only = re.sub("[^a-zA-Z]"," ",review_text)
    words = letters_only.lower().split()
    
    #create a set of stopwords so that we don't have to access corpus to search for a stopword
    stop = set(stopwords.words("english"))
    
    #removing stopwords from the raw_review
    meaningful_words = [w for w in words if w not in stop]
    
    return(" ".join(meaningful_words))
```


```python
# Just Checking
check_review = review_to_words(train.review[0])
print(check_review)
```

    stuff going moment mj started listening music watching odd documentary watched wiz watched moonwalker maybe want get certain insight guy thought really cool eighties maybe make mind whether guilty innocent moonwalker part biography part feature film remember going see cinema originally released subtle messages mj feeling towards press also obvious message drugs bad kay visually impressive course michael jackson unless remotely like mj anyway going hate find boring may call mj egotist consenting making movie mj fans would say made fans true really nice actual feature film bit finally starts minutes excluding smooth criminal sequence joe pesci convincing psychopathic powerful drug lord wants mj dead bad beyond mj overheard plans nah joe pesci character ranted wanted people know supplying drugs etc dunno maybe hates mj music lots cool things like mj turning car robot whole speed demon sequence also director must patience saint came filming kiddy bad sequence usually directors hate working one kid let alone whole bunch performing complex dance scene bottom line movie people like mj one level another think people stay away try give wholesome message ironically mj bestest buddy movie girl michael jackson truly one talented people ever grace planet guilty well attention gave subject hmmm well know people different behind closed doors know fact either extremely nice stupid guy one sickest liars hope latter
    


```python
#number of reviews
num_reviews = train.review.size
print("number of reviews :",num_reviews)
```

    number of reviews : 25000
    


```python
#storing all reviews in a list
clean_train_reviews = []
for i in range(num_reviews):
    clean_train_reviews.append(review_to_words(train.review[i]))
    if(i%5000==0):
        print("Breathe In... Breathe Out")
print("Cleaning Completed")
```

    Breathe In... Breathe Out
    Breathe In... Breathe Out
    Breathe In... Breathe Out
    Breathe In... Breathe Out
    Breathe In... Breathe Out
    Cleaning complete
    


```python
print("Creating a Bag of Words: ")

# We use CountVectorizer imported from sklearn.feature_extraction.text to create token counts of document


# Setting Parameters as None
vectorizer = CountVectorizer(analyzer="word",
                            tokenizer=None,
                            preprocessor=None,
                            stop_words=None,
                            max_features=5000)

# We train the classifer using fit_transform() method
train_data_features = vectorizer.fit_transform(clean_train_reviews)

#change the classifier into array
train_data_features = train_data_features.toarray()
print(train_data_features.shape)
```

    Creating a Bag of Words: 
    (25000, 5000)
    


```python
#see all the features names
vocab = vectorizer.get_feature_names()
print(" , ".join(vocab[0:10])," . . . . "," , ".join(vocab[-10:]))

```

    abandoned , abc , abilities , ability , able , abraham , absence , absent , absolute , absolutely  . . . .  yet , york , young , younger , youth , zero , zizek , zombie , zombies , zone
    


```python
#frequency of each word is found using np.sum()
dist = np.sum(train_data_features,axis=0)
ct = 0
for tag,count in zip(vocab,dist):
    print(tag,":",count,end="\n ")
```

    


```python
#Check if words starting with any alphabet is missing or not?
startswith = []
for val in vocab:
    if(val[0] not in startswith):
        startswith.append(val[0])
print(startswith)
```

    ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'y', 'z']
    


```python
#counting the total numbers of words starting
#counts = np.zeros((len(startswith)),dtype=np.int)
for val in vocab:
    index = startswith.append(val[0])

```


```python
for i in range(len(counts)):
    print("Alphabet: ",startswith[i]," Word Count: ",counts[i])

```


```python
# Lets do some Plotting
plt.figure(1,figsize=(15,5))
plt.plot(counts)
nums = [i for i in range(26)]
plt.xticks(nums,startswith)
plt.grid()
plt.ylabel("frequency")
print(plt.show())
```


![png](output_19_0.png)


    None
    


```python
# Using Random Forrest Classifier for classification
forest = RandomForestClassifier(n_estimators = 100, criterion = "entropy")
print("Fitting RandomForest . . . ")
forest = forest.fit(train_data_features,train["sentiment"])
```

    Fitting RandomForest . . . 
    


```python
# Using Naive-Bayes

naive = MultinomialNB()
print("Fitting NaiveBayes . . . ")
naive.fit(train_data_features,train["sentiment"])
```

    Fitting NaiveBayes . . . 
    




    MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)




```python
adaboost = AdaBoostClassifier(n_estimators = 100)
print("Fitting AdaBoost . . . ")
adaboost.fit(train_data_features,train["sentiment"])
print("Fitting completed.")
```

    Fitting AdaBoost . . . 
    Fitting completed.
    


```python
#Now lets check against Test Cases
test = pd.read_csv("F:/testData.tsv",header=0,delimiter="\t",quoting=3)
print("Shape :",test.shape)
```

    Shape : (25000, 2)
    


```python
num_reviews = len(test["review"])
clean_test_reviews = []
print("Cleaning and parsing . . . . ")
for i in range(0,num_reviews):
    if((i+1)%5000 == 0):
        print(i+1," reviews processed . . .")
    clean_review = review_to_words(test["review"][i])
    clean_test_reviews.append(clean_review)
print("Processing complete.")
```

    Cleaning and parsing . . . . 
    5000  reviews processed . . .
    10000  reviews processed . . .
    15000  reviews processed . . .
    20000  reviews processed . . .
    25000  reviews processed . . .
    Processing complete.
    


```python
test_data_features = vectorizer.fit_transform(clean_test_reviews)
test_data_features = test_data_features.toarray()
print("Prediction using RandomForest")
result1 = forest.predict(test_data_features)
print("Prediction using Naive Bayes")
result2 = naive.predict(test_data_features)
print("Prediction using AdaBoost")
result3 = adaboost.predict(test_data_features)
print("Completed")
```

    Prediction using RandomForest
    Prediction using Naive Bayes
    Prediction using AdaBoost
    Completed
    


```python
result = result1+result2+result3
for i in range(25000):
    if(result[i]==1):
        result[i]=0
    elif(result[i]==2):
        result[i]=1
    elif(result[i]==3):
        result[i]=1
output = pd.DataFrame(data = {"id":test["id"],"sentiment":result})
output.to_csv("F:/submission.csv", index=False, quoting=3)
```
