import pandas as pd
import numpy as np
import sklearn
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import cross_val_score
#from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
#from sklearn.metrics import accuracy_score #for using to test my pipeline test and train, vectorizer to classifiers which one is best combo
import nltk
from sklearn.tree import DecisionTreeClassifier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
import random as rd
import csv
from sklearn import preprocessing
#nltk.download()

#not used anymore after 1st submission.
def positivitytester(score):
    if (score > 3):
        #4 or 5 score good
        return 1
    elif (score == 3):
        return 0
    else:
        # 1 or 2 so bad score
        return -1
    
#given a text predict the score
#def predictscore(text):
    #a score from 0 to 5
    #return 0

#when given a positivity list, assign a score 1 to 5 based on probability from the training set
#not used anymore
def assignScoreBasedOnProbability(positivityList, oneScore, twoScore, threeScore, fourScore, fiveScore):
    ls = []
    totalScores = oneScore + twoScore + threeScore + fourScore + fiveScore
    oneProb = (oneScore)/(oneScore+twoScore)
    fiveProb = (fiveScore)/(fiveScore + fourScore)
    for pos in positivityList:
        prob = rd.random()
        if (pos == 1):
            if (prob < fiveProb):
                ls.append(5)
            else:
                ls.append(4)
        elif (pos == 0):
            ls.append(3)
        elif (pos == -1):
            if (prob < oneProb):
                ls.append(1)
            else:
                ls.append(2)
    return ls
def main():
    #My Original Goals:
    #apply LSA on data on the summary or text version.
    #k = 5 groups for ratings 0 to 5
    #can start with smaller test set's SVD
    #can give weights based on helpfulness score as a multiplier and a metric for more
    #accurate scoring.
    #Use the svd result for pairing score of 0 to 5
    #Maybe group 4 to 5 as positive, and group 0 to 2 as negative score.
    #hard to gauge 3 scores...
    #apply some classifier algo with the vectorized features
    #My first step, grab and sanitize the train data. Uninclude the ones that have no score
    #We train on the ones that have a score
    #next group 0 to 5 and append texts to it based on the scoring and its text.
    #treat each document as a text
    #try it on 1000 first random samples first
    #200 of each score?
    #score of 5
    #score of 4
    #score of 3
    #score of 2
    #score of 1
    #What I actually did in the end:
    #Whats done is just testing sklearn classifiers against either count vectorizer or tfidf vectorizer.


    #32 bit python attempt #bad idea
    #Parse datas in rows of 20000 since thats a good number my memory can take from reading 32bit python
    data = pd.read_csv("train.csv")
    #data = pd.read_csv("train.csv", header=0,sep=',', skiprows = 0, nrows = 20000)
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,20001),nrows=20000)
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,40001),nrows=20000)
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,60001),nrows=20000)
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,80001),nrows=20000)
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,100001),nrows=20000)
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,120001),nrows=20000)
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,140001),nrows=20000)
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,160001),nrows=20000)
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,180001),nrows=20000)
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,200001),nrows=20000)
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,220001),nrows=20000)
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,240001),nrows=20000)
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,260001),nrows=20000)
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,280001),nrows=20000)
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,300001),nrows=20000)
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,320001),nrows=20000)
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,340001),nrows=20000)
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,360001),nrows=20000)
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,380001),nrows=20000)
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,400001),nrows=20000)
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,420001),nrows=20000)
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,440001),nrows=20000)
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,460001),nrows=20000)
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,480001),nrows=20000)
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,500001),nrows=20000)
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,520001),nrows=20000)
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,540001),nrows=20000)
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,560001),nrows=20000)
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,580001),nrows=20000)
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,600001),nrows=20000)
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,620001),nrows=20000)
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,640001),nrows=20000)
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,660001),nrows=20000)
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,680001),nrows=20000)
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,700001),nrows=20000)
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,720001),nrows=20000)
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,740001),nrows=20000)
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,760001),nrows=20000)
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,780001),nrows=20000)
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,800001),nrows=20000)
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,820001),nrows=20000)
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,840001),nrows=20000)
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,860001),nrows=20000)
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,880001),nrows=20000)
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,900001),nrows=20000)
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,920001),nrows=20000)
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,940001),nrows=20000)
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,960001),nrows=20000)
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,980001),nrows=20000)
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,1000001),nrows=20000)
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,1020001),nrows=20000)
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,1040001),nrows=20000)
    #data = pd.read_csv("train.csv", sep=',', skiprows=range(1, 1001), nrows = 1000)
    #data = pd.read_csv("train.csv", sep=',', skiprows=range(1, 2001), nrows = 1000)


    #upgraded to 64 bit python on desktop...
    #data = pd.read_csv("train.csv", sep=',', nrows = 100000) #1.csv
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,100001),nrows=100000) #2.csv
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,200001),nrows=100000) #3.csv
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,300001),nrows=100000) #4.csv
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,400001),nrows=100000) #5.csv
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,500001),nrows=100000) #6.csv
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,600001),nrows=100000) #7.csv
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,700001),nrows=100000) #8.csv
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,800001),nrows=100000) #9.csv
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,900001),nrows=100000) #10.csv
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,1000001),nrows=100000) #11.csv
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,1100001),nrows=100000) #12.csv
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,1200001),nrows=100000) #13.csv
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,1300001),nrows=100000) #14.csv
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,1400001),nrows=100000) #15.csv
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,1500001),nrows=100000) #16.csv
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,1600001),nrows=100000) #17.csv

    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,1695501),nrows=2500)

    #try number 3 4 5 6 7...
    #data = pd.read_csv("train.csv", sep=',', nrows = 300000) #51.csv #61 #71 #81 91 111 121
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,300001),nrows= 300000) #52.csv  #62 #72 #82 92 112 122
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,600001),nrows= 300000) #53.csv #63 #73 #83 93 113 123
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,900001),nrows= 300000) #54.csv #64 #74 #84 94 114 124
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,1200001),nrows= 300000) #55.csv #65 #75 #85 95 115 125
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,1500001),nrows= 200000) #56.csv #66 #76 #86 96 116 126

    #data = pd.read_csv("train.csv", sep=',', nrows = 500000)
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,500001),nrows= 500000) #131
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,1000001),nrows= 500000) #132
    #data = pd.read_csv("train.csv", header=0, sep=',', skiprows=range(1,1500001),nrows= 200000) #133
    #max data points is 1,697,533
    df = pd.DataFrame(data)
    #print(df[0:10])
    #print(len(df))
    #print(df)
    #df = df[df['Text'].notna()] #make it so it skips rows that are nan text values
    #df.replace("", float("NaN"), inplace = True) #fills empty scores with NaN
    #df.dropna(subset = ['Score'], inplace= True) #drops those empty scores from table
    
    #print(df[0:10])
    #print(len(df))
    #print(df)
    #print(df.shape)
    #tfidf = TfidfVectorizer()
    #df['positivity'] = [positivitytester(score) for score in df['Score']] #set score to positvity
    #print(df['Score'].value_counts()) #prints the score values of each
    #print(df['positivity'].value_counts()) #prints the postivity values of each

    #oneScore = (df['Score'] == 1).sum()
    #twoScore = (df['Score'] == 2).sum()
    #threeScore = (df['Score'] == 3).sum()
    #fourScore = (df['Score'] == 4).sum() #and 7 years ago
    #fiveScore = (df['Score'] == 5).sum()
    #print(oneScore)
    #print(twoScore)
    #print(threeScore)
    #print(fourScore)
    #print(fiveScore)
    
    
    #print(df)
    #df2 = pd.DataFrame(data)
    #print(df2)
    #df2 = df2[df2['Score'] == ''] #keep the rows with empty scores
    #df2 = df2[df2['Text'].notna()]
    #print(df2)
    #dataF = df2[['Id', 'Score','Text']]
    #print(dataF)
    #df3 = dataF[dataF.isna().any(axis=1)] #keep all the empty scores

    #keep all the empty scores

    toPredict = df[df.isnull().Score]
    toPredictIds = toPredict['Id']
    toPredictIds = toPredictIds.reset_index(drop = True, inplace = False) #makes it so the Id is for later use row identifiable.
    toPredict = toPredict['Text'].replace(np.nan, '',regex=True)
    #print(df3)
    #toPredictIds = df3['Id']
    print(toPredictIds)
    print(toPredictIds.loc[0])

    df.dropna(inplace=True)
    #print(df3)
    #print(toPredictIds.loc[0, 'Id'])
    #print(df3.loc[5, 'Id'])
    #print(toPredictIds.to_n)
    #toPredict = df3["Text"]
    #print(toPredict)
    
    #print(df['positivity'])
    #downloaded with nltk to use the below
    #vectorizer = TfidfVectorizer(max_features = 300, stop_words='english', min_df=0.1, max_df=0.8) #not sure how well I can tune tfidf min_df and max_df, also takes too long.
    #stemmed_data = [" ".join(SnowballStemmer("english", ignore_stopwords=True).stem(word)  
        #for sent in sent_tokenize(message)
        #for word in word_tokenize(sent))
        #for message in df["Text"]] #tokenize each data which are basically documents of texts
    """
    textColumn = df["Text"]
    #print(textColumn.head())
    tfidf.fit(textColumn)
    X = tfidf.transform(textColumn)
    print(tfidf.get_feature_names()[:100])
    #print(textColumn[1])
    #print(X[1])
    #print(X[1, tfidf.vocabulary_["good"]])
    #print(X[1, tfidf.vocabulary_["not"]])
    #print(textColumn[0])
    #print(df.head(10))
    """
    #ks = vectorizer.fit(stemmed_data)
    #use the dtm
    #dtm = vectorizer.fit_transform(stemmed_data)
    #generate predictive data with niave bayes first
    #terms = vectorizer.get_feature_names()
    #centered_dtm = dtm - np.mean(dtm, axis=0)
    #print(terms)

    X = df["Text"] #len 86 at 100
    #y = df["positivity"]
    y = df["Score"]
    #print(len(X)) #86
    #X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0) #no test size is 0.25
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state=0)
    #print(len(X_train[y_train == 1])/len(X_train))
    #print(len(X_train[y_train == 0])/len(X_train))
    #print(len(X_train[y_train == -1])/len(X_train))
    #print(len(X_train)) #64 at 100
    #print(len(X_test)) #22 at 100
    #print(len(X_test[y_test == 1])/len(X_test))
    #print(len(X_test[y_test == 0])/len(X_test))
    #print(len(X_test[y_test == -1])/len(X_test))

    #vzX_train = vectorizer.fit_transform(X_train.astype(str))
    #vzX_test = vectorizer.fit_transform(X_test.astype(str))
    rf = RandomForestClassifier(class_weight="balanced")
    mnb = MultinomialNB() #use of naive bayes #1
    """
    Documentations states:
    The multinomial Naive Bayes classifier is suitable for classification with discrete features
    (e.g., word counts for text classification). The multinomial distribution normally requires integer feature counts.
    However, in practice, fractional counts such as tf-idf may also work.
    """
    vectorizer = TfidfVectorizer(max_features = 1000, stop_words='english', min_df=2, ngram_range=(1,2)) #not sure how well I can tune tfidf min_df and max_df, also takes too long.
    #stemmed_data = [" ".join(SnowballStemmer("english", ignore_stopwords=True).stem(word)  
        #for sent in sent_tokenize(message)
        #for word in word_tokenize(sent))
        #for message in X_train] 

    #print(stemmed_data)
    #ks = vectorizer.fit(stemmed_data)
    #print(ks)
    #use the dtm
    #dtm = vectorizer.fit_transform(stemmed_data)
    #generate predictive data with niave bayes first
    #terms = vectorizer.get_feature_names()
    #centered_dtm = dtm - np.mean(dtm, axis=0)
    #u, s, vt = np.linalg.svd(centered_dtm)

    #cv = CountVectorizer(max_features=1000,stop_words='english', ngram_range=(1, 3), min_df=4, max_df=0.8) #3
    #cv = CountVectorizer(max_features=3000,stop_words='english', min_df=0.2, max_df=0.7) #3
    #cv = CountVectorizer(max_features=3000,stop_words='english', min_df=0.1, max_df=0.8) #3
    #cv = CountVectorizer(stop_words='english', )
    #cv = CountVectorizer(max_features=300,stop_words='english') #3
    cv = CountVectorizer(stop_words='english', min_df=2, ngram_range=(1,2))

    dc = DecisionTreeClassifier()
    #pca = PCA(n_components=300, whiten=True)
    #svc = SVC(kernel='rbf', class_weight='balanced', C=5, gamma=0.001)
    #svc = SVC(kernel='rbf', class_weight='balanced', C=5)
    svc = LinearSVC()
    #pcd = pca.fit(center_dtm)

    #Could had bagged classifiers but that takes an enormous amount of time just to generalize...

    #svctf = make_pipeline(ks, svc)

    #vectorizer or cv and rf earlier for pipeline
    #pipeline = Pipeline([('vectorizer', cv),('classifier', mnb)]) #was the 2nd one scored online.
    #pipeline = Pipeline([('vectorizer', ks),('classifier', dc)])#originally itended to use tfidf vectorizer on the stemmed data but takes too long and also almost always gave 4 and 5s
    #pipeline = Pipeline([('vectorizer', ks),('classifier', dc)])#1 using decision trees since it seems more accurate when used with ks vectorizer

    #pipeline = Pipeline([('vectorizer', ks),('classifier', svc)]) #71 svc for SVC takes roughly more than 8 hours to run on 300,000 rows not practical given the amount of time cant test
    #pipeline = Pipeline([('vectorizer', ks),('classifier', dc)]) #61
    #sentiment_fit = pipeline.fit(X_train.values.astype('U'), y_train)
    #sentiment_fit = pipeline.fit(X_train.values.astype(str), y_train) #1

    #sentiment_fit = pipeline.fit(X_train.values.astype(str), y_train) #3 #4 #6 #71
    #sentiment_fit = pipeline.fit(vzX_train, y_train) #doesnt work
    #y_pred = sentiment_fit.predict(X_test)
    #y_pred = sentiment_fit.predict(toPredict.values.astype('U'))

    #y_pred = sentiment_fit.predict(toPredict.values.astype(str))#3

    #trying logregression
    #logreg = LogisticRegression(n_jobs=1, C=1e5)
    logreg = LogisticRegression(random_state=0)

    #print('accuracy %s' % accuracy_score(y_pred, y_test)) #from sklearn documentation to check accuracies
    #print(y_pred)

    #scaler = preprocessing.StandardScaler()
    #X_train_scaled = preprocessing.scale(X_train)
    #y_train_scaled = preprocessing.scale(y_train)
    #Predict_scaled = preprocessing.scale(toPredict)
    #X_scaled = scaler.transform(X_train)

    #linear regression
    #reg = LinearRegression()
    #pipeline = Pipeline([('vectorizer', cv),('classifier', reg)])
    #sentiment_fit = pipeline.fit(X_train.astype(str), y_train)
    #y_pred = sentiment_fit.predict(toPredict.values.astype(str))


    # Create a TruncatedSVD instance: svd
    #svd = TruncatedSVD(n_components= 100)
    # Create a KMeans instance: kmeans
    #kmeans = KMeans(n_clusters=5) #for 5 different scores
    # Create a pipeline: pipeline
    #pipeline = make_pipeline(svd, kmeans)
    #pipeline = make_pipeline(svd, svc) #realized its hard to classify the labels...
    
    #mattrx = vectorizer.fit_transform(X_train)
    #mattrxy = vectorizer.fit_transform(y_train)
    #ytrnmatrx = vectorizer.fit_transform(y_train)
    #mattrxpred = vectorizer.fit_transform(toPredict)
    # Fit the pipeline to articles
    #pipeline.fit(mattrx, mattrxy)
    #pipeline.fit(X_train, y_train)
    # Calculate the cluster labels: labels
    #y_pred = pipeline.predict(toPredict) #I also tried Kneighbors, kmeans but I couldnt get the labels to go back to scorings. Thats why i stuck with sklearn classifiers intead.

    #pipeline = Pipeline([('vectorizer', ks),('classifier', svc)]) #81 too long 9+hours for 300,000rows
    #pipeline = Pipeline([('vectorizer', ks),('classifier', logreg)]) #91 too long 9+hours for 300,000rows
    #pipeline = Pipeline([('vectorizer', cv),('classifier', logreg)]) #111
    #sentiment_fit = pipeline.fit(X_train.astype(str), y_train)
    #y_pred = sentiment_fit.predict(toPredict.values.astype(str))


    #Final combinations to use
    #vk1 = vectorizer.fit(X_train) #tfidf vectorizer takes too long, do these submissions last #131
    vk1 = cv.fit(X_train)
    print("vectorizer done")
    vk2 = vk1.transform(X_train)
    print("transform done")
    #applying my final picks on classifying models
    md = svc.fit(vk2, y_train) #svc 
    #md = logreg.fit(vk2, y_train) #logistic regression
    #md = mnb.fit(vk2, y_train) #multinomial niave bayes
    y_pred = md.predict(vk1.transform(toPredict))
    print("predict done")
    #print(y_pred)
    #print('accuracy %s' % accuracy_score(y_pred, y_test)) #from sklearn documentation to check accuracies
    #accuracy_score1 = #from s#
    #Use to find Found better score for in terms of correctness in combination for rows of 2500 samples for my train to tests
    #print(accuracy_score1)

    #pipe = make_pipeline([('vectorizer', dtm),('classifier', mnb)])
    #y_pred = pipe.fit_predict(toPredict)
    #print(y_train.head())
    #make pipeline function

    #print(X_train)
    #print(y_train)
    #print(toPredict)
    #print(y_test)
    #print(y_pred)

    #toWriteToCsv = assignScoreBasedOnProbability(y_pred, oneScore, twoScore, threeScore, fourScore, fiveScore)
    #print(toPredictIds)
    #print(len(toPredictIds))
    #print(toWriteToCsv)
    #print(len(toWriteToCsv))  

    #writing to each row base on id and score
    idLs = []
    file = open('201.csv', 'w', newline ='') 
  
    with file: 
        # identifying header   
        header = ['Id', 'Score'] 
        writer = csv.DictWriter(file, fieldnames = header) 
        writer.writeheader()
        #for i in range(len(toWriteToCsv)):
        for i in range(len(y_pred)):
            idNum = toPredictIds.loc[i] #grabbing the id for the respectively score it represents
            theSc = y_pred[i]
            writer.writerow({'Id' : idNum, 'Score' : theSc}) 

            #idLs.append(idNum)

    #print(idLs)
    
    
    #print(df2)
    #print(len(df2))
    #print(df2['Score'])
    #
    #print(u)
    #print(s)
    #print(vt)
    #print(dtm)
    #print(len(dtm))
    #print(len(terms))


    
main()
