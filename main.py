import json
import os
import re
import numpy as np
import pandas as pd
import sys
import io
import sklearn
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifierimport codecs
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
test_data = []
training_data = []
meal = []
cuisine = []
ingredients = []
individual_ingredients = []
def split_test_and_training(features, labels, percent_training=0.7):
    num_training_instances = int(percent_training * len(labels))

    training_features = features[:num_training_instances]
    test_features = features[num_training_instances:]

    training_labels = labels[:num_training_instances]
    test_labels = labels[num_training_instances:]

    return training_features, training_labels, test_features, test_labels
cuisine_predicted = ''
train_data = pd.read_json('C:/Users/jwala/OneDrive/Desktop/cuisine_train.json')
test_data = pd.read_json('C:/Users/jwala/OneDrive/Desktop/cuisine_test.json')
cv = CountVectorizer()
train_data['concat_ingredients'] = train_data['ingredients'].map(';'.join)
test_data['concat_ingredients'] = test_data['ingredients'].map(';'.join)
train_data.head()
def create_separate_lists(filename):
    with open( filename,encoding = 'utf-8') as req:
        data = json.load(req)
    data_frame = pd.DataFrame({'id':dish_id,'cuisine':cuisine,'ingredients':ingredients})
    data_frame['cuisine'].value_counts()
    for ingr in data_frame['ingredients']:
        ingr = ' '.join(ingr)
        ingredients.append(ingr)
    for i in ingredients:
        individual_ingredients.append(u''.encode('utf-8')
    for meal1 in data_frame['id']:
        meal1=''.join(meal1)
        meal.append(meal1)
    for v in data_frame['cuisine']:
        v = ''.join(v)
        cuisine.append(v)
    return meal,cuisine, individual_ingredients

def KNN_fitdata(training_data,cuisine,number):  
    number = int(number)
    values_close = KNeighborsClassifier(n_neighbors=number)
    values_close(train_data,test_data)
    y_pred = values_close.predict(test_data)
    metrics.accuracy_score(test_data, y_pred)
    values_close.fit(train_data,y_pred)
    return values_close.fit(training_data,cuisine)
X = cv.fit_transform(train_data['concat_ingredients'].values)
X
X_test = cv.transform(test_data['concat_ingredients'].values)
#X_test
id_test = test_data['id']
Y = train_data['cuisine']
Y.head()
def vectorizer_function(existing_ingredients,preferred_ingredients):
    existing_ingredients.append(preferred_ingredients)
    vectorizer = TfidfVectorizer(use_idf = True, stop_words = 'english',max_features = 4000)
    vectorized_ingredients = vectorizer.fit_transform(existing_ingredients)
    return (vectorized_ingredients.todense())
#using 2 classifier model : NaiaveBayes and KNN_Classifier model
NaiveModel = MultinomialNB().fit(X,Y)
Naive_PredictedY1s = NaiveModel.predict(X)
print("Naive Bayes Accurracy : %f " % np.mean ( Naive_PredictedY1s == Y))
print(classification_report(Naive_PredictedY1s, Y))
Model1 = RandomForestClassifier(max_depth=40, n_estimators=20).fit(X,Y)
Predicted_Cuisines = Model1.predict(X_test)

ing_array = ["baking powder;eggs;all-purpose flour;raisins;milk;white sugar"]
no_of_ingredients = input("Total Number Of Ingredients that can be possible are 3: ")
no_of_ingredients = int(no_of_ingredients)

ingredient = ""

for i in range(no_of_ingredients):
    ing = input("Enter Ingredient " + str(i) + " : ")
    ingredient = ingredient + ing + ";"

## Predicting User input ingredients

ing_array.append(ingredient)
User_in = cv.transform(ing_array)
Predicted = Model1.predict(User_in)
print("")
print("The predicted cuisine for input ingredients is : "+Predicted[1])
def KNN_classifier(test_data,nearby,neighbours):
    neighbours = int(neighbours)
    cuisine_predicted = nearby.predict_proba(test_data)[0]
    predicted_single_cuisine = nearby.predict(test_data)
    predicted_class = nearby.classes_
    print ("Cuisine %s" %(predicted_single_cuisine[0]))
"""
#printing the individual cuisine id and the cuisine values predicted as a dataframe
train.groupby('cuisine').num_ingredients.describe()
X = train.ingredients_str
y = train.cuisine
vect = CountVectorizer()
X_dtm = vect.fit_transform(X)
vect = CountVectorizer(token_pattern=r"'([a-z ]+)'")
knn = KNeighborsClassifier(n_neighbors=100)
knn.fit(train_data,test_data)
y_pred=knn.predict(test_data)
knn.fit(X,y)
X_new = new[feature_cols]
new_pred_class_knn = knn.predict(X_new)
new_pred_class_knn
#predicted the cuisine
"""
    for i in range(len(cuisine_predicted)):
        if not(cuisine_predicted[i] == 0.0):
            print ("Cuisine: %s(%f)" %(predicted_class[i],cuisine_predicted[i]*100))
    print ("Closest %d recipes : " % neighbours)
    match_perc,match_id = nearby.kneighbors(test_data)
    for i in range(len(match_id[0])):
        print ((meal[match_id[0][i]])) 


def classified_data(whole_data):
    training_data = whole_data[:len(whole_data)-1]
    test_data = whole_data[len(whole_data)-1]
    #train_data = pd.read_json('C:/Users/jwala/OneDrive/Desktop/cuisine_train.json')
    #test_data = pd.read_json('C:/Users/jwala/OneDrive/Desktop/cuisine_test.json')
    return (training_data,test_data)
        if not(cuisine_predicted[i] == 0.0):
            print ("Cuisine: %s(%f)" %(predicted_class[i],cuisine_predicted[i]*100))
    print ("Closest %d recipes : " % neighbours)
    match_perc,match_id = nearby.kneighbors(test_data)
    for i in range(len(match_id[0])):
        print ((meal[match_id[0][i]]))
    X_train_transformed = training_data.transform(training_data.values) #applying vectorizer for x-values of train set
X_test_transformed = training_data.transform(test_data.values)   #applying vectorizer for x-values of test set
def sub_function():
    preferred_ingredients = input("Enter the ingredients that you would want to match your cuisine with ")
    whole_data = vectorizer_function(individual_ingredients,preferred_ingredients)
    training_data,test_data = classified_data(whole_data)
    #Fitting the data to find the 5 closest recipes
    nearby = KNN_fitdata(training_data,cuisine,5)
    KNN_classifier(test_data,nearby,5)
    individual_ingredients.pop()
if __name__ == '__main__':
    create_separate_lists(filename = ("yummly.json"))
    sub_function()   
