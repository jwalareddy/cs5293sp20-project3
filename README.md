# cs5293sp20-project3
The Analyzer

## Description of the Project
The goal of the project is to create an application that take a list of ingredients from a user and attempts to predict the type of cuisine and similar meals. Consider a chef who has a list of ingredients and would like to change the current meal without changing the ingredients. The steps to develop the application should proceed as follows.

1) Pre-train or index all necessary classifiers using the existing datasets.
2) Ask the user to input all the ingredients that they are interested in.
3) Use the model to predict the type of cuisine and tell the user.
4) Find the top N closest foods (you can define N). Return the IDs of those dishes to the user. If a dataset does not have IDs associated with them you may add them arbitrarily.

## Analyzing the initial data
The initial data is obtained from the following link:
https://www.dropbox.com/s/f0tduqyvgfuin3l/yummly.json
It includes the master list of all possible dishes, their ingredients, an identifier, and the cuisine for different dishes. The data is stored in the yummly.json file in the form of a dictionary. One of the sample entries in the dataset is as follows : 
~~~
[
  {
    "id": 10259,
    "cuisine": "greek",
    "ingredients": [
      "romaine lettuce",
      "black olives",
      "grape tomatoes",
      "garlic",
      "pepper",
      "purple onion",
      "seasoning",
      "garbanzo beans",
      "feta cheese crumbles"
    ]
  }
]
~~~
## Function1 : Reading the data
Initially I loaded the json file giving the directory where it was stored on my local machine. Since it is in a dictionary format, I convert it into individual lists for each of id, cuisine and ingredients and used them for the next functions to be implemented.

## Data Cleaning approaches
I followed the following text cleaning procedures :
1) converted all entries to lowercase
2) remove the special characters, digits and other punctuations
~~~
ingredient.lower()                               
ingredient = re.sub("[^a-zA-Z]"," ",ingredient)   
~~~
## Classification algorithm
I used 2 different aprroaches to build my classification model. Initially I used the KNN classification algorithm to predict the nearby ingredient cuisines
For the testing and training dataset, I used the following import statement :
~~~
from sklearn.model_selection import train_test_split
~~~
I used 70% as the training data and 30% as the testing data for my dataset.
The function is as follows : 
~~~
def split_test_and_training(features, labels, percent_training=0.7):
~~~
I take the input from the n closest inputs from the user and trained the model using the KNearest Neighbours classifier. 
Also for a different approach that I also used, I gave the training set as the same yummly.json file and the testing dataset, I used the same dataset except for removing the cuisine label as it is to be predicted in the project.
My code is running for the both the different approaches used.

## Prediction code(Count Vectorizer)
~~~
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
~~~
## Screenshots of successful execution of my program(Module1)
This output is when I try to run the program using the KNN classifier model. After giving the command python main.py, the control goes to the list_creator function which returns the list of the individual items in the json dataset. On calling the main function which has the logic for user input, initially it creates the vecctorized form for the ingredients and returns that value. Based on the test and train data set and the logic for the KNN classifier, it predicts the number of closest items and the cuisine. 
![image](https://user-images.githubusercontent.com/27561736/81335583-72cb9100-906d-11ea-9475-e51940aceed8.png)

I have also implemented a similar approach on my Jupyter notebook to predict the cuisine based on the number of ingredients that we give. This separate program used the NaiveBayes classifier for classification and prediction. 

## Screenshots of successful execution of my program(Module2)
![image](https://user-images.githubusercontent.com/27561736/81346639-3fdec880-9080-11ea-81a1-8d7a17fce4eb.png)


## Accuracy of the program implementation
Accuracy obtained by using KNN classification model is 72%
I have attached the screenshot which shows the accuracy of the Naive Bayes classifier used (since it is a Jupyter notebook implementation):
![image](https://user-images.githubusercontent.com/27561736/81348974-b087e400-9084-11ea-889a-479a7acc233b.png)

![image](https://user-images.githubusercontent.com/27561736/81365211-2607aa80-90ad-11ea-8c8e-1cdb5abcc671.png)

## Turning text into features
To turn the text into features for the 2 modules that I chose, I used the CountVectorizer and the Tf-idf vectorizer. The Tf-idf vectorizer is based on the Bag of Words model.
~~~
vectorizer = TfidfVectorizer(use_idf = True, stop_words = 'english',max_features = 4000)
~~~
The reason for choosing the Tf-idf vectorizer is that it makes the computation process easier. In my dataset, I am extracting the individual ingredients and turning each of that into the vectorized form, so it extracts these particular terms relatively easy.
To predict the n-closest reciptes to the cuisine that has been generated and this vectorizer makes this process easy ( the similarity computation between 2 entities is easier)
In my oter model approach, I tried it with using the Count Vectorizer and it is primarily used for its better skewness
## What classifiers/clustering methods I choose and why?
As in the screenshots that I have updated, I have used 2 models KNN Classifier and Naive Bayes model. For KNN classifier, it requires training data which makes it faster than other classification algorithms. 
It does not require the training concept before making predictions, and when i change my test and train data, though this wil not affect the accuracy of the algorithm implemented.
Another reason to select this is for its easy implementation for the lesser number of arguments that it requires and can be used easily as an input to other algorithm mechanisms.
In another approach, I used Naive Bayes model. For this approach, I considered a very small sample of train.json file. For such small data, it is easier to implement considering the size.
For smaller data, the implementation was fast and was easier to make predictions for the cuisine generation.
I assumed scalability to be a good advantage as well.

## What N did I choose ?
In my code, there is a subsection that mentions the N closest that I would want to predict.
~~~
KNN_classifier(test_data,nearby,5)
~~~
In the KNN_Classifer function, I used the predict_proba function to get the probability estimates of the test data that I have passed as an input. Alternately, I can also specify this as a user defined argument, where the user will have the ability to generate the nearest n recipes and n is any arbitary value that the user wants to enter. But in my program i specified it as 5.
I tried giving the user input to enter the number of closest recipes that he would want to find :
~~~
 user_input = input("Enter n where n is the number of closest recipes that you would want to find")
 nearby = KNN_fitdata(training_data,cuisine,user_input)
~~~
So here based on the user_input, the KNN model is fitted and the user_input number of closest recipes are outputted.
## Alternatives
We can improve the accuracy of the above results using the following approaches : 
1) Better data cleaning and data preprocessing steps
2) We could train a neural network for better classification results.
## Finding the similarity measure 
TO find the similarity between a given cuisine and others, I used the cosine similarity coefficient.

## Tests
I defined my tests to verify the length of the ingredients, id and the cuisine generated when loaded as a dataframe.
~~~
x, cuisine_values = main.create_separate_lists(file)
assert len(cuisine_values) == 1
y, meal_values = main.create_separate_lists(file)
assert len(meal_values)==1
z, indiv_ingredients = main.create_separate_lists(file)
assert len(indiv_ingredients)==1
~~~
## Submitting my code :
I have also made a git tag on my repository with the latest commit :
~~~
git tag v1.0
git push origin v1.0
~~~
