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

## Alternatives
We can improve the accuracy of the above results using the following approaches : 
1) Better data cleaning and data preprocessing steps
2) We could train a neural network for better classification results.
## Finding the similarity measure 
TO find the similarity between a given cuisine and others, I used the cosine similarity coefficient.
## Submitting my code :
I have also made a git tag on my repository with the latest commit :
~~~
git tag v1.0
git push origin v1.0
~~~
