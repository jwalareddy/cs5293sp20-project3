# cs5293sp20-project3
The Analyzer

## Description of the Project
The goal of the project is to create an application that take a list of ingredients from a user and attempts to predict the type of cuisine and similar meals. Consider a chef who has a list of ingredients and would like to change the current meal without changing the ingredients. The steps to develop the application should proceed as follows.

1) Pre-train or index all necessary classifiers using the existing datasets.
2) Ask the user to input all the ingredients that they are interested in.
3) Use the model to predict the type of cuisine and tell the user.
4) Find the top N closest foods (you can define N). Return the IDs of those dishes to the user. If a dataset does not have IDs associated with them you may add them arbitrarily.

## Function1 : Reading the data
Initially I loaded the json file giving the directory where it was stored on my local machine. Since it is in a dictionary format, I convert it into individual lists for each of id, cuisine and ingredients and used them for the next functions to be implemented.

## Vectorized form of the ingredients
Next, I used the Tf-idf vectorizer to turn in the ingredients in a vectorized form.

## Classification algorithm
I used 2 different aprroaches to build my classification model. Initially I used the KNN classification algorithm to predict the nearby ingredient cuisines
For the testing and training dataset, I used the following import statement :
~~~
from sklearn.model_selection import train_test_split
~~~
I used 90% as the training data and 10% as the testing data for my dataset.
Also for a different approach that I also used, I gave the training set as the same yummly.json file and the testing dataset, I used the same dataset except for removing the cuisine label as it is to be predicted in the project.
My code is running for the both the different approaches used.

## Screenshots of successful execution of my program

