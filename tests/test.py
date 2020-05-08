import pytest
import Project3

from Project3 import main
file = open('yummly.json')
def test_data():
    x, cuisine_values = main.create_separate_lists(file)   
    assert len(cuisine_values) == 1
    y, meal_values = main.create_separate_lists(file)
    assert len(meal_values)==1
    z, indiv_ingredients = main.create_separate_lists(file)
    assert len(indiv_ingredients)==1
