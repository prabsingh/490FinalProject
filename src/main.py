from ContextRecommender import *
import numpy as np


def main():

    recommender = ContextRecommender()
    recommender.loadWb()
    recommender.createContextCSR()
    recommender.createModelRecommender("trainCSV.csv", "testCSV.csv")
    recommender.recommend()




main()
