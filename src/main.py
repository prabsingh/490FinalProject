from openpyxl import load_workbook
from CSR import *
from ContextRecommender import *
import os.path



def main():

    recommender = ContextRecommender()
    recommender.loadWb()
    recommender.createContextCSR()
    recommender.recommend()
    print("Done")


main()
