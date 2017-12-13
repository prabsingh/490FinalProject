from openpyxl import load_workbook
from CSR import *
from ContextRecommender import *
import os.path
import numpy as np


def getUsers(train):
    user_list = []

    user_column = train['A']
    for cell in range(2, len(user_column)):
        if user_column[cell].value not in user_list:
            user_list.append(user_column[cell].value)

    return user_list


def main():

    recommender = ContextRecommender()
    recommender.loadWb()
    recommender.createContextCSR()
    recommender.createModelRecommender("trainCSV.csv","testCSV.csv")
    recommender.recommend()

    ''' data_set = np.loadtxt('csvMusic.csv', delimiter=',')
    row, cols = data_set.shape

    drivingCSR = CSR()
    csvDriving = CSR()

    wb = load_workbook('Data_InCarMusic.xlsx')
    ws = wb['TrainSet']
    user = getUsers(ws)

    drivingCSR.build_from_excel(ws, user, 4)
    csvDriving.build_from_numpy(data_set, user, 3)

    spec = CSR()
    spec2 = CSR()
    spec.build_no_context_numpy(data_set, user)
    print(spec.row_ptr)
    spec2.build_no_context(ws, user)
    print(spec2.row_ptr)'''




main()
