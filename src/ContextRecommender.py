from openpyxl import load_workbook
from CSR import *
import os.path
import math
import numpy as np


class ContextRecommender:
    def __init__(self):
        self.wb = None
        self.train = None
        self.test = None
        self.contextList = []

    def loadWb(self):
        # load wb on mac
        self.wb = load_workbook(os.path.dirname(__file__) + '/Data_InCarMusic.xlsx')
        #self.train = data_set = np.loadtxt(os.path.dirname(__file__) + '/trainCSV.csv', delimiter=',')
        #self.test = data_set = np.loadtxt(os.path.dirname(__file__) + '/testCSV.csv', delimiter=',')

        # load wb on windows
        #self.wb = load_workbook(os.path.dirname(__file__) + '\\Data_InCarMusic.xlsx')
        # self.train = data_set = np.loadtxt(os.path.dirname(__file__) + '\\csvMusic.csv', delimiter=',')

        self.train = self.wb["TrainSet"]
        self.test = self.wb["TestSet"]

    # return the list of unique users
    def getUsers(self):
        user_list = []

        user_column = self.train['A']
        #col_len = self.train.shape[0]
        for cell in range(2, len(user_column)):
        #for cell in range(0, col_len):
            if user_column[cell].value not in user_list:
            #if self.train[cell][0] not in user_list:
                user_list.append(user_column[cell].value)
                #user_list.append(self.train[cell][0])

        return user_list

    def createContextCSR(self):
        user_list = self.getUsers()

        # for all the dimension create a CSR and add to List
        for dim in range(4, 12):
            dimCSR = CSR()
            dimCSR.build_from_excel(self.train, user_list, dim)
            self.contextList.append(dimCSR)

        # create a special CSR where no context was given
        noContextCSR = CSR()
        noContextCSR.build_no_context(self.train, user_list)
        self.contextList.append(noContextCSR)

    def user_item_avg(self, user, item):
        # avg across all items
        allSum = 0.0
        totalItemsRated = 0

        # avg across given item
        itemSum = 0.0
        numItemRated = 0

        # loop through all context
        for context in self.contextList:
            # for each item that user rated
            for i in range(context.row_ptr[user - 1], context.row_ptr[user]):
                # accumulated for total Avg
                allSum += context.context[i - 1]
                totalItemsRated += 1

                if (context.col_ind[i - 1] == item):
                    # accumlate for item Avg
                    itemSum += context.context[i - 1]
                    numItemRated += 1

        # if the item has not been rated by the user
        # then return the overall avg

        if (numItemRated == 0):
            if (totalItemsRated == 0):
                return 2.5
            else:
                return allSum / totalItemsRated
        else:
            return itemSum / numItemRated

    def findContextWeight(self, user, item, dim, contextValue):
        cumulativeWeight = 0.0
        numUsers = 0

        transposeDim = self.contextList[dim].transpose()

        # for all users that have rated the item in the context
        for j in range(transposeDim.row_ptr[item - 1], transposeDim.row_ptr[item]):
            if (transposeDim.val[j] == contextValue):
                otherUser = transposeDim.col_ind[j]

                # find the change in rating and weight it based on User-User similarity
                changeInRating = self.user_item_avg(otherUser, item) - transposeDim.context[j]
                uuSim = self.contextList[dim - 4].calc_user_user_sim(user, otherUser)
                cumulativeWeight += (changeInRating * uuSim)
                numUsers += 1

        if (numUsers != 0):
            return (cumulativeWeight / numUsers)
        else:
            return -1

    def recommend(self):
        # open and close the file to clear it
        f = open("output.txt", 'w')
        f.close()

        sumSquareError = 0.0
        absoluteError = 0.0
        numRating = 0.0

        numRows = len(self.test["A"])
        for i in range(2, numRows + 1):
            user = self.test.cell(row=i, column=1).value
            item = self.test.cell(row=i, column=2).value

            userBaseline = self.user_item_avg(user, item)

            overallWeight = 0.0
            numDim = 0.0
            # for each dimension that has a value
            for dim in range(0, len(self.contextList) - 1):

                contextValue = self.test.cell(row=i, column=dim + 4).value
                if (contextValue != 0):
                    contextWeight = self.findContextWeight(user, item, dim, contextValue)

                    if (contextWeight != -1):
                        overallWeight += contextWeight
                        numDim += 1

            # also look at the CSR with no context given
            # Since we do not know what the context given is,
            # it is given a weight of 1/2 of a regular context
            noContextWeight = .5

            weight = self.findContextWeight(user, item, len(self.contextList) - 1, 0)

            if (weight != - 1):
                overallWeight += (weight * noContextWeight)
                numDim += noContextWeight

            # prediction is the baseline + avg of the contextWeight
            contextModifier = 0
            if (numDim != 0):
                contextModifier = overallWeight / numDim

            predictRating = userBaseline + contextModifier
            actualRating = int(self.test.cell(row=i, column=3).value)
            diff = 0
            if (actualRating > predictRating):
                diff = actualRating - predictRating
            else:
                diff = predictRating - actualRating

            sumSquareError += (actualRating - predictRating) ** 2
            absoluteError += diff
            numRating += 1

            f = open("output.txt", 'a')
            f.write("{} {} \n".format(user, predictRating))

        print("RMSE : {0}".format(math.sqrt(sumSquareError / numRating)))
        print("MAE  : {0}".format(absoluteError / numRating))
