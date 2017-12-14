import CSR
import numpy as np
import math
import random
from collections import defaultdict


class ModelBasedRecommender:

    def __init__(self, k, beta):
        self.trainCSR = CSR.CSR()
        self.trainTransposeCSR = CSR.CSR()
        self.testCSR =  CSR.CSR()
        self.k = int(k)
        self.beta = float(beta)

    def cleanData(self, trainFile, user_list):

        # For User in data set
        # Index dict[songID] = list of rows
        # For each song ID, sort list
        # Randomly select row from list
        # Add to new data set

        data_list = []
        row = 0
        x, y = trainFile.shape
        # Each user is represented once in user_list
        for index, user in enumerate(user_list):
            column = 0
            num_nz = 0
            song_dict = defaultdict(list)
            row_list = []
            while row < x and user == trainFile[row][0]:
                song_dict[trainFile[row][1]].append(row)
                row += 1

            for key in song_dict:
                row_list.append(random.choice(song_dict[key]))

            for val in row_list:
                data_list.append(trainFile[val])

        cleaned_arr = np.array(data_list)

        return cleaned_arr

    #initialize the factor matricies U,V with 1/k 
    def initialize(self, trainFile, testFile, user_list):

        cleanedTrainFile = self.cleanData(trainFile, user_list)

        # By nature of the CSR building methods - there are some questions here.
        # 3 is a placeholder
        self.trainCSR.build_from_numpy(trainFile, user_list, 3)
        self.trainTransposeCSR = self.trainCSR.transpose(False)
        self.testCSR.build_from_numpy(testFile, user_list, 3)

        self.U = np.empty([self.trainCSR.rows, self.k])
        self.U.fill(1.0/self.k)
        self.V = np.empty([self.trainTransposeCSR.rows, self.k])
        self.V.fill(1.0/self.k)

    def factorMatrix(self):
        #Repeat ALS        
        for i in range(0,2):
            self.updateU()
            self.updateV()
            
    def updateU(self):
        numUsers = self.trainCSR.rows
        for user in range(1, numUsers + 1 ):
            startRow = self.trainCSR.row_ptr[user - 1] - 1
            endRow = self.trainCSR.row_ptr[user] - 1
            
            cumulSum = [0] * self.k
            #for all items that user has rated
            for i in range(startRow, endRow):
                rowV = self.trainCSR.col_ind[i]
                rating = self.trainCSR.val[i]
                for k in range(0, self.k):
                    cumulSum[k] = cumulSum[k] + (rating * self.V[rowV][k])

            #compute sum(viT * vi)  this should be a kXk matrix
            cumulMat = np.zeros((self.k,self.k))
            for i in range(startRow, endRow):
                rowV = self.trainCSR.col_ind[i]
                vi = self.V[rowV]
                vi = np.asmatrix(vi)
                viT = np.transpose(vi)
                viTvi = np.dot(viT,vi)
                cumulMat = np.add(cumulMat, viTvi)
                
            #add beta
            betaMatrix = np.zeros((self.k,self.k))
            for i in range(0,self.k):
                betaMatrix[i][i] = self.beta

            cumulMat = np.add(cumulMat, betaMatrix)

            #inverse the k x k matrix
            cumulMat = np.linalg.inv(cumulMat)

            #compute the new Ui
            self.U[user - 1] = np.dot(cumulSum, cumulMat)


    def updateV(self):
        numItems = self.trainTransposeCSR.rows
        for item in range(1, numItems + 1):
            startRow = self.trainTransposeCSR.row_ptr[item - 1] -1
            endRow = self.trainTransposeCSR.row_ptr[item] - 1

            cumulSum = [0] * self.k
            #for all users that have rated the item
            for i in range(startRow, endRow):
                rowU = self.trainTransposeCSR.col_ind[i]
                rating = self.trainTransposeCSR.val[i]
                for k in range(0,self.k):
                    cumulSum[k] = cumulSum[k] + (rating * self.U[rowU][k])

            #compute sum(uiT * ui) this should be a k x k matrix
            cumulMat = np.zeros((self.k, self.k))
            for i in range(startRow, endRow):
                rowV = self.trainTransposeCSR.col_ind[i]
                ui = self.U[rowU]
                ui = np.asmatrix(ui)
                uiT = np.transpose(ui)
                uiTui = np.dot(uiT, ui)
                cumulMat = np.add(cumulMat, uiTui)

            #add beta
            betaMatrix = np.zeros((self.k, self.k))
            for i in range(0, self.k):
                betaMatrix[i][i] = self.beta

            cumulMat = np.add(cumulMat, betaMatrix)

            #inverse the k x k matrix
            cumulMat = np.linalg.inv(cumulMat)

            #comput the new Vi
            self.V[item - 1] = np.dot(cumulSum, cumulMat)

    #Based on the factored U,V predict the rating
    def predictRating(self, user, item):
        rating = np.dot(self.U[user], self.V[item])
        return rating

    def compareAndOutput(self, fileName):
        sumSquareError = 0
        AE = 0

        with open(fileName, 'w') as outputFile:
            #write the fist line of row col nnz
            outputFile.write('{0} {1} {2}'.format(self.testCSR.rows, self.testCSR.columns, self.testCSR.nnz))

            #for each user 
            for user in range(0,self.testCSR.rows):
                outputFile.write('\n')

                #for each item that user has rated predict rating based on model
                for col in range(self.testCSR.row_ptr[user] - 1, self.testCSR.row_ptr[user + 1] - 1):
                    item = self.testCSR.col_ind[col] 
                    rating = self.testCSR.val[col]
                    predictedRating = int(round(self.predictRating(user, item)))

                    #since rating is only from 1 to 5, anything larger/small will be set to the extremes
                    if(predictedRating > 5):
                        predictedRating = 5
                    elif(predictedRating < 1):
                        predictedRating = 1

                    #output the new rating and output
                    outputFile.write('{0} {1} '.format(item + 1, predictedRating))

                    # compute the error
                    diff = abs(predictedRating - rating)

                    AE += diff
                    sumSquareError += diff**2
            

        print("RMSE {0}".format(math.sqrt(float(sumSquareError)/self.testCSR.nnz)))
        print("MAE  {0}".format(float(AE)/self.testCSR.nnz))

    def frobeniusNorm(self, matrix):
        norm = 0
        #iterate through all item accumlating square of value
        for row in range(len(matrix)):
            for col in range(len(matrix[row])):
                norm += matrix[row][col] ** 2
        return norm

    def objectiveFunction(self):
        uNorm = self.frobeniusNorm(self.U)
        vNorm = self.frobeniusNorm(self.V)

        error = 0
        #iterate through all items in training data set and compare to predictedValue
        for row in range(0, self.trainCSR.rows):
            for col in range(self.trainCSR.row_ptr[row], self.trainCSR.row_ptr[row + 1]):
                item = self.trainCSR.col_ind[col] -1
                rating = self.trainCSR.context[col]
                predictedRating = int(round(self.predictRating(row, item)))
                error += (rating - predictedRating)**2

        return error + (self.beta * (uNorm + vNorm))
        
            
        
