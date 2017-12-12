import sys
import ModelBasedRecommender as MBrecommender
import time

def main():
    trainFile  = sys.argv[1]
    testFile   = sys.argv[2]
    k          = sys.argv[3]
    beta       = sys.argv[4]
    lr         = sys.argv[5]
    outputFile = sys.argv[6]

    recommender = MBrecommender.ModelBasedRecommender(k, beta)
    print("initialize variables")
    recommender.initialize(trainFile,testFile)

    startTime = time.time()
    
    print("Begin Factor")
    recommender.factorMatrix()
    print("End Factor")

    print('Begin Prediction')
    recommender.compareAndOutput(outputFile)
    print('End Prediction')
    
    endTime = time.time()
    runTime = (endTime - startTime) * 1000
    print('Running Time:  {0} ms'.format(runTime))


    
main()
    
