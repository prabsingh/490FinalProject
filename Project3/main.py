import sys
import ModelBasedRecommender as MBrecommender
import time

def main():
    trainFile  = "train_rating.txt"#sys.argv[1]
    testFile   = "test_rating.txt"#sys.argv[2]
    k          = 50 #sys.argv[3]
    beta       = .2 #sys.argv[4]
    lr         = 1 #sys.argv[5]
    outputFile = "output.txt" #sys.argv[6]

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
    
