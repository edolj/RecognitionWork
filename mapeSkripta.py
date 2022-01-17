import os

trainSet = "/trainSet/"
testSet = "/testSet/"

parentPath = os.getcwd() + trainSet
#print(parentPath)

for number in range(100):
    os.mkdir(parentPath + str(number+1))

parentPath = os.getcwd() + testSet
#print(parentPath)

for number in range(100):
    os.mkdir(parentPath + str(number+1))