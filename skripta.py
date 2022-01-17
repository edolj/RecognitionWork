import shutil,os

f = open("./ids.csv", "r")
lines = f.readlines()

trainSet = "/trainSet/"
testSet = "/testSet/"

parentPath = os.getcwd() + testSet
#print(parentPath)

count = 0
for line in lines:
    count = count + 1
    file, id = line.split(",")
    #print(type(int(id)))
    
    shutil.copy(file, parentPath+str(int(id)))
    
    if count == 250:
       break

f.close()
