#-*- coding: utf-8 -*-


from numpy import *
import operator
from os import listdir
def createDataSet():
    group= array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels
def img2vector(filename):
    returnVect=zeros((1,1024))
    fr=open(filename)
    for i in range(32):
        lineStr=fr.readline()
        for j in range(32):
            returnVect[0,32*i+j]=int(lineStr[j])
    return returnVect
def classify0(inX,dataSet,labels,k):
    #shape[0],是指数组的一维的长度
    dataSetSize=dataSet.shape[0]
    #得到的是dataSetSize行数的inX，列数为1.（这个列数是指inx整体为一列）
    diffMat=tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat=diffMat**2#平方
    sqDistances=sqDiffMat.sum(axis=1)#axis=1是指列数，然后sum是相加
    distances=sqDistances**0.5#开方
    sortedDisIndicies=distances.argsort()#argsort是排序，将元素按照由小到大的顺序返回下标，比如([3,1,2]),它返回的就是([1,2,0])
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDisIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]
def handwritingClassTest():
    hwLabels=[]
    trainingFileList=listdir('C:\\Users\\Administrator\\PycharmProjects\\xuexi\\digits\\trainingDigits')
    m=len(trainingFileList)
    trainingMat=zeros((m,1024))
    for i in range(m):
        fileNameStr=trainingFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:]=img2vector('C:\\Users\\Administrator\\PycharmProjects\\xuexi\\digits\\trainingDigits\\%s'%fileNameStr)
    testFileList=listdir('digits/testDigits')
    errorCount=0
    mTest=len(testFileList)
    for i in range(mTest):
        fileNameStr=testFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        vectorUnderTest=img2vector('C:\\Users\\Administrator\\PycharmProjects\\xuexi\\digits\\testDigits\\%s'%fileNameStr)
        classifierResult=classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print('the classifier came back with : %d ,the real answer is :%d'%(classifierResult,classNumStr))
        if (classifierResult != classNumStr):errorCount+=1.0
    print("\n识别的错误数字: %d" % errorCount)
    print("\n错误率: %f" % (errorCount/float(mTest)))
handwritingClassTest()
