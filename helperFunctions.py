import numpy as np
from sklearn.utils import shuffle
import math




def filteringEx(NoiseDic,x, actuals, errors, epoch):
    
    print("Starting filiering method")
    indexs = []
  
    std,threeshold ,ex = threshold3(errors, epoch)
       
    if std > 0:  
        #Collecting indexes of noise instances
        for i in range(len(ex)):            
            if ex[i] > threeshold:
                indexs.append(i)
        
  
    # variable to have the number of instances to be eliminated   
    NoisyInstances = list()
    NoOfNoisyInstances = len(indexs)
    CorrectNo = 0 # number of discarded instances as they really noise
    WrongNo = 0   # number of discarded instances as they not noise
    for i in indexs:
        NoisyInstances.append(x[i])
        if (actuals[i,-1]==0):
            WrongNo +=1
        else:
            CorrectNo +=1
            
            
    print('Total Number of Discarded instances %d , %d are correct where %d are not' %((CorrectNo+WrongNo),CorrectNo ,WrongNo))
    print("here 1")
    
  
    allIndd = np.arange(x.shape[0])    #taking all indexes of x   
        
    indd2=[]
    count=0
    for i in range(x.shape[0]):
        if (allIndd[i] not in indexs):
            indd2.insert(count,allIndd[i])
        count+=1
    x2 = x[indd2]
    actuals2 = actuals[indd2]
         
    print("====> ",(x2.shape[0] - x.shape[0])," noisy instances  were eleminated by the filter")
        
    return x2, actuals2, NoiseDic, NoisyInstances


def threshold3(data,epoch):
    print("Starting threshold method")
     #get ewa and mean for every instance
    ewa = []
    for i in range(len(data)):
        v=0  
        b0 = 0.9
        b1 = 0.1
        for j in range(len(data[0])):
            
            v = (b0*v)+ (b1*data[i][j])
        ewa.append(v)    
       
    std = np.std(ewa)
    mean = np.mean(ewa)
    mx = np.max(ewa)
    threshold = (std) + mean         

    #get insances located in the read area 
    countLess = 0
    countHigher = 0
    redArea =[]
    for i in range(len(ewa)):
        if ewa[i]>= threshold:
            redArea.append(ewa[i])
            countHigher = countHigher +1
        else:
            countLess = countLess +1

    #get the mean of read area
    m= np.mean(redArea)
    s = np.std(redArea)
#    
#    
    border = (countLess*2.5)/97.5
    border = round(border)
          
    redArea = np.sort(redArea)
    
    print("")
    print("BorderNo :", border)
    print("EWA No:", len(ewa))
    print("RedArea No", len(redArea))
    print("")
    
    
    if border < len(redArea):      
        
        for i in range(len(redArea)):
            border = border -1
            if border == 0:
                suggestedThreshold = redArea[i]
                break
    else:
        suggestedThreshold = np.max(redArea)
    
    #return mean of read are to be the threshold
    return s,suggestedThreshold, ewa



    
    
    
def collectErrors(totalErrors ,errors):  
        
    if len(totalErrors)==0:        
        for i in range(len(errors)):
            totalErrors.append([])
            totalErrors[i].append(errors[i])       
    else:
       
        for i in range(len(errors)):
            totalErrors[i].append(errors[i])    
    return totalErrors



def generate_batches(trainX, trainY, batch_size):
    trainX, trainY = shuffle(trainX, trainY, random_state=0)
    num_batches = math.ceil(float(trainX.shape[0])/batch_size)

    return np.array_split(trainX, num_batches), np.array_split(trainY, num_batches)


