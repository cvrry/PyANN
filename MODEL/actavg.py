def batchaverage(BAn, An, B):

    i = 0
    for x in An:
        j = 0    
        for y in x:
            temp = 0
            for b in range(B):
                temp = temp + BAn[b][i][j]/B
            
            
            An[i][j] = temp

            j = j+1
        
        i = i+1

    return An

def batcherr(BErr, Err, B):
    j = 0
    for y in Err:
        temp = 0
        for b in range(B):
            temp = temp + BErr[b][j]/B
        
        Err[j] = temp
        j = j + 1

    return Err
