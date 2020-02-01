f = open('train', 'r')

f1 = open('train_2', 'w')

i = 0
for j in f.readlines():
    if i < 100:
        print 'write successfully'
        f1.write(j)
        i +=1
    
