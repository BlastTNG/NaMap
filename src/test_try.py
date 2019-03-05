import numpy as np 

t = None
print 'Ciao'
#while True:
for i in range(100):
    print i
    try:
        t = np.loadtxt('test')
    #except isinstance(t, np.ndarray) == True:
    except IOError:
        print 'File Error'
        continue

    