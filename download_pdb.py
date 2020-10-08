import wget
import numpy as np
import os
from urllib.error import HTTPError
import sys

g = open(sys.argv[1])
lines = g.readlines()
g.close()

lis = [line.strip() for line in lines if len(line.strip()) > 0] 

pdb = set(lis)
for pb in d['targets']:
        '''
        if type(i) == float:
            print (i, 'is nan')
            continue
        try:
            print (i.split(',')[0])
            count += 1
            pb = i.split(',')[0]
        except AttributeError:
            print (i[0])
            count += 1
            pb = i[0]
        '''
        #print (pb) 
        if pb not in pdb and len(pb) == 4:
            print (pb)
            pdb.add(pb)
            try :
            	url = 'https://files.rcsb.org/download/'+pb+'.pdb'
            	wget.download(url,pb+'.pdb')
            except HTTPError:
                print ('error for', pb)

#print ('Number of samples :', count)
print (len(pdb))


#print (set(d['targets']) - pdb)


