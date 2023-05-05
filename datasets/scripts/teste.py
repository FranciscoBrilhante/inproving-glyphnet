import math
import numpy as np
from scipy.stats import kstest,levene,ttest_rel, wilcoxon
import os

if __name__=="__main__":
    """
    with open('/home/francisco/logs/1/data_dispersion.txt','r') as f:
        lines=f.readlines()
        lines=[line.strip().split() for line in lines]
        a=[float(line[0]) for line in lines]
        b=[float(line[1]) for line in lines]
        
    #test for  homogeneity of variance
    print(levene(a,b))
          
    #perform Kolmogorov-Smirnov test for normality of a
    print(kstest(a, 'norm'))
    #perform Kolmogorov-Smirnov test for normality of a
    print(kstest(b, 'norm'))

    #t-test
    print(wilcoxon(a,b,alternative='greater'))
    """
