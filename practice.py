# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 16:32:44 2021

@author: user
"""
### CHAP = I) NUMBER AND VARIABLES


#1) generate 10 random integers that are divisible by 8
import random as r
import math
dir(math)
    
for i in range(10):
    print(r.randrange(8,100,8))


# II) generate 10 random floating numbers with decimal restricted to 3

for i in range(10):
    print(round(r.uniform(1,10),3))


# III) you are working in a telecom company and are assigned to generate 100 random 10-digit
#      telephone numbers between 8000000000 to 8999999999

for i in range(100):
    print(r.randrange(8000000000,8999999999))

# IV) declare any constant variable and use it to calculate something
PI=3.14
R=3
area=PI*R*R
print('area of circle',area)



# V) take any negative number. do all the actions in 1 line of code: absolute,square,divide by PI

nu_1=-10
pi=3.14
print(abs(nu_1),pow(nu_1,2),math.pi(nu_1))


import tensorflow  as tf
from tensorflow import keras


dir(tf)








