# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 10:04:21 2021

@author: user
"""
import sys
print(sys.version)

import tensorflow as tf
import numpy as np

# create tensor
# 1-D tensor
np.round(np.round.uniform(5,50,10),1)

# 2-D tensor
np.random.randint(10,75,24).reshape(-1,4)


# n-D tensor
np.random.randint(0,255,36).reshape(-1,3,4)

#-----------------------------------------------------------

# eager execution is enabled, by default
tf.executing_eagerly()

# creating object in tensorflow
# contant, variables, placeholders

# constant
weight = tf.constant(55)
type(weight)
print(weight)

# create a constant with a specific data type
weight = tf.constant(55,tf.int8)
print(weight)




# get the shape of the tensor
weight.get_shape()
height.get_shape()

# Array : convert numpy to tensorflow object
rates = tf.constant(np.random.randint(10,75,24).reshape(-1,4))
type(rates)
print(rates)
rates.get_shape()

# convert numpy to tensorflow object : method 2
rates2 = np.random.randint(5,30,20).reshape(-1,4)
rates2
type(rates2)
rates2 = tf.constant(rates2)
type(rates2)
print(rates2)

# arithmatic operation on the constant
height + 1.5
height - 0.7
height * 1.5
height / 0.4

# Array
rates + 2
rates -5
rates*2
rates/15

# this will give an error
rates*1.4

# in TF 









































