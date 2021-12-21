# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 16:40:15 2021

@author: ycxia
"""
import tensorflow as tf
import numpy as np
from GCNLayer import Layer, uniform, dot

      
class ChannelAttention(Layer):
    """Channel attention layer."""
    def __init__(self,  stride=1,normalization=False, padding='SAME', \
                 act=tf.nn.sigmoid,**kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.act=act
        self.normalization = normalization
        self.stride=stride
        self.padding=padding
        

    
    def _call(self, inputs):
        x=tf.expand_dims(inputs,1) 
        self.filters=x.shape[2]
        att=tf.layers.conv1d(x, filters=self.filters, kernel_size=3, \
                                strides=self.stride, padding=self.padding, activation=self.act)
        att=tf.squeeze(att,1)
        output=inputs*(1+att)
        output = tf.nn.relu(output)
        if self.normalization:
            output=self.Batch_Normalization(output)

        return self.act(output)
    
    def Batch_Normalization(self,features):
        s_f=features
        means=tf.reduce_mean(s_f,axis=0,keep_dims=True)
        sigmas2=tf.reduce_mean(tf.pow(s_f-means,2),axis=0,keep_dims=True)
        output=(s_f-means)/(tf.sqrt(sigmas2)+1e-7)
        return output