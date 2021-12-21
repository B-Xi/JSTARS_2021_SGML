# -*- coding: utf-8 -*-
import tensorflow as tf
from GCNLayer import *
from CCALayer import ChannelAttention as CCA
import numpy as np

def masked_softmax_cross_entropy(preds, labels, mask):
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= tf.transpose(mask)
    return tf.reduce_mean(tf.transpose(loss))

# with the centers
def distance(features, centers):
    f_2 = tf.reduce_sum(tf.pow(features, 2), axis=1, keep_dims=True)
    c_2 = tf.reduce_sum(tf.pow(centers, 2), axis=1, keep_dims=True)
    dist = f_2 - 2*tf.matmul(features, centers, transpose_b=True) + tf.transpose(c_2, perm=[1,0])
    return dist

def compute_centers(preds,labels,mask):
    centers=tf.Variable(tf.zeros([preds.shape[1],preds.shape[1]]))
    y=tf.argmax(labels,axis=1)
    xlabled=tf.gather(preds,tf.squeeze(tf.where(tf.squeeze(mask,1)),1))
    for i in range(preds.shape[1]):
        xlabled_i=tf.gather(xlabled,tf.squeeze(tf.where(tf.equal(y,i)),1))
        centers[i,:].assign(tf.reduce_mean(xlabled_i,axis=0,keep_dims=True))
    return centers
    
# def masked_softmax_metric_loss(preds, labels, mask):
#     f=tf.nn.softmax(preds)
#     centers=compute_centers(f,labels,mask)
#     c_l=tf.gather(centers,tf.argmax(labels,1))
#     # loss = 1/tf.reduce_sum(tf.pow(f - c_l, 2), axis=1)
#
#     xc_dist=tf.exp(tf.reduce_sum(tf.pow(f-c_l,2),axis=1))
#     xc_dists=tf.exp(distance(f,centers))
#     loss = (tf.log(xc_dist/(tf.reduce_sum(xc_dists, axis=1)-xc_dist)+1))
#
#     mask = tf.cast(mask, dtype=tf.float32)
#     mask /= tf.reduce_mean(mask)
#     loss *= tf.transpose(mask)
#     return tf.reduce_mean(tf.transpose(loss))

def masked_softmax_metric_loss(preds, labels, mask):
    f=tf.nn.softmax(preds)
    centers=compute_centers(f,labels,mask)
    c_l=tf.gather(centers,tf.argmax(labels,1))
    xc_dist = tf.multiply(tf.nn.l2_normalize(f, dim=1), tf.nn.l2_normalize(c_l, dim=1))
    xc_dist =tf.exp(tf.reduce_sum(xc_dist,axis=1))
    xc_dists = tf.reduce_sum(tf.multiply(tf.expand_dims(tf.nn.l2_normalize(f, dim=1),1), tf.nn.l2_normalize(centers, dim=1)),2)

    xc_dists=tf.exp(xc_dists)
    loss = -(tf.log(xc_dist/(tf.reduce_sum(xc_dists, axis=1)-xc_dist)))

    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= tf.transpose(mask)
    return tf.reduce_mean(tf.transpose(loss))

def masked_accuracy(preds, labels, mask):
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= tf.transpose(mask)
    return tf.reduce_mean(tf.transpose(accuracy_all))

class GCNModel3(object):
    def __init__(self, features, labels,l,idx, learning_rate, num_classes, mask, support, scale_num, h):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.classlayers = []
        self.labels = labels
        self.inputs = features
        #self.f=f
        self.lab=l.flatten()
        self.idx=[x.flatten() for x in idx]
        self.scale_num = scale_num
        self.loss = 0
        self.support = support
        self.concat_vec = []
        self.outputs = None
        self.num_classes = num_classes
        self.hidden1 = h 
        self.mask = mask
        self.build()
        
    def _build(self):
        for scale_idx in range(self.scale_num):
            activations = []
            activations.append(self.inputs[scale_idx])
            self.classlayers.append(GraphConvolution(act = tf.nn.relu,
                                      input_dim = np.shape(self.inputs[scale_idx])[1],
                                      output_dim = self.hidden1,
                                      support = self.support[scale_idx],
                                      bias = True,
                                      normalization=True
                                      ))   
            layer = self.classlayers[-1]        
            hidden = layer(activations[-1])
            activations.append(hidden)
            
            #cca
            self.classlayers.append(CCA(normalization=True,act= tf.nn.sigmoid))
            layer = self.classlayers[-1]
            hidden = layer(activations[-1])
            activations.append(hidden)

            self.classlayers.append(GraphConvolution(act = lambda x:x,
                                      input_dim = self.hidden1,
                                      output_dim = self.num_classes,
                                      support = self.support[scale_idx],
                                      bias = True,
                                      normalization=True
                                      ))   
            layer = self.classlayers[-1]
            hidden = layer(activations[-1])
            activations.append(hidden)

            if scale_idx == 0:
                self.outputs = tf.gather(activations[-1],self.idx[scale_idx][(self.idx[scale_idx]>0)*(self.lab>0)]-1)
            else:
                self.outputs += tf.gather(activations[-1],self.idx[scale_idx][(self.idx[scale_idx]>0)*(self.lab>0)]-1)

    def build(self):
        self._build()
        self._loss()
        self._accuracy()
        self.opt_op = self.optimizer.minimize(self.loss)     

    def _loss(self):
        # Cross entropy error
        self.loss += 1*masked_softmax_cross_entropy(self.outputs, self.labels, self.mask)
        self.loss += 0.1*masked_softmax_metric_loss(self.outputs, self.labels, self.mask)

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.labels, self.mask)
    
    def Get01Mat(self, mat1): #set the location (where is not zero) to 1, only considering the correlations among the neighboring examples.
        [r, c] = np.shape(mat1)
        mat_01 = np.zeros([r, c])
        pos1 = np.argwhere(mat1!=0)
        mat_01[pos1[:,0], pos1[:,1]] = 1
        return np.array(mat_01, dtype='float32')
