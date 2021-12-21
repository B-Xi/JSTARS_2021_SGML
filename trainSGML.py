# -*- coding: utf-8 -*-
import numpy as np
from GCNModel3 import GCNModel3
from BuildSPInst_A import *
import tensorflow as tf
import time
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
iter=0
for iter in range(1):
    time_start=time.time()
    def GCNevaluate(mask1, labels1):
        t_test = time.time()
        outs_val = sess.run([GCNmodel.loss, GCNmodel.accuracy], feed_dict={labels: labels1, mask: mask1})
        return outs_val[0], outs_val[1], (time.time() - t_test)

    data_name = 'IP'
    num_classes = 16

    learning_rate = 0.0005
    epochs=500
    img_gyh = data_name+'_gyh'
    img_gt = data_name+'_gt'

    Data = load_HSI_data(data_name)
    model = GetInst_A(Data['useful_sp_lab'], Data[img_gyh], Data[img_gt], Data['trpos'])
    
    def get_mask(gt, poses, sets):
        pixel_mask_all=np.zeros(gt.shape, dtype='bool')
        
        if sets=='all':
            pixel_mask_all[gt>0]=True
            gt_tr_te=gt[pixel_mask_all]
        else:
            pixel_mask_all[poses[:,0]-1,poses[:,1]-1]=True
            pixel_mask=pixel_mask_all[gt>0]
        return pixel_mask
    
    # create gt_tr_te
    pixel_mask_tr_te=np.zeros(Data[img_gt].shape, dtype='bool')
    pixel_mask_tr_te[Data[img_gt]>0]=True
    gt_tr_te=Data[img_gt][pixel_mask_tr_te]
    # create pixel_mask_tr gt_nonzeros
    pixel_mask_tr_te*=False
    pixel_mask_tr_te[Data['trpos'][:,0]-1,Data['trpos'][:,1]-1]=True
    pixel_mask_tr=pixel_mask_tr_te[Data[img_gt]>0]
    pixel_mask_val=get_mask(Data[img_gt],Data['valpos'],'val')
    pixel_mask_te=~(pixel_mask_tr^pixel_mask_val)
    gt_nonzeros=(Data[img_gt])[Data[img_gt]>0]

    #gt_nonzeros_tr=gt_nonzeros[pixel_mask_tr]
    gt_1hot = np.zeros([pixel_mask_tr.shape[0], num_classes]) # one-hot coding
    for row_idx in range(gt_1hot.shape[0]):
        col_idx = int(gt_nonzeros[row_idx])-1
        gt_1hot[row_idx, col_idx] = 1
    gt_1hot_tr=np.array(gt_1hot)
    gt_1hot_tr[pixel_mask_te^pixel_mask_val]*=False
    gt_1hot_val=np.array(gt_1hot)
    gt_1hot_val[pixel_mask_te^pixel_mask_tr]*=False
    gt_1hot_te=np.array(gt_1hot)
    gt_1hot_te[pixel_mask_tr^pixel_mask_val]*=False
    pixel_mask_tr=np.expand_dims(pixel_mask_tr, axis=1)
    pixel_mask_val=np.expand_dims(pixel_mask_val, axis=1)
    pixel_mask_te=np.expand_dims(pixel_mask_te, axis=1)

    useful_sp_lab = [np.array(x, dtype='int32') for x in model.useful_sp_lab]
    sp_mean = [np.array(x, dtype='float32') for x in model.sp_mean]
    sp_label = [ np.array(x, dtype='float32') for x in model.sp_label]
    sp_support = []

    #sp_support=list(scio.loadmat('data//'+data_name+'//result//support.mat')['sp_support'])
    for A_x in model.sp_A:
        sp_A = np.array(A_x, dtype='float32')
        sp_support.append(np.array(model.CalSupport(sp_A), dtype='float32'))

    ############################################

    mask = tf.placeholder("int32", [None, 1])
    labels = tf.placeholder("float", [None, num_classes])
    # Normalize sp_mean
    #sp_mean /= sp_mean.sum(1).reshape(-1, 1)

    seed=123
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # 构建proposed model
    GCNmodel = GCNModel3(features = sp_mean, labels = labels,l=Data[img_gt],idx=useful_sp_lab, learning_rate = learning_rate,
                        num_classes = num_classes, mask = mask, support = sp_support, scale_num = 3, h = 32)

    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    train_t = time.time()
    for epoch in range(epochs):
        # Training step
        outs = sess.run([GCNmodel.opt_op, GCNmodel.loss, GCNmodel.accuracy], feed_dict={ labels:gt_1hot_tr,
                        mask:pixel_mask_tr })
        outsval = sess.run([GCNmodel.loss, GCNmodel.accuracy], feed_dict={ labels:gt_1hot_val,
                        mask:pixel_mask_val })
        print("Epoch:", '%04d' % (epoch + 1),
              "train_loss=", "{:.2f}".format(outs[1]),
              "train_acc=", "{:.2f}".format(outs[2]),
              "val_loss=", "{:.2f}".format(outsval[0]),
              "val_acc=", "{:.2f}".format(outsval[1]))
        
        # print("Epoch:", '%04d' % (epoch + 1), "val_loss=", "{:.5f}".format(outsval[0]),
        #       "val_acc=", "{:.5f}".format(outsval[1]))
    print("Optimization Finished!")
    training_time = time.time() - train_t
    print("Training time =", str(time.time() - train_t))
    # Testing
    test_cost, test_acc, test_duration = GCNevaluate(pixel_mask_te, gt_1hot_te)
    print("Test set results:", "cost=", "{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
    #######
    pred_map=np.zeros_like(Data[img_gt])

    #superPixel-wise accuracy
    outputs = sess.run(GCNmodel.outputs)
    predictions=np.argmax(outputs, axis=1)+1
    pred_map[Data[img_gt]>0]=predictions
    #######
    resultpath='data//'+data_name+'//result//'
    matrix = np.zeros((num_classes, num_classes))
    outputs_decode = np.argmax(outputs[pixel_mask_te[:,0]], 1)
    test_labels = np.argmax(gt_1hot_te[pixel_mask_te[:,0]],1)
    n = pixel_mask_te.sum()
    with open(resultpath+'prediction.txt', 'w') as f:
        for i in range(n):
            pre_label = int(outputs_decode[i])
            f.write(str(pre_label)+'\n')
            matrix[pre_label][test_labels[i]] += 1
    np.savetxt(resultpath+'result_matrix.txt', matrix, fmt='%d', delimiter=',')
    print(''+str(np.int_(matrix)))
    print(np.sum(np.trace(matrix)))
    # print('OA = '+str(OA)+'\n')
    ua = np.diag(matrix)/np.sum(matrix, axis=0)
    AA=np.sum(ua)/matrix.shape[0]

    precision = np.diag(matrix)/np.sum(matrix, axis=1)
    matrix = np.mat(matrix)
    OA = np.sum(np.trace(matrix)) / float(n)

    Po = OA
    xsum = np.sum(matrix, axis=1)
    ysum = np.sum(matrix, axis=0)
    Pe = float(ysum*xsum)/(np.sum(matrix)**2)
    Kappa = float((Po-Pe)/(1-Pe))

    AP=np.sum(precision)/matrix.shape[0]

    # print('ua =')
    for i in range(num_classes):
        print(ua[i])
    print(AA)
    print(OA)
    print(Kappa)
    print()
    for i in range(num_classes):
        print(precision[i])
    print(AP)
    f.close()
    #################################################

    scio.savemat('data/'+data_name+'/result/pred.mat',{'pred':pred_map})
    scio.savemat('data/'+data_name+'/result/support.mat',{'sp_support':sp_support})
    iter=iter+1

