import numpy as np
import scipy.io as scio  

def Con2Numpy(filepath,var_name):
    data = scio.loadmat(filepath)  
    x = data[var_name]
    x1 = x.astype(float)
    return x1

def map2pos(map):
    pos=np.argwhere(map>0)+1
    return pos

def normalize_spectral(X):
    X=X.astype('float32')
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i,j,:] = (X[i,j,:] - X[i,j,:].min()) / (X[i,j,:].max() - X[i,j,:].min())
    return X

def normalize_spatial(X):
    X=X.astype('float32')
    for i in range(X.shape[2]):
            X[:,:,i] = (X[:,:,i] - X[:,:,i].min()) / (X[:,:,i].max() - X[:,:,i].min())
    return X

def get_idx_train_val(gt, poses, rate):
    # 随机10%
    permutation = np.random.permutation(poses.shape[0])
    idx_val= permutation[0:round(poses.shape[0]*rate)]
    mask = np.ones((poses.shape[0]), dtype=np.bool)
    mask[idx_val] = False
    return poses[mask], poses[idx_val]


def get_idx_train_val1(gt, poses, rate):
    val_idx = list()
    poses=poses-1
    gt_tr = gt[poses[:,0],poses[:,1]]
    num_classes=max(gt_tr)
    mask = np.ones((poses.shape[0]), dtype=np.bool)
    for i in range(1,num_classes+1):
        idx_c = np.where(gt_tr == i)[0]
        num_c = idx_c.shape[0]
        idx=np.arange(num_c)
        np.random.shuffle(idx)
        val_idx.append(idx_c[idx[:round(num_c * rate)]])#各类随机10%
        #val_idx.append(idx_c[-round(num_c * rate):])#%各类后10%
    idx_val = np.concatenate(val_idx)
    mask[idx_val] = False
    return poses[mask]+1, poses[idx_val]+1

def load_HSI_data(data_name):
    Data = dict()
    path = './/data//'+data_name+'//'
    img_gyh = data_name+'_gyh'
    img_gt = data_name+'_gt'
    Data['useful_sp_lab'] = scio.loadmat('data//'+'IP'+'//labels_superpixel.mat')['labels_superpixel']
    Data[img_gt] = np.array(Con2Numpy(path+'Gt','groundtruth'), dtype='int')
    Data[img_gyh] = normalize_spatial(Con2Numpy(path+'data','spectral_data'))
    Data['trpos'] = np.array(map2pos(Con2Numpy(path+'trainingMap','trainingMap')),dtype='int')
    np.random.shuffle(Data['trpos'])
    Data['trpos'],Data['valpos']=get_idx_train_val(Data[img_gt], Data['trpos'], rate=0.1)
    return Data

if __name__ == '__main__':
    load_HSI_data('IP')