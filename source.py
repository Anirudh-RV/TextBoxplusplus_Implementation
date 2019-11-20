# In[0]


# In[1]
import numpy as np
import matplotlib.pyplot as plt
import keras
import time
import os
import pickle
import cv2

from tbpp_model import TBPP512, TBPP512_dense
from tbpp_utils import PriorUtil
from ssd_data import InputGenerator
from utils.model import load_weights
from utils.training import Logger

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

# In[3]
# TextBoxes++ + DenseNet
model = TBPP512_dense(softmax=False)

weights_path = None
freeze = []
batch_size = 6
experiment = 'dsodtbpp512fl_synthtext'

# In[4]
prior_util = PriorUtil(model)

if weights_path is not None:
    load_weights(model, weights_path)

for layer in model.layers:
    layer.trainable = not layer.name in freeze


#optim = keras.optimizers.SGD(lr=1e-3, momentum=0.9, decay=0, nesterov=True)
optim = keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=0.001, decay=0.0)

# weight decay
regularizer = keras.regularizers.l2(5e-4) # None if disabled
#regularizer = None
for l in model.layers:
    if l.__class__.__name__.startswith('Conv'):
        l.kernel_regularizer = regularizer

loss = TBPPFocalLoss(lambda_conf=10000.0, lambda_offsets=1.0)

model.compile(optimizer=optim, loss=loss.compute, metrics=loss.metrics)

img = cv2.imread('test.jpg')
img = cv2.resize(img,(512,512))
img = np.expand_dims(img, axis=0)

ans = model.predict(img)

boxes = decode(ans[0] , img_w , img_h , iou)

print(boxes)
