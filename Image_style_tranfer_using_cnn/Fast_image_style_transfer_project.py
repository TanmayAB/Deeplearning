
# coding: utf-8

# ## Import Statements

# In[1]:


import tensorflow as tf
import functools
import vgg
import time
import os
import numpy as np
import random
import cv2
import scipy
import scipy.misc


# ## Defining Functions

# ### CNN Functions

# In[2]:


def conv_inst_relu(input_layer, num_filters, kernel_size, strides):

    output_layer = tf.layers.conv2d(input_layer, num_filters, kernel_size, strides, padding='SAME')
    output_layer = instance_norm(output_layer)
    output_layer = tf.nn.relu(output_layer)

    return output_layer


# In[3]:


def conv_trans_inst_relu(input_layer, num_filters, kernel_size, strides):

    output_layer = tf.layers.conv2d_transpose(input_layer, num_filters, kernel_size, strides, padding='SAME')
    output_layer = instance_norm(output_layer)
    return tf.nn.relu(output_layer)


# In[4]:


def residual_block(input_layer, kernel_size=3):
    
    output_layer = conv_inst_relu(input_layer, 128, kernel_size, 1)
    
    return input_layer + conv_inst_relu(output_layer, 128, kernel_size, 1)


# ### Best model with Three convolutional layers,  Five residual blocks and Three Convolutional Transformation layeers

# In[5]:


def forward_pass(image):
    
    conv1 = conv_inst_relu(image, 32, 9, (1, 1))
    conv2 = conv_inst_relu(conv1, 64, 3, (2, 2))
    conv3 = conv_inst_relu(conv2, 128, 3, (2, 2))
    
    resi1 = residual_block(conv3, (3, 3))
    resi2 = residual_block(resi1, (3, 3))
    resi3 = residual_block(resi2, (3, 3))
    resi4 = residual_block(resi3, (3, 3))
    resi5 = residual_block(resi4, (3, 3))
    
    convT1 = conv_trans_inst_relu(resi5, 64, 3, (2, 2))
    convT2 = conv_trans_inst_relu(convT1, 32, 3, (2, 2))
    convT3 = conv_trans_inst_relu(convT2, 3, 9, 1)
    
    trans = tf.nn.tanh(convT3) * 150 + 255./2
    
    return trans


# ### Longer network with additional convoluational layer and Conv Transofrmation layer

# In[6]:


# def forward_pass(image):
    
#     conv1 = conv_inst_relu(image, 32, 9, (1, 1))
#     conv2 = conv_inst_relu(conv1, 64, 3, (2, 2))
#     conv3 = conv_inst_relu(conv2, 128, 3, (2, 2))
#     conv4 = conv_inst_relu(conv3, 256, 3, (2, 2))
    
#     resi1 = residual_block(conv4, (3, 3))
#     resi2 = residual_block(resi1, (3, 3))
#     resi3 = residual_block(resi2, (3, 3))
#     resi4 = residual_block(resi3, (3, 3))
#     resi5 = residual_block(resi4, (3, 3))
    
#     convT1 = conv_trans_inst_relu(resi5, 128, 3, (2, 2))
#     convT2 = conv_trans_inst_relu(convT1, 64, 3, (2, 2))
#     convT3 = conv_trans_inst_relu(convT2, 32, 3, (2, 2))
#     convT4 = conv_trans_inst_relu(convT3, 3, 9, 1)
    
#     trans = tf.nn.tanh(convT4) * 150 + 255./2
    
#     return trans


# ### Shorter network with only two convoluational layers, three residual blocks and Conv Transofrmation layers

# In[7]:


# def forward_pass(image):
    
#     conv1 = conv_inst_relu(image, 32, 9, (1, 1))
#     conv2 = conv_inst_relu(conv1, 64, 3, (2, 2))
    
#     resi1 = residual_block(conv2, (3, 3))
#     resi2 = residual_block(resi1, (3, 3))
#     resi3 = residual_block(resi2, (3, 3))
    
#     convT1 = conv_trans_inst_relu(resi3, 32, 3, (2, 2))
#     convT2 = conv_trans_inst_relu(convT1, 3, 9, 1)
    
#     trans = tf.nn.tanh(convT2) * 150 + 255./2
    
#     return trans


# ### Normalization functions

# In[8]:


def instance_norm(input_layer, train=True):
    batch, rows, cols, ch = [i.value for i in input_layer.get_shape()]
    mu, sigma_sq = tf.nn.moments(input_layer, [1,2], keep_dims=True)
    shift = tf.Variable(tf.zeros([ch]))
    scale = tf.Variable(tf.ones([ch]))
    epsilon = 1e-3
    normalized = (input_layer-mu)/(sigma_sq + epsilon)**(.5)
    return scale * normalized + shift


# In[9]:


def batch_norm(input_layer, train=True):
    ouput_layer = tf.layers.batch_normalization(input_layer, training=True)
    return output_layer


# ### Initializer functions

# In[10]:


def placeholder_initializer(style_shape, batch_shape):
   
    style_image = tf.placeholder(tf.float32, shape=style_shape, name='style_image')
    X_content = tf.placeholder(tf.float32, shape=batch_shape, name="X_content")

    return style_image, X_content


# ### Loss function

# In[11]:


def loss():
   
    global STYLE_TARGET

    mod = len(CONTENT_TARGETS) % BATCH_SIZE
    if mod > 0:
        STYLE_TARGET = STYLE_TARGET[:-mod] 

    style_features = {}

    batch_shape = (BATCH_SIZE,256,256,3)
    style_shape = (1,) + STYLE_TARGET.shape
    
    with tf.Graph().as_default(), tf.device('/cpu:0'), tf.Session() as sess:
        style_image, X_content = placeholder_initializer(style_shape, batch_shape)        
        style_image_pre = vgg.preprocess(style_image)
        net = vgg.net(VGG_PATH, style_image_pre)
        style_pre = np.array([STYLE_TARGET])
        for layer in STYLE_LAYERS:
            features = net[layer].eval(feed_dict={style_image:style_pre})
            features = np.reshape(features, (-1, features.shape[3]))
            gram = np.matmul(features.T, features) / features.size
            style_features[layer] = gram

        X_pre = vgg.preprocess(X_content)

        content_features = {}
        content_net = vgg.net(VGG_PATH, X_pre)
        content_features[CONTENT_LAYER] = content_net[CONTENT_LAYER]

        preds = forward_pass(X_content/255.0)
        preds_pre = vgg.preprocess(preds)

        net = vgg.net(VGG_PATH, preds_pre)

        tensor_shape = content_features[CONTENT_LAYER].get_shape()
        mul = 1
        for item in tensor_shape.as_list():
            mul *= item
        content_size = mul * BATCH_SIZE
        content_loss = CONTENT_WEIGHT * (2 * tf.nn.l2_loss(
            net[CONTENT_LAYER] - content_features[CONTENT_LAYER]) / content_size
        )

        style_losses = []
        for style_layer in STYLE_LAYERS:
            layer = net[style_layer]
            bs, height, width, filters = map(lambda i:i.value,layer.get_shape())
            size = height * width * filters
            feats = tf.reshape(layer, (bs, height * width, filters))
            feats_T = tf.transpose(feats, perm=[0,2,1])
            grams = tf.matmul(feats_T, feats) / size
            style_gram = style_features[style_layer]
            style_losses.append(2 * tf.nn.l2_loss(grams - style_gram)/style_gram.size)

        style_loss = STYLE_WEIGHT * functools.reduce(tf.add, style_losses) / BATCH_SIZE
       
        loss = content_loss + style_loss

        train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
        sess.run(tf.global_variables_initializer())

        for epoch in range(EPOCHS):
            num_examples = len(CONTENT_TARGETS)
            counts = 0
            while counts * BATCH_SIZE < num_examples:
                start_time = time.time()
                curr = counts * BATCH_SIZE
                step = curr + BATCH_SIZE
                X_batch = np.zeros(batch_shape, dtype=np.float32)
                for j, img_p in enumerate(CONTENT_TARGETS[curr:step]):
                    
                    STYLE_TARGET = cv2.imread(STYLE_PATH)
                    STYLE_TARGET = cv2.cvtColor(STYLE_TARGET,cv2.COLOR_BGR2RGB)
                    STYLE_TARGET = cv2.resize(STYLE_TARGET,(256,256))
                    X_batch[j] = STYLE_TARGET.astype(np.float32)

                counts += 1
                
                train_step.run(feed_dict={X_content:X_batch})
                end_time = time.time()
                delta_time = end_time - start_time
                                
                is_last = epoch == EPOCHS - 1 and counts * BATCH_SIZE >= num_examples
                
                if is_last:
                    to_get = [style_loss, content_loss, loss, preds]
                    test_feed_dict = {
                       X_content:X_batch
                    }

                    tup = sess.run(to_get, feed_dict = test_feed_dict)
                    _style_loss,_content_loss,_loss,_preds = tup
                    losses = (_style_loss, _content_loss, _loss)
                    
                    saver = tf.train.Saver()
                    res = saver.save(sess, MODEL_SAVE_PATH)
                    print('Model successfully saved at : ', str(MODEL_SAVE_PATH + 'fns.ckpt'))
                    yield(epoch, counts, _preds, losses) 


# ## Declaring Parameters

# In[12]:


STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
CONTENT_LAYER = 'relu4_2'

CONTENT_WEIGHT = 7.5e0
STYLE_WEIGHT = 1e2

LEARNING_RATE = 1e-3
EPOCHS = 2
BATCH_SIZE = 4

MODEL_SAVE_PATH = './saved_model/fns.ckpt'
VGG_PATH = './imagenet-vgg-verydeep-19.mat'
TRAIN_PATH = './train_sample/'
STYLE_PATH = './style/wave.jpg'

TEST_IMAGE_PATH = './test_images/batsman.jpg'
OUTPUT_PATH = './output/'


# In[13]:


files_list = []
for filename in os.listdir(TRAIN_PATH):
    if filename.endswith(".DS_Store"):
        A = 1
    else:
        files_list.append(filename)  

CONTENT_TARGETS = [os.path.join(TRAIN_PATH,x) for x in files_list]


STYLE_TARGET= cv2.imread(STYLE_PATH)
STYLE_TARGET = cv2.cvtColor(STYLE_TARGET,cv2.COLOR_BGR2RGB)


# In[14]:


for epoch, counts, _preds, losses in loss():
    style_loss, content_loss, loss = losses

    print('Epoch : ',epoch)
    print('Iterations: ', counts)
    print('Content Loss : ', content_loss)
    print('Style Loss : ', style_loss)
    print('Overall Loss : ', loss)
    print("************")


# In[15]:


print(TEST_IMAGE_PATH)
in_img = cv2.imread(TEST_IMAGE_PATH)
in_img = cv2.cvtColor(in_img, cv2.COLOR_BGR2RGB)
in_img.shape = (1,) + in_img.shape


# ## Perform Image style transfer using trained model

# In[16]:


with tf.Graph().as_default(), tf.Session() as sess:
    
    img = tf.placeholder(tf.float32, shape=in_img.shape,name='input_image')
    
    trans = forward_pass(img)
    saver = tf.train.Saver()
    saver.restore(sess, str(MODEL_SAVE_PATH))
    
    transformed_image_before_reshape = sess.run(trans, feed_dict={img:in_img}) 
    transformed_image = transformed_image_before_reshape[0]
    cv2.imwrite(OUTPUT_PATH + 'transformed_image.jpg', transformed_image)
    print("Transformed stored at : ", OUTPUT_PATH + "transformed_image.jpg")

