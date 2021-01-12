
import tensorflow as tf
import h5py 
# You'll generate plots of attention in order to see which parts of an image
# our model focuses on during captioning
import matplotlib.pyplot as plt

# Scikit-learn includes many helpful utilities
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from PIL import Image
import collections
import random
import re
import numpy as np
import os
import time
import json
from glob import glob
from PIL import Image
import pickle
from nltk.translate import bleu 

import cv2 as cv


#%%

# Some machine specific configurations.

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

#%%

proj_root_dir = ""

trainim_dir = proj_root_dir + "dataset/images/train/"
testim_dir = proj_root_dir + "dataset/images/test/"

word_embed_dir = proj_root_dir + "dataset/word_embeddings/"

eee443_dataset_dir = proj_root_dir + "dataset/eee443_dataset/"

model_storage_dir = proj_root_dir + "models/"

#%%
class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, features, hidden):
    # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

    # hidden shape == (batch_size, hidden_size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
    hidden_with_time_axis = tf.expand_dims(hidden, 1)

    # attention_hidden_layer shape == (batch_size, 64, units)
    attention_hidden_layer = (tf.nn.tanh(self.W1(features) +
                                         self.W2(hidden_with_time_axis)))

    # score shape == (batch_size, 64, 1)
    # This gives you an unnormalized score for each image feature.
    score = self.V(attention_hidden_layer)

    # attention_weights shape == (batch_size, 64, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * features
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x

class RNN_Decoder(tf.keras.Model):
#   def __init__(self, embedding_dim, units, vocab_size, embedding_matrix):
  def __init__(self, embedding_dim, units, vocab_size):
    super(RNN_Decoder, self).__init__()
    self.units = units

#     self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
# trainable=False)
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

    # self.lstm = tf.keras.layers.LSTM(self.units,
    #                                return_sequences=True,
    #                                return_state=True,
    #                                recurrent_initializer='glorot_uniform')

    self.gru = tf.keras.layers.GRU(self.units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    
    self.fc1 = tf.keras.layers.Dense(self.units)
    self.fc2 = tf.keras.layers.Dense(vocab_size)

    self.attention = BahdanauAttention(self.units)

  def call(self, x, features, hidden):
    # defining attention as a separate model
    context_vector, attention_weights = self.attention(features, hidden)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    # output, state, carry_state = self.lstm(x)
    output, state = self.gru(x)

    # shape == (batch_size, max_length, hidden_size)
    x = self.fc1(output)

    # x shape == (batch_size * max_length, hidden_size)
    x = tf.reshape(x, (-1, x.shape[2]))

    # output shape == (batch_size * max_length, vocab)
    x = self.fc2(x)

    return x, state, attention_weights

  def reset_state(self, batch_size):
    return tf.zeros((batch_size, self.units))

#%%
train_image_paths2 = []
for a in range(82783):
    # loc = 'E:\\neural\\finIm/im' + str(a) + ".png"
    loc = trainim_dir + 'im' + str(a) + ".png"
    if (os.path.exists(loc)):
        train_image_paths2.append(loc)
        
test_image_paths = []
for a in range(40504):
    loc = testim_dir + 'im' + str(a) + ".png"
    # loc = 'C:\\Users\\BARAN/Desktop/Dersler/EEE/EEE443/phtyon/finIm/im' + str(a) + ".png"
    if (os.path.exists(loc)):
        test_image_paths.append(loc)   
        
# tot=0
# for a in range(40504):
#     loc = 'testIm/im' + str(a) + ".png"
#     #loc = 'C:\\Users\\BARAN/Desktop/Dersler/EEE/EEE443/phtyon/finIm/im' + str(a) + ".png.npy"
#     if (os.path.exists(loc)):
#         tot +=1
#%%
f1 = h5py.File(eee443_dataset_dir + 'eee443_project_dataset_train.h5', "r")
print("Keys: %s" % f1.keys())
train_cap = np.array(f1["train_cap"])
train_imid = np.array(f1["train_imid"]) - 1
train_ims = np.array(f1["train_ims"])
train_url = np.array(f1["train_url"]) # URL of the image
word_code = np.array(f1["word_code"])

f2 = h5py.File(eee443_dataset_dir + 'eee443_project_dataset_test.h5', "r")
print("Keys: %s" % f2.keys())
test_cap = np.array(f2["test_caps"])
test_imid = np.array(f2["test_imid"])
test_ims = np.array(f2["test_ims"])
test_url = np.array(f2["test_url"]) # URL of the image

#%%
@tf.autograph.experimental.do_not_convert
def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    # img = tf.image.resize(img, (224, 224))
    img = tf.image.resize(img, (299, 299))
    # img = tf.keras.applications.resnet_v2.preprocess_input(img)
    img = tf.keras.applications.inception_resnet_v2.preprocess_input(img)
    return img, image_path
 
# image_model2 = tf.keras.applications.ResNet50V2(include_top=False,weights='imagenet')
image_model2 = tf.keras.applications.InceptionResNetV2(include_top=False,weights='imagenet')
new_input2 = image_model2.input
hidden_layer2 = image_model2.layers[-1].output 
image_features_extract_model2 = tf.keras.Model(new_input2, hidden_layer2)
#%%
# Get unique images
encode_train2 = train_image_paths2

# Feel free to change batch_size according to your system configuration
image_dataset2 = tf.data.Dataset.from_tensor_slices(encode_train2)
# image_dataset2 = image_dataset2.map(
#   load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(64)
image_dataset2 = image_dataset2.map(
  load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(32)
#%%
i = 0
for img, path in image_dataset2:
    i +=1
    if i%100 == 0:
        print(i)
    batch_features2 = image_features_extract_model2(img)
    batch_features2 = tf.reshape(batch_features2,
                              (batch_features2.shape[0], -1, batch_features2.shape[3]))
    for bf, p in zip(batch_features2, path):
      path_of_feature = p.numpy().decode("utf-8")
      np.save(path_of_feature, bf.numpy())
#%%
# Get unique images
encode_test2 = test_image_paths

# Feel free to change batch_size according to your system configuration
image_dataset3 = tf.data.Dataset.from_tensor_slices(encode_test2)
# image_dataset3 = image_dataset3.map(
#   load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(64)
image_dataset3 = image_dataset3.map(
  load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(32)
#%%
i = 0
for img, path in image_dataset3:
    i +=1
    if i%100 == 0:
        print(i)
    batch_features3 = image_features_extract_model2(img)
    batch_features3 = tf.reshape(batch_features3,
                              (batch_features3.shape[0], -1, batch_features3.shape[3]))
    for bf, p in zip(batch_features3, path):
      path_of_feature = p.numpy().decode("utf-8")
      np.save(path_of_feature, bf.numpy())
          
#%%           
img_to_cap_vector2 = collections.defaultdict(list) 
for i in range(len(train_imid)):
    cap = train_cap[i]
    imid = train_imid[i]
    key = trainim_dir + 'im' + str(imid) + ".png"
    if(os.path.exists(key)):
        img_to_cap_vector2[key].append(cap)        

img_to_cap_vector3 = collections.defaultdict(list) 
for i in range(len(test_imid)):
    cap = test_cap[i]
    imid = test_imid[i]
    key = testim_dir + 'im' + str(imid) + ".png"
    if(os.path.exists(key)):
        img_to_cap_vector3[key].append(cap)  
#%%
# Create training and validation sets using an 85-15 split randomly.
img_keys2 = list(img_to_cap_vector2.keys())
random.shuffle(img_keys2)
slice_index2 = int(len(img_keys2)*0.85)
img_name_train_keys2, img_name_val_keys2 = img_keys2[:slice_index2], img_keys2[slice_index2:]

img_name_train2 = []
cap_train2 = []
for imgt in img_name_train_keys2:
  capt_len = len(img_to_cap_vector2[imgt])
  img_name_train2.extend([imgt] * capt_len)
  cap_train2.extend(img_to_cap_vector2[imgt])

img_name_val2 = []
cap_val2 = []
for imgv in img_name_val_keys2:
  capv_len = len(img_to_cap_vector2[imgv])
  img_name_val2.extend([imgv] * capv_len)
  cap_val2.extend(img_to_cap_vector2[imgv])
  
img_keys3 = list(img_to_cap_vector3.keys())
img_name_test2 = []
cap_test2 = []
for imgv in img_keys3:
  capv_len = len(img_to_cap_vector3[imgv])
  img_name_test2.extend([imgv] * capv_len)
  cap_test2.extend(img_to_cap_vector3[imgv])

#%%
word_ind = np.asarray(np.asarray(word_code.tolist()))
words = np.asarray(np.asarray(word_code.dtype.names)) 
# np.squeeze reduces the dimension by 1. These conversions should be made for sorting 
word_ind = np.squeeze(word_ind.astype(int)) 
words = np.squeeze(np.reshape(words, (1, 1004))) 
# arg sort returns the indices to make the sorting 
sort_indices = np.argsort(word_ind) # use the argsort to sort both words and word_indices 
words = np.array(words)[sort_indices] 
word_ind = np.array(word_ind)[sort_indices]



#%%
# Feel free to change these parameters according to your system's configuration

# BATCH_SIZE = 32
# BATCH_SIZE = 256
BATCH_SIZE = 128
BUFFER_SIZE = 1000
embedding_dim = 300
units = 256
vocab_size = len(words)
num_steps = len(img_name_train2) // BATCH_SIZE
val_num_steps = len(img_name_val2) // BATCH_SIZE
# Shape of the vector extracted from InceptionV3 is (64, 2048)
# These two variables represent that vector shape
features_shape = 2048
attention_features_shape = 64 
# attention_features_shape = 49
#%%
# Load the numpy files
def map_func(img_name, cap):
  img_tensor = np.load(img_name.decode('utf-8')+'.npy')
  return img_tensor, cap 

dataset = tf.data.Dataset.from_tensor_slices((img_name_train2, cap_train2))

# Use map to load the numpy files in parallel
dataset = dataset.map(lambda item1, item2: tf.numpy_function(
          map_func, [item1, item2], [tf.float32, tf.int32]),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Shuffle and batch
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE) 

val_dataset = tf.data.Dataset.from_tensor_slices((img_name_val2, cap_val2))

# Use map to load the numpy files in parallel
val_dataset = val_dataset.map(lambda item1, item2: tf.numpy_function(
          map_func, [item1, item2], [tf.float32, tf.int32]),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Shuffle and batch
val_dataset = val_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
val_dataset = val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE) 
# #%%
# embeddings_index = dict()
# f = open(word_embed_dir + 'glove.6B.300d.txt', encoding="utf8")

# for line in f:
#     values = line.split()
#     word = values[0]
#     coefs = np.asarray(values[1:], dtype='float32')
#     embeddings_index[word] = coefs

# f.close()
# #%%
# print('Loaded %s word vectors.' % len(embeddings_index))

# # create a weight matrix for words in training docs
# embedding_matrix = np.zeros((1004, 300))
# unks = np.array([3, 55, 80, 492, 561, 621])
# i = 0
# for word in words:
#     embedding_vector = embeddings_index.get(word)
    
#     a = np.zeros(300)
#     if i == 0:
#         a[100] = 1
#         embedding_matrix[i] = a
#     elif i == 1:
#         a[0] = 1
#         embedding_matrix[i] = a
#     elif i == 2:
#         a[-1] = 1
#         embedding_matrix[i] = a
#     elif any(unks == i):
#         a[200] = 1
#         embedding_matrix[i] = a
#     else:
#         embedding_matrix[i] = embedding_vector
#     i +=1
#%%
encoder = CNN_Encoder(embedding_dim)
# decoder = RNN_Decoder(embedding_dim, units, vocab_size, embedding_matrix)
decoder = RNN_Decoder(embedding_dim, units, vocab_size)
optimizer = tf.keras.optimizers.Adam()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_) 

#%%
t_loss_plot = []
v_loss_plot = []
#%% 
# @tf.function
def train_step(img_tensor, target):
  loss = 0

  # initializing the hidden state for each batch
  # because the captions are not related from image to image
  hidden = decoder.reset_state(batch_size=target.shape[0])

  dec_input = tf.expand_dims([np.where(words == 'x_START_')[0][0]] * target.shape[0], 1)

  with tf.GradientTape() as tape:
      features = encoder(img_tensor)
      for i in range(1, target.shape[1]):
          # passing the features through the decoder
          predictions, hidden, _ = decoder(dec_input, features, hidden)
          loss += loss_function(target[:, i], predictions)

          # using teacher forcing
          dec_input = tf.expand_dims(target[:, i], 1)

  total_loss = (loss / int(target.shape[1]))

  trainable_variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, trainable_variables)

  optimizer.apply_gradients(zip(gradients, trainable_variables))

  return loss, total_loss

def val_step(img_tensor, target):
  loss = 0
  hidden = decoder.reset_state(batch_size=target.shape[0])
  dec_input = tf.expand_dims([np.where(words == 'x_START_')[0][0]] * target.shape[0], 1)
  with tf.GradientTape() as tape:
      features = encoder(img_tensor)
      for i in range(1, target.shape[1]):
          # passing the features through the decoder
          predictions, hidden, _ = decoder(dec_input, features, hidden)
          loss += loss_function(target[:, i], predictions)

          dec_input = tf.expand_dims(target[:, i], 1)

  total_loss = (loss / int(target.shape[1]))
  return loss, total_loss

#%% Train 
# EPOCHS = 1
EPOCHS = 20
toSave = proj_root_dir
ep_st = time.time()
for epoch in range(EPOCHS):
    start = time.time()
    total_loss = 0
    val_loss = 0
    for (batch, (img_tensor, target)) in enumerate(dataset):
        batch_loss, t_loss = train_step(img_tensor, target)
        total_loss += t_loss
        
        if batch % 100 == 0:
            print ('Epoch {} Batch {} Train Loss {:.4f}'.format(
              epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))
        
    for (batch, (img_tensor, target)) in enumerate(val_dataset):
        batch_loss, t_loss = val_step(img_tensor, target)
        val_loss += t_loss
        
        if batch % 100 == 0:
            print ('Epoch {} Batch {} Val Loss {:.4f}'.format(
              epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))

    # storing the epoch end loss value to plot later
    t_loss_plot.append(total_loss / num_steps)
    v_loss_plot.append(val_loss / val_num_steps)
    print ('Epoch {} Train Loss {:.6f}'.format(epoch + 1,
                                         total_loss/num_steps))
    print ('Epoch {} Val Loss {:.6f}'.format(epoch + 1,
                                         val_loss/val_num_steps))
    print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
    print ('Total time upto now {} min\n'.format((time.time() - ep_st)/60)) 
    # Save the weights after epoches
    decoder.save_weights(toSave + "/ep" + str(epoch) + "/decoder")
    encoder.save_weights(toSave + "/ep" + str(epoch) + "/encoder")
    
#%%
decoder.save_weights(model_storage_dir + 'decoder_inresnet_gru_custom')
encoder.save_weights(model_storage_dir + 'encoder_inresnet_gru_custom')

# decoder.load_weights(model_storage_dir + 'incresnetv2_gru/' + 'decoder_inresnet_gru_custom')
# encoder.load_weights(model_storage_dir + 'incresnetv2_gru/' + 'encoder_inresnet_gru_custom')
# #%%
# decoder.load_weights('D:/neural final imgs/decoder_lstmv3')
# encoder.load_weights('D:/neural final imgs/encoder_lstmv3')

# #%%
# plt.plot(t_loss_plot)
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title(' Train Loss Plot')
# plt.savefig(model_storage_dir + 'train_loss.png')
# # plt.show()
# plt.clf()
# plt.plot(v_loss_plot)
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Validation Loss Plot')
# plt.savefig(model_storage_dir + 'test_loss.png')
# # plt.show()
#%%
def evaluate(image,max_length):
    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model2(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([np.where(words == 'x_START_')[0][0]], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        

        if words[predicted_id] == 'x_END_':
            return result, attention_plot
        
        result.append(words[predicted_id])
        
        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot

def plot_attention(image, result, attention_plot):
    temp_image = np.array(Image.open(image))

    fig = plt.figure(figsize=(10, 10))

    len_result = len(result)
    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))
        ax = fig.add_subplot(len_result//2, len_result//2, l+1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.show()


rid = np.random.randint(0, len(img_name_val2))
image = img_name_val2[rid]
plt.imshow(Image.open(image))
real_caption = ' '.join([words[i] for i in cap_val2[rid] if i not in [0]])
result, attention_plot = evaluate(image,17)

print ('Real Caption:', real_caption)
print ('Prediction Caption:',' '.join(result))
# plot_attention(image, result, attention_plot)

encoder.summary()
decoder.summary()

#%%

rid = np.random.randint(0, len(img_name_val2))
image = img_name_val2[rid]
plt.imshow(Image.open(image))
real_caption = ' '.join([words[i] for i in cap_val2[rid] if i not in [0]])
result, attention_plot = evaluate(image,17)

print ('Real Caption:', real_caption)
print ('Prediction Caption:',' '.join(result))
# plot_attention(image, result, attention_plot)

def show_captioned2(img_path, str_caption_predicted, str_caption_real):
    print('Real Caption:', str_caption_real)
    print('Prediction Caption:', str_caption_predicted)
    image = cv.imread(img_path)
    height, width = image.shape[:2]
    window_name = 'image'
    cv.putText(image, 'Real Caption:' + str_caption_real, (2, round(height - 40)), cv.FONT_HERSHEY_COMPLEX, 0.4, (100, 255, 255), 1)
    cv.putText(image, 'Predicted Caption:' + str_caption_predicted, (2, round(height - 20)), cv.FONT_HERSHEY_COMPLEX, 0.4, (100, 255, 255), 1)
    cv.imshow(window_name, image)
    cv.waitKey(0)

print(rid)
show_captioned2(img_name_val2[rid], ' '.join(result), real_caption)


#%%    BLEU 
bleu1,bleu2,bleu3,bleu4 = 0,0,0,0

lng =  len(img_keys3) 
for i in range(lng):
    chos = np.random.randint(len(img_keys3))
# for i in range(300):
    hidden = decoder.reset_state(batch_size=1)
    image = img_keys3[chos]
    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_test = image_features_extract_model2(temp_input)
    img_tensor_test = tf.reshape(img_tensor_test, (img_tensor_test.shape[0], -1, img_tensor_test.shape[3]))

    features = encoder(img_tensor_test)
    result = []
    
    dec_input = tf.expand_dims([np.where(words == 'x_START_')[0][0]], 0)
    result.append('x_START_')
    flag = 1
    for a in range(1,17): 
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)
        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy() 
        if words[predicted_id] == 'x_END_':
            flag = 0
            result.append(words[predicted_id])
            continue
        if flag:
            result.append(words[predicted_id])
            dec_input = tf.expand_dims([predicted_id], 0)
        else:
            result.append(words[0])

    reference = []
    for k in img_to_cap_vector3[img_keys3[chos]]:
        reference.append(words[k].tolist())
    
    bleu1 += bleu(reference, result, weights=(1, 0, 0, 0))
    bleu2 += bleu(reference, result, weights=(0, 1, 0, 0))
    bleu3 += bleu(reference, result, weights=(0, 0, 1, 0))
    bleu4 += bleu(reference, result, weights=(0, 0, 0, 1))
    if i % 1000 == 0:
        print(i)

print('Individual 1-gram: %f' % (100*bleu1/lng))
print('Individual 2-gram: %f' % (100*bleu2/lng))
print('Individual 3-gram: %f' % (100*bleu3/lng))
print('Individual 4-gram: %f' % (100*bleu4/lng))

#%%

encoder.summary()
decoder.summary()
