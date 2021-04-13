#!/usr/bin/env python
# -*-coding: utf-8 -*-

#SCRIPT: data_processing_flickr30k.py
#AUTHOR: Osman Mamun
#DATE CREATED: 04-12-2021

import os
import tensorflow as tf
import importlib
from PIL import Image
import pandas as pd
import json
from tqdm import tqdm
from glob import glob
import numpy as np
import pickle

class CNN_Encoder(tf.keras.Model):
    def __init__(self, image_pretrained_model=None):
        super(CNN_Encoder, self).__init__()
        self.image_pretrained_model = image_pretrained_model

    def build(self, input_shape):
        mod_path = f'tensorflow.keras.applications.{self.image_pretrained_model[0]}'
        pre_mod = importlib.import_module(mod_path)
        self.image_model = getattr(pre_mod,
            self.image_pretrained_model[1])(include_top=False, weights='imagenet')
        self.new_input = self.image_model.input
        self.hidden_layer = self.image_model.layers[-1].output
        self.image_features_extract_model = tf.keras.Model(self.new_input,
            self.hidden_layer)
        self.image_features_extract_model.trainable = False

    def call(self, x):
        return self.image_features_extract_model(x)

class WordVectorizer:
    def fit(self, texts=None, top_k=5000):
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
        oov_token="<unk>", filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
        self.tokenizer.fit_on_texts(texts)
        self.tokenizer.word_index['<pad>'] = 0
        self.tokenizer.index_word[0] = '<pad>'
    def predict(self, texts=None):
        train_seqs = self.tokenizer.texts_to_sequences(texts)
        cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs,
        padding='post')
        return cap_vector

class ImageDataset:
    def __init__(self, 
                 name='COCO', 
                 image_pretrained_model=('vgg16', 'VGG16')):
        self.name = name
        self.image_pretrained_model = image_pretrained_model
        self.root = '/Users/mamu867/PNNL_Mac/Springboard/image_caption_generator/'
    
    @staticmethod
    def read_file(filename):
        _, extension = os.path.splitext(filename)

        if extension not in ['.json', '.txt', '.csv']:
            print(f'File read for extension {extension} is not yet available.')
            return
        if extension == '.txt':
            with open(filename, 'r') as f:
                return f.read()
        if extension == '.json':
            with open(filename, 'r') as f:
                return pd.DataFrame.from_dict(json.load(f)['annotations'])
        if extension == '.csv':
            return pd.read_csv(filename, delimiter='|')
    
    def load_annotation(self):
        if self.name == 'Flickr30k':
            fpath = self.root + 'data/raw/Flickr30k/flickr30k_images/'
            self.annotations = self.__class__.read_file(fpath + 'results.csv')
            self.annotations['image_path'] = self.annotations.apply(
                lambda x: fpath + x['image_name'], axis=1)
            self.annotations.rename(columns={' comment': 'caption'}, inplace=True)
            self.annotations['image_path'] = self.annotations.apply(
                lambda x: fpath + 'flickr30k_images/' + x['image_name'], axis=1)
            self.annotations['image_id'] = self.annotations.apply(
                lambda x: float(x['image_name'].split('.')[0]),axis=1)
            self.annotations.sort_values(by='image_id', inplace=True)
            self.annotations.dropna(inplace=True)
            self.annotations['caption'] = self.annotations.apply(
                lambda x: '<start> ' + x['caption'] + ' <end>', axis = 1)
        else:
            root_path = self.root + 'data/raw/COCO/'
            train_path = root_path + 'annotations/captions_train2014.json'
            #val_path = root_path + 'annotations/captions_val2014.json'
            self.annotations = self.__class__.read_file(train_path)
            self.annotations['image_path'] = self.annotations.apply(
                lambda x: root_path + 'train2014/' + 'COCO_train2014_{0:012d}.jpg'.format(x['image_id']), axis=1)
            self.annotations.sort_values(by='image_id', inplace=True)
            self.annotations.dropna(inplace=True)
            self.annotations['caption'] = self.annotations.apply(
                lambda x: '<start> ' + x['caption'] + ' <end>', axis = 1)
            #self.annotations_val = self.__class__.read_file(val_path)
            #self.annotations_val['image_path'] = self.annotations_val.apply(
            # lambda x: root_path + 'val2014/' + 'COCO_val2014_{0:012d}.jpg'.format(
            # x['image_id']), axis=1)
            #self.annotations_val.sort_values(by='image_id', inplace=True)
    
    @tf.autograph.experimental.do_not_convert
    def load_image(self, image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (299, 299))
        mod_path = f'tensorflow.keras.applications.{self.image_pretrained_model[0]}'
        pre_mod = importlib.import_module(mod_path)
        img = pre_mod.preprocess_input(img)
        return img

    def get_image_features(self, f_locs, image_features_extract_model):
        X = tf.convert_to_tensor(np.array([self.load_image(f).numpy() 
        for f in f_locs]), dtype=tf.float64)
        batch_features = image_features_extract_model(X)
        batch_features = tf.reshape(batch_features, 
                                (batch_features.shape[0], -1, 
                                 batch_features.shape[3]))
        return batch_features
    
    def vec_initializer(self, train_size=400, batch_size=64, top_k=10000):
        self.wvec = WordVectorizer()
        texts = self.annotations['caption'].tolist()[:train_size*batch_size]
        self.wvec.fit(texts=texts, top_k=top_k)
        self.tokenizer = self.wvec.tokenizer
        self.cnn_encoder = CNN_Encoder(self.image_pretrained_model)

    def data_processor(self, batch_size=64, n_take=(1000, 200, 200)):
        for n, i in enumerate(range(0, len(self.annotations), batch_size), 1):
            print('Currently aggregating data for: {}'.format(n))
            if self.name == 'COCO':
                if n < n_take[0]+1:
                    path = self.root + '/data/raw/COCO/train_vectors'
                    data_type = 'train'
                    count = n 
                elif n > n_take[0] and n < sum(n_take[:2])+1:
                    path = self.root + '/data/raw/COCO/val_vectors'
                    data_type = 'val'
                    count = n - n_take[0]
                elif n > sum(n_take[:2]) and n < sum(n_take)+1:
                    path = self.root + '/data/raw/COCO/test_vectors'
                    data_type = 'test'
                    count = n - n_take[0] - n_take[1]
                else:
                    break
            if self.name == 'Flickr30k':
                if n < n_take[0]+1:
                    path = self.root + '/data/raw/Flickr30k/flickr30k_images/train_vectors'
                    data_type = 'train'  
                    count = n 
                elif n > n_take[0] and n < sum(n_take[:2])+1:
                    path = self.root + '/data/raw/Flickr30k/flickr30k_images/val_vectors'
                    data_type = 'val'
                    count = n - n_take[0]
                elif n > sum(n_take[:2]) and n < sum(n_take)+1:
                    path = self.root + '/data/raw/Flickr30k/flickr30k_images/test_vectors'
                    data_type = 'test'
                    count = n - n_take[0] - n_take[1]
                else:
                    break

            temp_df = self.annotations[i:i+batch_size]
            y = self.wvec.predict(temp_df['caption'].tolist())
            X = self.get_image_features(temp_df['image_path'].values, self.cnn_encoder)
            np.save(os.path.join(path, '{}_{}_{}_X_y_{:04d}_{:012d}_{:012d}.npy'.format(
                        data_type,
                        self.image_pretrained_model[0], 
                        self.image_pretrained_model[1], 
                        count, 
                        int(temp_df['image_id'].min()), 
                        int(temp_df['image_id'].max()))), {'X': X, 'y': y})
                
if __name__ == '__main__':
    imdset_flick = ImageDataset('Flickr30k')
    imdset_flick.load_annotation()
    imdset_flick.vec_initializer(train_size=160)

    with open('/Users/mamu867/PNNL_Mac/Springboard/image_caption_generator/data/interim/fickr30k/tokenizer.pickle', 'wb') as handle:
        pickle.dump(imdset_flick.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    imdset_flick.data_processor(n_take=(160, 40, 5))
