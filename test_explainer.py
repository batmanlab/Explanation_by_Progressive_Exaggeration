import sys 
import os
from classifier.DenseNet import pretrained_classifier
from explainer.networks_128 import Discriminator_Ordinal, Generator_Encoder_Decoder
import tensorflow.contrib.slim as slim
import tensorflow as tf
import numpy as np
from utils import *
from losses import *
import pdb
import yaml
import time
import scipy.io as sio
from datetime import datetime
import random
import warnings
import argparse
warnings.filterwarnings("ignore", category=DeprecationWarning) 
np.random.seed(0)

def convert_ordinal_to_binary(y,n):
    y = np.asarray(y).astype(int)
    new_y = np.zeros([y.shape[0], n])
    new_y[:,0] = y
    for i in range(0,y.shape[0]):
        for j in range(1,y[i]+1):
            new_y[i,j] = 1
    return new_y
def Train():    
    parser = argparse.ArgumentParser()
    parser.add_argument(
    '--config', '-c', default='configs/celebA_Young_Explainer.yaml')   
    args = parser.parse_args()
    
    # ============= Load config =============
    config_path = args.config
    config = yaml.load(open(config_path))
    print(config)
    
    # ============= Experiment Folder=============
    assets_dir = os.path.join(config['log_dir'], config['name'])
    log_dir = os.path.join(assets_dir, 'log')
    ckpt_dir = os.path.join(assets_dir, 'ckpt_dir')
    sample_dir = os.path.join(assets_dir, 'sample')
    test_dir = os.path.join(assets_dir, 'test')
    
    # ============= Experiment Parameters =============
    ckpt_dir_cls = config['cls_experiment']   
    BATCH_SIZE = config['num_bins']
    EPOCHS = config['epochs']
    channels = config['num_channel']
    input_size = config['input_size'] 
    NUMS_CLASS_cls = config['num_class']   
    NUMS_CLASS = config['num_bins']
    target_class = config['target_class']
    lambda_GAN = config['lambda_GAN']
    lambda_cyc = config['lambda_cyc']
    lambda_cls = config['lambda_cls']  
    save_summary = int(config['save_summary'])
    ckpt_dir_continue = ckpt_dir
    count_to_save = config['count_to_save']
    # ============= Data =============
    try:
        categories, file_names_dict = read_data_file(config['image_label_dict'])
    except:
        print("Problem in reading input data file : ", config['image_label_dict'])
        sys.exit()
    data = np.asarray(file_names_dict.keys())
    print("The classification categories are: ")
    print(categories)
    print('The size of the training set: ', data.shape[0])
    
    # ============= placeholder =============
    x_source = tf.placeholder(tf.float32, [None, input_size, input_size, channels])
    y_s = tf.placeholder(tf.int32, [None, NUMS_CLASS])  
    y_source = y_s[:,0]
    train_phase = tf.placeholder(tf.bool)
    
    y_t = tf.placeholder(tf.int32, [None, NUMS_CLASS]) 
    y_target = y_t[:,0]
    
    # ============= G & D =============    
    G = Generator_Encoder_Decoder("generator") # with conditional BN, SAGAN: SN here as well
    D = Discriminator_Ordinal("discriminator") #with SN and projection
    
    real_source_logits = D(x_source, y_s, NUMS_CLASS, "NO_OPS")
    
    fake_target_img, fake_target_img_embedding = G(x_source, train_phase, y_target, NUMS_CLASS)
    fake_source_img, fake_source_img_embedding = G(fake_target_img, train_phase, y_source, NUMS_CLASS)
    fake_source_recons_img, x_source_img_embedding = G(x_source, train_phase, y_source, NUMS_CLASS)    
    fake_target_logits = D(fake_target_img, y_t, NUMS_CLASS, None)    
    
    # ============= pre-trained classifier =============      
    real_img_cls_logit_pretrained,  real_img_cls_prediction = pretrained_classifier(x_source, NUMS_CLASS_cls, reuse=False, name='classifier')
    fake_img_cls_logit_pretrained, fake_img_cls_prediction = pretrained_classifier(fake_target_img, NUMS_CLASS_cls, reuse=True)
    real_img_recons_cls_logit_pretrained, real_img_recons_cls_prediction = pretrained_classifier(fake_source_img, NUMS_CLASS_cls, reuse=True)
    
    # ============= session =============
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    # ============= Checkpoints =============
    print(" [*] before training, Load checkpoint ")
    print(" [*] Reading checkpoint...")

    ckpt = tf.train.get_checkpoint_state(ckpt_dir_continue)   
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(ckpt_dir_continue, ckpt_name))
        print(ckpt_dir_continue, ckpt_name)
        print("Successful checkpoint upload")
    else:
        print("Failed checkpoint load")
        sys.exit()

    # ============= load pre-trained classifier checkpoint =============
    class_vars = [var for var in slim.get_variables_to_restore() if 'classifier' in var.name]
    name_to_var_map_local = {var.op.name: var for var in class_vars}               
    temp_saver = tf.train.Saver(var_list=name_to_var_map_local)
    ckpt = tf.train.get_checkpoint_state(ckpt_dir_cls) 
    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
    temp_saver.restore(sess, os.path.join(ckpt_dir_cls, ckpt_name))
    print("Classifier checkpoint loaded.................")
    print(ckpt_dir_cls, ckpt_name)
    
    # ============= Testing =============
    real_img = np.empty([0])
    fake_images = np.empty([0])
    embedding = np.empty([0])
    s_embedding = np.empty([0])
    recons = np.empty([0])
    real_pred = np.empty([0])
    fake_pred = np.empty([0])
    recons_pred = np.empty([0])
    names = np.empty([0]) 
    
    np.random.shuffle(data)   
    np.random.shuffle(data) 
    np.random.shuffle(data) 
    data = data[0:count_to_save]
    for i in range(data.shape[0] // BATCH_SIZE):
        image_paths = data[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        img, labels = load_images_and_labels(image_paths, '',1, file_names_dict, input_size, channels, do_center_crop=True)
        img_repeat = np.repeat(img, NUMS_CLASS, 0)

        labels = labels.ravel()
        labels = np.repeat(labels, NUMS_CLASS, 0)
        source_labels = convert_ordinal_to_binary(labels,NUMS_CLASS)

        target_labels = np.asarray([np.asarray(range(NUMS_CLASS)) for j in range(img.shape[0])])
        target_labels = target_labels.ravel()
        target_labels = convert_ordinal_to_binary(target_labels,NUMS_CLASS) 

        FAKE_IMG, f_embed, recons_img, real_p, fake_p, recons_p, s_embed = sess.run([fake_target_img, fake_target_img_embedding, fake_source_img, real_img_cls_prediction, fake_img_cls_prediction, real_img_recons_cls_prediction,x_source_img_embedding], feed_dict={y_t: target_labels,  x_source: img_repeat, train_phase: False, y_s:source_labels})  
        if i == 0:
            real_img = img
            fake_images = FAKE_IMG
            embedding = f_embed
            s_embedding = s_embed
            recons = recons_img
            names = np.asarray(image_paths)
            real_pred = real_p
            fake_pred = fake_p
            recons_pred = recons_p           
        else:
            real_img = np.append(real_img, img, axis = 0)
            fake_images =np.append(fake_images, FAKE_IMG, axis = 0)
            embedding = np.append(embedding, f_embed, axis = 0)
            s_embedding = np.append(s_embedding, f_embed, axis = 0)
            recons = np.append(recons, recons_img, axis = 0)
            names = np.append(names, np.asarray(image_paths), axis = 0)
            real_pred = np.append(real_pred, real_p, axis = 0)
            fake_pred = np.append(fake_pred, fake_p, axis = 0)
            recons_pred = np.append(recons_pred, recons_p, axis = 0)
        print(i)

        if i % 100 == 0:
            np.save(os.path.join(test_dir + '/real_img.npy'),   real_img     )
            np.save(os.path.join(test_dir + '/fake_images.npy'),   fake_images     )
            np.save(os.path.join(test_dir + '/embedding.npy'),   embedding     )
            np.save(os.path.join(test_dir + '/s_embedding.npy'),   s_embedding     )
            np.save(os.path.join(test_dir + '/recons.npy'),   recons     )
            np.save(os.path.join(test_dir + '/names.npy'),   names     )
            np.save(os.path.join(test_dir + '/real_pred.npy'),   real_pred     )    
            np.save(os.path.join(test_dir + '/fake_pred.npy'),   fake_pred     )    
            np.save(os.path.join(test_dir + '/recons_pred.npy'),   recons_pred     )   

    np.save(os.path.join(test_dir + '/real_img.npy'),   real_img     )
    np.save(os.path.join(test_dir + '/fake_images.npy'),   fake_images     )
    np.save(os.path.join(test_dir + '/embedding.npy'),   embedding     )
    np.save(os.path.join(test_dir + '/s_embedding.npy'),   s_embedding     )
    np.save(os.path.join(test_dir + '/recons.npy'),   recons     )
    np.save(os.path.join(test_dir + '/names.npy'),   names     )
    np.save(os.path.join(test_dir + '/real_pred.npy'),   real_pred     )    
    np.save(os.path.join(test_dir + '/fake_pred.npy'),   fake_pred     )    
    np.save(os.path.join(test_dir + '/recons_pred.npy'),   recons_pred     ) 



            

if __name__ == "__main__":
    Train()