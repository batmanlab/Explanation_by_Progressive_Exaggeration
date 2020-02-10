import numpy as np
import sys 
import os
import tensorflow.contrib.slim as slim
from datetime import datetime
import pdb
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")
from DenseNet_Layers import *
from Data_Reader import *
from DenseNet_Classifier import pretrained_classifier
import argparse

def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cls_experiment',        type=str,       default= '../output/classifier', help='Folder where trained classifier checkpoints are saved.')
    parser.add_argument('--input_data_file_train', type=str,      default= '../Data/CelebA/Young_binary_classification_train.txt',              help='Path containing image names and labels for training set.')
    parser.add_argument('--input_data_file_test',  type=str,      default= '../Data/CelebA/Young_binary_classification_test.txt',              help='Path containing image names and labels for test set.')
    parser.add_argument('--image_dir',             type=str,      default= '../Data/CelebA/images', help='The folder containing the images.')
    
    parser.add_argument('--num_channel',          type=int,       default= 3,           help='Number of channels in the input images.')
    parser.add_argument('--input_size',           type=int,       default= 128,           help='Size of the input image.')
    parser.add_argument('--num_classes',          type=int,       default= 1,            help='Number of classes in classifier. Default=2 for binary classifier.')
    parser.add_argument('--batch_size',           type=int,       default= 64,            help='Batch size of the GAN training.')
    
    args = parser.parse_args()
    
    # ============= Experiment Folder=============
    output_dir = os.path.join(cls_experiment,'test')
    try: os.makedirs(output_dir)
    except: pass
    
    # ============= Experiment Parameters =============
    N_CLASSES = args.num_classes
    BATCH_SIZE = args.batch_size
    EPOCHS =1
    channels = args.num_channel
    input_size = args.input_size
        
    # ============= Data =============
    categories, file_names_dict_train = read_data_file(args.input_data_file_train, args.image_dir)
    try:
        categories, file_names_dict_train = read_data_file(args.input_data_file_train, args.image_dir)
        categories, file_names_dict_test = read_data_file(args.input_data_file_test, args.image_dir)
    except:
        print("Problem in reading input data file : ", args.input_data_file_train, args.input_data_file_test)
        sys.exit()
    data_train = np.asarray(file_names_dict_train.keys())
    data_test = np.asarray(file_names_dict_test.keys())
    print("The classification categories are: ")
    print(categories)
    print('The size of the training set: ', data_train.shape[0])
    print('The size of the testing set: ', data_test.shape[0])
    
    # ============= placeholder =============
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, input_size, input_size, channels], name='x-input')
        y_ = tf.placeholder(tf.int64, [None, N_CLASSES], name='y-input') 
        isTrain = tf.placeholder(tf.bool) 
    
    logit, prediction, _ = pretrained_classifier(x, n_label=N_CLASSES, reuse=False,  name='classifier', isTrain =isTrain)
    # ============= Loss functions =============  
    classif_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=y_,logits=logit)
    loss = tf.losses.get_total_loss()
    # ============= Optimization functions =============    
    train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)
    
    # ============= session =============
    sess=tf.InteractiveSession()
    # Note that this list of variables only include the weights and biases in the model.
    lst_vars = []
    for v in tf.global_variables():
        lst_vars.append(v) 
    saver = tf.train.Saver(var_list=lst_vars)
    sess.run(tf.global_variables_initializer())
    
    # ============= Checkpoints =============
    print("Before testing, Load checkpoint ")
    print("Reading checkpoint...")
    ckpt = tf.train.get_checkpoint_state(args.cls_experiment)
    if ckpt and ckpt.model_checkpoint_path: 
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(args.cls_experiment, ckpt_name))
        print(ckpt_name)
        print("Successful checkpoint upload")
    else:
        print("Failed checkpoint load: ", args.cls_experiment)
        sys.exit()
    
    # ============= Training =============
    names = np.empty([0])
    prediction_y = np.empty([0])
    true_y = np.empty([0])

    for epoch in range(EPOCHS):  
        num_batch = int(data_train.shape[0]/BATCH_SIZE)
        for i in range(0, num_batch):
            start = i*BATCH_SIZE
            ns = data_train[start:start+BATCH_SIZE]
            xs, ys = load_images_and_labels(ns, N_CLASSES, file_names_dict_train, input_size, channels, True)
            [_pred] = sess.run([prediction], feed_dict={x:xs, isTrain:False, y_: ys}) 
            if i == 0:
                names = np.asarray(ns)
                prediction_y = np.asarray(_pred)
                true_y = np.asarray(ys)
            else:
                names = np.append(names, np.asarray(ns), axis= 0)
                prediction_y = np.append(prediction_y, np.asarray(_pred), axis=0)
                true_y = np.append(true_y, np.asarray(ys), axis= 0)
                
        np.save(output_dir + '/name_train.npy', names)
        np.save(output_dir + '/prediction_y_train.npy', prediction_y)
        np.save(output_dir + '/true_y_train.npy', true_y)

        names = np.empty([0])
        prediction_y = np.empty([0])
        true_y = np.empty([0])
        data_test = data_test[perm]  
        num_batch = int(data_test.shape[0]/BATCH_SIZE)
        for i in range(0, num_batch):
            start = i*BATCH_SIZE
            ns = data_test[start:start+BATCH_SIZE]
            xs, ys = load_images_and_labels(ns, N_CLASSES, file_names_dict_test, input_size, channels, True)
            [_pred] = sess.run([prediction], feed_dict={x:xs, isTrain:False, y_: ys}) 
            if i == 0:
                names = np.asarray(ns)
                prediction_y = np.asarray(_pred)
                true_y = np.asarray(ys)
            else:
                names = np.append(names, np.asarray(ns), axis= 0)
                prediction_y = np.append(prediction_y, np.asarray(_pred), axis=0)
                true_y = np.append(true_y, np.asarray(ys), axis= 0)
        np.save(output_dir + '/name_test.npy', names)
        np.save(output_dir + '/prediction_y_test.npy', prediction_y)
        np.save(output_dir + '/true_y_test.npy', true_y)
    
if __name__ == "__main__":
    main()