import numpy as np
import pandas as pd
import sys 
import os
import pdb
import yaml
import tensorflow as tf
from classifier.DenseNet_Again import pretrained_classifier
from utils import read_data_file, load_images_and_labels
import argparse
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore")

def test():
    parser = argparse.ArgumentParser()
    parser.add_argument(
    '--config', '-c', default='configs/celebA_YSBBB_Classifier.yaml'
)   
    args = parser.parse_args()
    # ============= Load config =============
    config_path = args.config
    config = yaml.load(open(config_path))
    print(config)
    # ============= Experiment Folder=============
    output_dir = os.path.join(config['log_dir'], config['name'])
    classifier_output_path = os.path.join(output_dir, 'classifier_output')
    try: os.makedirs(classifier_output_path)
    except: pass
    past_checkpoint = output_dir
    # ============= Experiment Parameters =============
    BATCH_SIZE = config['batch_size']
    channels = config['num_channel']
    input_size = config['input_size'] 
    N_CLASSES = config['num_class']  
    # ============= Data =============
    try:
        categories, file_names_dict = read_data_file(config['image_label_dict'])
    except:
        print("Problem in reading input data file : ", config['image_label_dict'])
        sys.exit()
    data_train = np.load(config['train'])
    data_test = np.load(config['test'])
    print("The classification categories are: ")
    print(categories)
    print('The size of the training set: ', data_train.shape[0])
    print('The size of the testing set: ', data_test.shape[0])
    
    # ============= placeholder =============
    with tf.name_scope('input'):
        x_ = tf.placeholder(tf.float32, [None, input_size, input_size, channels], name='x-input')
        y_ = tf.placeholder(tf.int64, [None, N_CLASSES], name='y-input') 
        isTrain = tf.placeholder(tf.bool) 
    # ============= Model =============
    
    if N_CLASSES == 1:
        y = tf.reshape(y_, [-1])
        y = tf.one_hot(y,2,on_value=1.0,off_value=0.0,axis=-1)
        logit,prediction = pretrained_classifier(x_, n_label=2, reuse=False,  name='classifier', isTrain =isTrain)
    else:
        logit,prediction = pretrained_classifier(x_, n_label=N_CLASSES, reuse=False,  name='classifier', isTrain =isTrain)
        y = y_
    classif_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=y,logits=logit)
    loss = tf.losses.get_total_loss()
    # ============= Variables =============
    # Note that this list of variables only include the weights and biases in the model.
    lst_vars = []
    for v in tf.global_variables():
        lst_vars.append(v) 
    # ============= Session =============
    sess=tf.InteractiveSession()
    saver = tf.train.Saver(var_list=lst_vars)
    tf.global_variables_initializer().run()
    # ============= Load Checkpoint =============
    if past_checkpoint is not None:
        ckpt = tf.train.get_checkpoint_state(past_checkpoint+'/')
        if ckpt and ckpt.model_checkpoint_path: 
            print("HERE...................lod checkpoint.........")
            print(str(ckpt.model_checkpoint_path))
            saver.restore(sess, tf.train.latest_checkpoint(past_checkpoint+'/'))
        else:
            sys.exit()
    else:
        sys.exit()
    # ============= Testing Save the Output =============
    names = np.empty([0])
    prediction_y = np.empty([0])
    true_y = np.empty([0])
    for epoch in range(1):  
        num_batch = int(data_train.shape[0]/BATCH_SIZE)
        for i in range(0, num_batch):
            start = i*BATCH_SIZE
            ns = data_train[start:start+BATCH_SIZE]
            xs, ys = load_images_and_labels(ns, config['image_dir'],N_CLASSES, file_names_dict, input_size, channels, do_center_crop=True)
            [_pred] = sess.run([prediction], feed_dict={x_:xs, isTrain:False, y_: ys}) 
            if i == 0:
                names = np.asarray(ns)
                prediction_y = np.asarray(_pred)
                true_y = np.asarray(ys)
            else:
                names = np.append(names, np.asarray(ns), axis= 0)
                prediction_y = np.append(prediction_y, np.asarray(_pred), axis=0)
                true_y = np.append(true_y, np.asarray(ys), axis= 0)
        np.save(classifier_output_path + '/name_train1.npy', names)
        np.save(classifier_output_path + '/prediction_y_train1.npy', prediction_y)
        np.save(classifier_output_path + '/true_y_train1.npy', true_y)

        names = np.empty([0])
        prediction_y = np.empty([0])
        true_y = np.empty([0])
        num_batch = int(data_test.shape[0]/BATCH_SIZE)
        for i in range(0, num_batch):
            start = i*BATCH_SIZE
            ns = data_test[start:start+BATCH_SIZE]
            xs, ys = load_images_and_labels(ns, config['image_dir'],N_CLASSES, file_names_dict, input_size, channels, do_center_crop=True)
            [_pred] = sess.run([prediction], feed_dict={x_:xs, isTrain:False, y_: ys}) 
            if i == 0:
                names = np.asarray(ns)
                prediction_y = np.asarray(_pred)
                true_y = np.asarray(ys)
            else:
                names = np.append(names, np.asarray(ns), axis= 0)
                prediction_y = np.append(prediction_y, np.asarray(_pred), axis=0)
                true_y = np.append(true_y, np.asarray(ys), axis= 0)
        np.save(classifier_output_path + '/name_test1.npy', names)
        np.save(classifier_output_path + '/prediction_y_test1.npy', prediction_y)
        np.save(classifier_output_path + '/true_y_test1.npy', true_y)
    
if __name__ == "__main__":
    test()