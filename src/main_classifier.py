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
    parser.add_argument('--name',                  type=str,      default= 'DenseNet_Young',           help='Name of the experiment')
    parser.add_argument('--input_data_file_train', type=str,      default= '../Data/CelebA/Young_binary_classification_train.txt',              help='Path containing image names and labels for training set.')
    parser.add_argument('--input_data_file_test',  type=str,      default= '../Data/CelebA/Young_binary_classification_test.txt',              help='Path containing image names and labels for test set.')
    parser.add_argument('--image_dir',             type=str,      default= '../Data/CelebA/images', help='The folder containing the images.')
    parser.add_argument('--ckpt_dir_continue',     type=str,      default= '',             help='Path to the directory with the checkpoint of the classifier, to continue the training from a saved checkpoint.')
    
    parser.add_argument('--num_channel',          type=int,       default= 3,           help='Number of channels in the input images.')
    parser.add_argument('--input_size',           type=int,       default= 128,           help='Size of the input image.')
    parser.add_argument('--num_classes',          type=int,       default= 1,            help='Number of classes in classifier. Default=1 for binary classifier.')
    parser.add_argument('--batch_size',           type=int,       default= 64,            help='Batch size of the GAN training.')
    parser.add_argument('--epochs',               type=int,       default= 5,            help='Number of epochs for training.')
    
    args = parser.parse_args()
    
    # ============= Experiment Folder=============
    output_dir = os.path.join('../output/classifier',args.name)
    try: os.makedirs(output_dir)
    except: pass
    
    # ============= Experiment Parameters =============
    N_CLASSES = args.num_classes
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    channels = args.num_channel
    input_size = args.input_size
    continue_train = False
    
    ckpt_dir_continue = args.ckpt_dir_continue
    if ckpt_dir_continue == '':
        continue_train = False
    else:
        continue_train = True
        
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
    
    fp = open(os.path.join(output_dir, 'setting.txt'))
    fp.write('name:'+str(args.name)+'\n')
    fp.write('input_data_file_train:'+str(args.input_data_file_train)+'\n')
    fp.write('input_data_file_test:'+str(args.input_data_file_test)+'\n')
    fp.write('image_dir:'+str(args.image_dir)+'\n')
    fp.write('ckpt_dir_continue:'+str(args.ckpt_dir_continue)+'\n')
    fp.write('num_channel:'+str(args.num_channel)+'\n')
    fp.write('num_classes:'+str(args.num_classes)+'\n')
    fp.write('batch_size:'+str(args.batch_size)+'\n')
    fp.write('output_dir:'+str(output_dir)+'\n')
    fp.write('categories:'+str(categories)+'\n')
    fp.write('batch_size:'+str(args.batch_size)+'\n')
    fp.write('batch_size:'+str(args.batch_size)+'\n')
    fp.close()
    
    
    
    
    
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
    if continue_train:
        print("Before training, Load checkpoint ")
        print("Reading checkpoint...")
        ckpt = tf.train.get_checkpoint_state(ckpt_dir_continue)
        if ckpt and ckpt.model_checkpoint_path: 
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(ckpt_dir_continue, ckpt_name))
            print(ckpt_name)
            print("Successful checkpoint upload")
        else:
            print("Failed checkpoint load")
            sys.exit()
    
    # ============= Training =============
    train_loss = []
    test_loss = []

    for epoch in range(EPOCHS):
        total_loss = 0.0
        perm = np.arange(data_train.shape[0])
        np.random.shuffle(perm)
        data_train = data_train[perm]  
        num_batch = int(data_train.shape[0]/BATCH_SIZE)
        for i in range(0, num_batch):
            start = i*BATCH_SIZE
            ns = data_train[start:start+BATCH_SIZE]
            xs, ys = load_images_and_labels(ns, N_CLASSES, file_names_dict_train, input_size, channels, True)
            [_, _loss] = sess.run([train_step, loss], feed_dict={x:xs, isTrain:True, y_: ys}) 
            total_loss += _loss
            if i == 3:
                break
        total_loss /= i
        print("Epoch: " + str( epoch) + " loss: " + str(total_loss) + '\n')
        train_loss.append(total_loss)

        total_loss = 0.0
        perm = np.arange(data_test.shape[0])
        np.random.shuffle(perm)
        data_test = data_test[perm]  
        num_batch = int(data_test.shape[0]/BATCH_SIZE)
        for i in range(0, num_batch):
            start = i*BATCH_SIZE
            ns = data_test[start:start+BATCH_SIZE]
            xs, ys = load_images_and_labels(ns, N_CLASSES, file_names_dict_test, input_size, channels, True)
            [_loss] = sess.run([loss], feed_dict={x:xs, isTrain:False, y_: ys}) 
            total_loss += _loss
            if i == 3:
                break
        total_loss /= i
        print("Epoch: "+ str(epoch) + " Test loss: "+ str(total_loss) + '\n')
        test_loss.append(total_loss)
        
        if epoch %2 == 0:
            checkpoint_name = os.path.join( output_dir, 'cp1_epoch'+str(epoch)+'.ckpt')
            save_path = saver.save(sess, checkpoint_name)
            np.save( os.path.join( output_dir,'train_loss.npy'), np.asarray(train_loss))
            np.save( os.path.join( output_dir,'test_loss.npy'), np.asarray(test_loss))
    f.close()
    
if __name__ == "__main__":
    main()