import numpy as np
import sys 
import os
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.absolute()))
import pdb
import yaml
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")
from utils import read_data_file, load_images_and_labels
from classifier.DenseNet import pretrained_classifier
import argparse

def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument(
    '--config', '-c', default='configs/celebA_DenseNet_Classifier.yaml'
)    
    args = parser.parse_args()
    # ============= Load config =============
    config_path = os.path.join(str(pathlib.Path(__file__).parent.absolute()), args.config)
    config = yaml.load(open(config_path))
    print(config)
    # ============= Experiment Folder=============
    output_dir = os.path.join(config['log_dir'], config['name'])
    try: os.makedirs(output_dir)
    except: pass
    
    # ============= Experiment Parameters =============
    BATCH_SIZE = config['batch_size']
    EPOCHS = config['epochs']
    channels = config['num_channel']
    input_size = config['input_size']   
    ckpt_dir_continue = config['ckpt_dir_continue']    
    if ckpt_dir_continue == '':
        continue_train = False
    else:
        continue_train = True
        
    # ============= Data =============    
    try:
        categories, file_names_dict = read_data_file(config['image_label_dict'])
        N_CLASSES = len(categories)
    except:
        print("Problem in reading input data file : ", config['image_label_dict'])
        sys.exit()
    data_train = np.load(config['train'])
    data_test = np.load(config['test'])
    print("The classification categories are: ")
    print(categories)
    print('The size of the training set: ', data_train.shape[0])
    print('The size of the testing set: ', data_test.shape[0])
    
    fp = open(os.path.join(output_dir, 'setting.txt'), 'w')
    fp.write('config_file:'+str(config_path)+'\n')
    fp.close()

    # ============= placeholder =============
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, input_size, input_size, channels], name='x-input')
        y_ = tf.placeholder(tf.int64, [None, N_CLASSES], name='y-input') 
        isTrain = tf.placeholder(tf.bool) 
    
    logit = pretrained_classifier(x, n_label=N_CLASSES, reuse=False,  name='classifier', isTrain =isTrain)
    # ============= Loss functions =============  
    classif_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=y_,logits=logit)
    loss = tf.losses.get_total_loss()
    # ============= Optimization functions =============    
    train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)
    # ============= summary =============    
    cls_loss = tf.summary.scalar('classif_loss', classif_loss)
    total_loss = tf.summary.scalar('total_loss', loss)
    sum_train = tf.summary.merge([cls_loss, total_loss])  
    # ============= session =============
    sess=tf.InteractiveSession()
    # Note that this list of variables only include the weights and biases in the model.
    lst_vars = []
    for v in tf.global_variables():
        lst_vars.append(v) 
    saver = tf.train.Saver(var_list=lst_vars)
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(output_dir + '/train', sess.graph)  
    writer_test = tf.summary.FileWriter(output_dir + '/test', sess.graph)

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
    itr_train = 0
    itr_test = 0
    for epoch in range(EPOCHS):
        total_loss = 0.0
        perm = np.arange(data_train.shape[0])
        np.random.shuffle(perm)
        data_train = data_train[perm]  
        num_batch = int(data_train.shape[0]/BATCH_SIZE)
        for i in range(0, num_batch):
            start = i*BATCH_SIZE
            ns = data_train[start:start+BATCH_SIZE]
            xs, ys = load_images_and_labels(ns, config['image_dir'],N_CLASSES, file_names_dict, input_size, channels, do_center_crop=True)
            [_, _loss,summary_str] = sess.run([train_step, loss, sum_train], feed_dict={x:xs, isTrain:True, y_: ys}) 
            writer.add_summary(summary_str, itr_train)  
            itr_train+=1
            total_loss += _loss
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
            xs, ys = load_images_and_labels(ns, config['image_dir'], N_CLASSES, file_names_dict, input_size, channels, do_center_crop = True)            
            [_loss, summary_str] = sess.run([loss, sum_train], feed_dict={x:xs, isTrain:False, y_: ys}) 
            writer_test.add_summary(summary_str, itr_test)
            itr_test+=1
            total_loss += _loss
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