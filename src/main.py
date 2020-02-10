import sys 
import os
from datetime import datetime
import warnings
import argparse
warnings.filterwarnings("ignore") 
sys.path.append(os.path.dirname(__file__))
from explain import Discriminator_Ordinal, Generator_Encoder_Decoder
from DenseNet_Classifier import pretrained_classifier
#from AlexNet_Classifier import pretrained_classifier
from Data_Reader import *


def convert_ordinal_to_binary(y,n):
    y = np.asarray(y).astype(int)
    new_y = np.zeros([y.shape[0], n])
    new_y[:,0] = y
    for i in range(0,y.shape[0]):
        for j in range(1,y[i]+1):
            new_y[i,j] = 1
    return new_y

def main():   
    parser = argparse.ArgumentParser()
    parser.add_argument('--cls_experiment',       type=str,       default= '../output/classifier', help='Main folder where trained classifier, that our model is going to explain is saved. It should have save checkpoints to load the classifier.')
    parser.add_argument('--name',                 type=str,       default= 'Explainer',    help='Name of the experiment')
    parser.add_argument('--input_data_file',      type=str,       default= '../Data/tmp.txt',              help='The path to the data file containing image-paths and the bin-number labels.')
    parser.add_argument('--image_dir',            type=str,       default= '../Data/CelebA/images', help='The folder containing the images.')
    parser.add_argument('--ckpt_dir_continue',     type=str,      default= '../output/explainer/',     help='Path to the directory with the checkpoint of the explainer, to continue the training from a saved checkpoint.')
    
    parser.add_argument('--num_channel',          type=int,       default= 3,           help='Number of channels in the input images.')
    parser.add_argument('--input_size',           type=int,       default= 128,           help='Size of the input image.')
    parser.add_argument('--num_classes_cls',      type=int,       default= 2,            help='Number of classes in classifier. Default=2 for binary classifier.')
    parser.add_argument('--target_class',         type=int,       default= 1,             help='Target class of the classifier which the explainer is going to explain.')
    parser.add_argument('--num_classes_explainer',type=int,       default= 10,              help='The number of bins in which p(y|x) is discretized into. Default = 10 corresponding to delta (step-size = 0.1).')
    parser.add_argument('--delta',                type=float,     default= 0.1,             help='The size of the bin, delta (step-size = 0.1).')    
    parser.add_argument('--batch_size',           type=int,       default= 32,            help='Batch size of the GAN training.')
    parser.add_argument('--epochs',               type=int,       default= 300,         help='Number of epochs for training.')
    parser.add_argument('--lambda_GAN',           type=int,       default= 1,              help='The hyper-paramter for explainer training controlling adverserial loss.')
    parser.add_argument('--lambda_cyc',           type=int,       default= 100,             help='The hyper-parameter for explainer training controlling reconstruction and cyclic loss.')
    parser.add_argument('--lambda_cls',           type=int,       default= 1,              help='The hyper-paramter for explainer training controlling the KL divergence loss for the classifier under inspection.')    
     
    args = parser.parse_args()

    # ============= Classifier =============
    classifier_experiment_dir  = args.cls_experiment
    if not os.path.exists(classifier_experiment_dir):
        print('Provide path to the folder where classifier\'s checkpoint is saved. The provided path: ' + str(classifier_experiment_dir) + ' is not valid.')
        sys.exist()
    
    # ============= Experiment Folder=============
    current_time = datetime.now().strftime("%H%M%S")
    assets_dir = os.path.join('../output/explainer',args.name+'_'+current_time)
    log_dir = os.path.join(assets_dir, 'log')
    ckpt_dir = os.path.join(assets_dir, 'ckpt_dir')
    sample_dir = os.path.join(assets_dir, 'sample')
    test_dir = os.path.join(assets_dir, 'test')
    # make directory if not exist
    try: os.makedirs(log_dir)
    except: pass
    try: os.makedirs(ckpt_dir)
    except: pass
    try: os.makedirs(sample_dir)
    except: pass
    try: os.makedirs(test_dir)
    except: pass
    
    # ============= Experiment Parameters =============
    lambda_GAN = args.lambda_GAN
    lambda_cyc = args.lambda_cyc
    lambda_cls = args.lambda_cls
    NUMS_CLASS = args.num_classes_explainer
    NUMS_CLASS_cls = args.num_classes_cls
    target_class = args.target_class
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    channels = args.num_channel
    input_size = args.input_size
    delta = args.delta
    continue_train = False

    ckpt_dir_continue = args.ckpt_dir_continue
    if ckpt_dir_continue == '':
        continue_train = False
    else:
        ckpt_dir_continue = os.path.join(ckpt_dir_continue, 'ckpt_dir')
        continue_train = True
    # ============= Data =============
    try:
        categories, file_names_dict = read_data_file(args.input_data_file, args.image_dir)
    except:
        print("Problem in reading input data file : ", args.input_data_file)
        sys.exit()
    data = np.asarray(file_names_dict.keys())
    print("The binning categories are: ")
    print(categories)
        
    # ============= placeholder =============
    x_source = tf.placeholder(tf.float32, [BATCH_SIZE, input_size, input_size, channels])
    y_source = tf.placeholder(tf.int32, [BATCH_SIZE, NUMS_CLASS]) 
    train_phase = tf.placeholder(tf.bool)
    y_target = tf.placeholder(tf.int32, [None, NUMS_CLASS]) 
    
    # ============= Condition =============    
    y_s = y_source[:,0] #numerical condition, bin-number between [0, NUMS_CLASS-1]
    y_t = y_target[:,0] #numerical condition, bin-number between [0, NUMS_CLASS-1]
    # ============= G & D =============    
    G = Generator_Encoder_Decoder("generator")
    D = Discriminator_Ordinal("discriminator")
    # ============= GAN =============    
    #Generator
    fake_target_img, fake_target_img_embedding = G(x_source, train_phase, y_t, NUMS_CLASS)
    fake_source_img, fake_source_img_embedding = G(fake_target_img, train_phase, y_s, NUMS_CLASS)
    fake_source_recons_img, fake_source_recons_img_embedding = G(x_source, train_phase, y_s, NUMS_CLASS)
    #Discriminator
    real_source_logits = D(x_source, y_source, NUMS_CLASS, "NO_OPS")
    fake_target_logits = D(fake_target_img, y_target, NUMS_CLASS, None)
    #pre-trained classifier
    real_img_cls_logit,  real_img_cls_prediction, _ = pretrained_classifier(x_source, NUMS_CLASS_cls, reuse=False, name='classifier')
    fake_img_cls_logit, fake_img_cls_prediction, _ = pretrained_classifier(fake_target_img, NUMS_CLASS_cls, reuse=True, name='classifier')
    real_img_recons_cls_logit, real_img_recons_cls_prediction, _ = pretrained_classifier(fake_source_img, NUMS_CLASS_cls, reuse=True, name='classifier')

    # ============= Loss functions =============    
    # pre-trained classifier loss, Entropy loss, KL divergence loss
    real_p = tf.cast(y_t, tf.float32)*delta
    fake_q = fake_img_cls_prediction[:,target_class] 
    fake_evaluation = (real_p * tf.log(fake_q) ) + ( (1-real_p) * tf.log(1-fake_q) )
    fake_evaluation = -tf.reduce_mean(fake_evaluation)
    
    recons_evaluation = (real_img_cls_prediction[:,target_class] * tf.log(real_img_recons_cls_prediction[:,target_class]) ) + ( (1-real_img_cls_prediction[:,target_class]) * tf.log(1-real_img_recons_cls_prediction[:,target_class]) )
    recons_evaluation = -tf.reduce_mean(recons_evaluation)  
    
    # Adversarial Loss
    D_loss_GAN = discriminator_loss('hinge', real_source_logits, fake_target_logits) 
    G_loss_GAN = generator_loss('hinge', fake_target_logits)
    
    # cyclic and reconstruction loss
    G_loss_cyc = l1_loss(x_source, fake_source_img) 
    G_loss_rec = l2_loss(fake_source_recons_img_embedding, fake_source_img_embedding)
    
    #Overall Loss
    G_loss = (G_loss_GAN * lambda_GAN) + ((G_loss_rec + G_loss_cyc) * lambda_cyc) + ((fake_evaluation  + recons_evaluation) * lambda_cls)
    D_loss = (D_loss_GAN * lambda_GAN) 
    
    # ============= Optimization functions =============    
    D_opt = tf.train.AdamOptimizer(2e-4, beta1=0., beta2=0.9).minimize(D_loss, var_list=D.var_list())
    G_opt = tf.train.AdamOptimizer(2e-4, beta1=0., beta2=0.9).minimize(G_loss, var_list=G.var_list())
    
    # ============= summary =============
    real_img_sum = tf.summary.image('real_img', x_source)
    fake_img_sum = tf.summary.image('fake_target_img', fake_target_img)
    fake_source_img_sum = tf.summary.image('fake_source_img', fake_source_img)
    fake_source_recons_img_sum = tf.summary.image('fake_source_recons_img', fake_source_recons_img)
    loss_g_sum = tf.summary.scalar('loss_g', G_loss)
    loss_d_sum = tf.summary.scalar('loss_d', D_loss)
    loss_g_GAN_sum = tf.summary.scalar('loss_g_GAN', G_loss_GAN)
    loss_d_GAN_sum = tf.summary.scalar('loss_d_GAN', D_loss_GAN)
    loss_g_cyc_sum = tf.summary.scalar('G_loss_cyc', G_loss_cyc)
    G_loss_rec_sum = tf.summary.scalar('G_loss_rec', G_loss_rec)    
    evaluation_fake = tf.summary.scalar('fake_evaluation', fake_evaluation)
    evaluation_recons = tf.summary.scalar('recons_evaluation', recons_evaluation)
    g_sum = tf.summary.merge([loss_g_sum, loss_g_GAN_sum, loss_g_cyc_sum, real_img_sum, G_loss_rec_sum, fake_img_sum, fake_source_recons_img_sum,evaluation_fake, evaluation_recons])
    d_sum = tf.summary.merge([loss_d_sum, loss_d_GAN_sum])
    
    # ============= session =============
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    
    writer = tf.summary.FileWriter(log_dir, sess.graph)
    
    # ============= Checkpoints =============
    #Load explainer checkpoint if required
    if continue_train :
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
    
    #Load pre-trained classifier checkpoint
    class_vars = [var for var in slim.get_variables_to_restore() if 'classifier' in var.name]
    name_to_var_map_local = {var.op.name: var for var in class_vars}               
    temp_saver = tf.train.Saver(var_list=name_to_var_map_local)
    ckpt = tf.train.get_checkpoint_state(classifier_experiment_dir) 
    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
    try:
        temp_saver.restore(sess, os.path.join(classifier_experiment_dir, ckpt_name))
        print("Classifier checkpoint loaded.................")
    except:
        print("Problem in loading classifier checkpoint: " + os.path.join(classifier_experiment_dir, ckpt_name))
        sys.exit()
        
    # ============= Training =============
    counter = 1
    for e in range(1, EPOCHS+1):
        np.random.shuffle(data)
        for i in range(data.shape[0] // BATCH_SIZE):
            image_paths = data[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            img, labels = load_images_and_labels(image_paths, 'CelebA',1, attr_list)
            labels = labels.ravel()
            labels = convert_ordinal_to_binary(labels,NUMS_CLASS)
            target_labels = np.random.randint(0, high=NUMS_CLASS, size=BATCH_SIZE)
            target_labels = convert_ordinal_to_binary(target_labels,NUMS_CLASS)
            
            _, d_loss, summary_str = sess.run([D_opt, D_loss, d_sum], feed_dict={y_target:target_labels, x_source: img, train_phase: True, y_source: labels})
            writer.add_summary(summary_str, counter)
            
            if (i+1) % 5 == 0:
                _, g_loss, summary_str = sess.run([G_opt,G_loss, g_sum], feed_dict={y_target: target_labels, x_source: img, train_phase: True, y_source: labels})
                writer.add_summary(summary_str, counter)
            
            counter += 1
            if counter % 1000 == 0:
                try:
                    save_results(sess, counter)
                except:
                    print("Error in save results")
                    
                saver.save(sess, ckpt_dir + "/model%2d.ckpt" % counter)
                print(counter, i, e, g_loss, d_loss)
            
            if counter > 100000:
                saver.save(sess, ckpt_dir + "/model%2d.ckpt" % counter)
                sys.exit()
            

def save_results(sess,step,image_paths):                
    img, labels = load_images_and_labels(image_paths, 'CelebA',1, attr_list)               
    labels = labels.ravel()
    labels = convert_ordinal_to_binary(labels,NUMS_CLASS)
    target_labels = np.asarray([0,1,2,3,4,5,6,7,8,0])
    target_labels = convert_ordinal_to_binary(target_labels,NUMS_CLASS)
    output_fake_img = []
    output_fake_img_d = []
    for index in range(NUMS_CLASS):
        source_img = []
        for index_j in range(NUMS_CLASS):
            source_img.append(img[index])
        source_img = np.asarray(source_img) #[n, 128, 128, 3]

        FAKE_IMG, fake_logits_ = sess.run([fake_target_img, fake_target_logits], feed_dict={y_target: target_labels,  x_source: source_img, train_phase: False})
        output_fake_img.append(FAKE_IMG)
        output_fake_img_d.append(fake_logits_)

    # save samples
    sample_file = os.path.join(sample_dir, '%06d_1.jpg'%(step))
    save_images(output_fake_img[0], output_fake_img[1], output_fake_img[2], output_fake_img[3], sample_file)
    sample_file = os.path.join(sample_dir, '%06d_2.jpg'%(step))
    save_images(output_fake_img[4], output_fake_img[5], output_fake_img[6], output_fake_img[7], sample_file)

    np.save(sample_file.split('.jpg')[0] + '_d_fake.npy' , np.asarray(output_fake_img_d))
    np.save(sample_file.split('.jpg')[0] + '_y.npy' , labels)
                
                
if __name__ == "__main__":
    main()
