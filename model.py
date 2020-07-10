"""
Implementation of the model.

Parts of the code are inherited from the official CAAE implementation (https://arxiv.org/abs/1702.08423).
"""

import os
import sys
import time
from glob import glob

import numpy as np
import tensorflow as tf
from scipy.io import loadmat, savemat

from config import *
from image_ops import *
from subnetworks import encoder, generator, discriminator_img, discriminator_z
from vgg_face import face_embedding

class Model(object):
    """
    Implementation of the model used.
    """
    def __init__(self, session):        
        self.session = session
        self.vgg_weights = loadmat(vgg_face_path)
        
        # -- INPUT PLACEHOLDERS -----------------------------------------------------------
        # ---------------------------------------------------------------------------------
        self.input_image = tf.placeholder(
            tf.float32,
            [size_batch, size_image, size_image, 3],
            name='input_images'
        )

        self.valence = tf.placeholder(
            tf.float32,
            [size_batch, 1],
            name='valence_labels'
        )
        
        self.arousal = tf.placeholder(
            tf.float32,
            [size_batch, 1],
            name='arousal_labels'
        )

        self.z_prior = tf.placeholder(
            tf.float32,
            [size_batch, num_z_channels],
            name='z_prior'
        )
        
        
        # -- GRAPH ------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------
        print ('\n\t SETTING  UP THE GRAPH')

        with tf.variable_scope(tf.get_variable_scope()):
            with tf.device('/device:GPU:0'): 
                
                # -- NETWORKS -------------------------------------------------------------
                # -------------------------------------------------------------------------
                
                # encoder:  
                self.z = encoder(self.input_image)

                # generator: z + arousal + valence --> generated image   
                self.G = generator(self.z, 
                                   valence=self.valence, 
                                   arousal=self.arousal)

                # discriminator on z
                self.D_z, self.D_z_logits = discriminator_z(self.z)

                # discriminator on G
                self.D_G, self.D_G_logits = discriminator_img(self.G, 
                                                              valence=self.valence, 
                                                              arousal=self.arousal)

                # discriminator on z_prior
                self.D_z_prior, self.D_z_prior_logits = discriminator_z(self.z_prior,
                                                                        reuse_variables=True)

                # discriminator on input image
                self.D_input, self.D_input_logits = discriminator_img(self.input_image,
                                                                      valence=self.valence,
                                                                      arousal=self.arousal,
                                                                      reuse_variables=True)
                
                # -- LOSSES ---------------------------------------------------------------
                # -------------------------------------------------------------------------
                
                # ---- VGG LOSS --------------------------------------------------------- 
                real_conv1_2, real_conv2_2, real_conv3_2, real_conv4_2, real_conv5_2 = face_embedding(self.vgg_weights, self.input_image[:16])
                fake_conv1_2, fake_conv2_2, fake_conv3_2, fake_conv4_2, fake_conv5_2 = face_embedding(self.vgg_weights, self.G[:16])

                conv1_2_loss = tf.reduce_mean(tf.abs(real_conv1_2 - fake_conv1_2)) / 224. / 224.
                conv2_2_loss = tf.reduce_mean(tf.abs(real_conv2_2 - fake_conv2_2)) / 112. / 112.
                conv3_2_loss = tf.reduce_mean(tf.abs(real_conv3_2 - fake_conv3_2)) / 56. / 56.
                conv4_2_loss = tf.reduce_mean(tf.abs(real_conv4_2 - fake_conv4_2)) / 28. / 28.
                conv5_2_loss = tf.reduce_mean(tf.abs(real_conv5_2 - fake_conv5_2)) / 14. / 14.
                self.vgg_loss = conv1_2_loss + conv2_2_loss + conv3_2_loss + conv4_2_loss + conv5_2_loss
                # -----------------------------------------------------------------------
            
            # reconstruction loss of encoder+generator
            self.EG_loss = tf.reduce_mean(tf.abs(self.input_image - self.G))  # L1 loss

            # loss function of discriminator on z
            self.D_z_loss_prior = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_z_prior_logits, labels=tf.ones_like(self.D_z_prior_logits))
            )
            self.D_z_loss_z = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_z_logits, labels=tf.zeros_like(self.D_z_logits))
            )
            self.E_z_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_z_logits, labels=tf.ones_like(self.D_z_logits))
            )
            # loss function of discriminator on image
            self.D_img_loss_input = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_input_logits, labels=tf.ones_like(self.D_input_logits))
            )
            self.D_img_loss_G = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_G_logits, labels=tf.zeros_like(self.D_G_logits))
            )
            self.G_img_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_G_logits, labels=tf.ones_like(self.D_G_logits))
            )
            
            # -- TRAINABLE VARIABLES ----------------------------------------------------------
            # ---------------------------------------------------------------------------------
            trainable_variables = tf.trainable_variables()
            # variables of encoder
            self.E_variables = [var for var in trainable_variables if 'E_' in var.name]
            # variables of generator
            self.G_variables = [var for var in trainable_variables if 'G_' in var.name]
            # variables of discriminator on z
            self.D_z_variables = [var for var in trainable_variables if 'D_z_' in var.name]
            # variables of discriminator on image
            self.D_img_variables = [var for var in trainable_variables if 'D_img_' in var.name]


            # -- SUMMARY ----------------------------------------------------------------------
            # ---------------------------------------------------------------------------------
            with tf.device('/device:CPU:0'):
                self.z_summary = tf.summary.histogram('z', self.z)
                self.z_prior_summary = tf.summary.histogram('z_prior', self.z_prior)
                self.EG_loss_summary = tf.summary.scalar('EG_loss', self.EG_loss)
                self.D_z_loss_z_summary = tf.summary.scalar('D_z_loss_z', self.D_z_loss_z)
                self.D_z_loss_prior_summary = tf.summary.scalar('D_z_loss_prior', self.D_z_loss_prior)
                self.E_z_loss_summary = tf.summary.scalar('E_z_loss', self.E_z_loss)
                self.D_z_logits_summary = tf.summary.histogram('D_z_logits', self.D_z_logits)
                self.D_z_prior_logits_summary = tf.summary.histogram('D_z_prior_logits', self.D_z_prior_logits)
                self.D_img_loss_input_summary = tf.summary.scalar('D_img_loss_input', self.D_img_loss_input)
                self.D_img_loss_G_summary = tf.summary.scalar('D_img_loss_G', self.D_img_loss_G)
                self.G_img_loss_summary = tf.summary.scalar('G_img_loss', self.G_img_loss)
                self.D_G_logits_summary = tf.summary.histogram('D_G_logits', self.D_G_logits)
                self.D_input_logits_summary = tf.summary.histogram('D_input_logits', self.D_input_logits)
                self.vgg_loss_summary = tf.summary.scalar('VGG_loss', self.vgg_loss)

                # for saving the graph and variables
                self.saver = tf.train.Saver(max_to_keep=10)
        
    def train(self,
              num_epochs=2,  # number of epochs
              learning_rate=0.0002,  # learning rate of optimizer
              beta1=0.5,  # parameter for Adam optimizer
              decay_rate=1.0,  # learning rate decay (0, 1], 1 means no decay
              use_trained_model=False,  # used the saved checkpoint to initialize the model
              ):
        
        # set learning rate decay
        with tf.variable_scope(tf.get_variable_scope()):
            with tf.device('/device:CPU:0'):
                self.EG_global_step = tf.Variable(0, trainable=False, name='global_step')

        
        # -- LOAD FILE NAMES --------------------------------------------------------------
        # ---------------------------------------------------------------------------------
        # ---- TRAINING DATA
        path = training_data_path
        file_names = [path + x for x in os.listdir(path) if not int(x.split('s')[2])/1000 <-1]
        file_names = self.fill_up_equally(file_names)
        size_data = len(file_names)
        np.random.shuffle(file_names)
        # ---- VALIDATION DATA
        val_path = validation_data_path
        self.validation_files = [val_path + v for v in os.listdir(val_path) if not int(v.split('s')[2])/1000 <-1]
        
        # -- LOSS FUNCTIONS ---------------------------------------------------------------
        # ---------------------------------------------------------------------------------
        self.loss_EG = self.EG_loss + self.vgg_loss/3 +  0.01 * self.G_img_loss + 0.01 * self.E_z_loss 
        self.loss_Di = self.D_img_loss_input + self.D_img_loss_G        
        self.loss_Di = self.D_img_loss_input + self.D_img_loss_G
        
        
        # -- OPTIMIZERS -------------------------------------------------------------------
        # ---------------------------------------------------------------------------------
        with tf.device('/device:GPU:0'): 
            
            EG_learning_rate = tf.train.exponential_decay(
                learning_rate=learning_rate,
                global_step=self.EG_global_step,
                decay_steps=size_data / size_batch * 2,
                decay_rate=decay_rate,
                staircase=True
            )

            # optimizer for encoder + generator
            self.EG_optimizer = tf.train.AdamOptimizer(
                learning_rate=EG_learning_rate,
                beta1=beta1
            ).minimize(
                loss=self.loss_EG,
                global_step=self.EG_global_step,
                var_list=self.E_variables + self.G_variables
            )

            # optimizer for discriminator on z
            self.D_z_optimizer = tf.train.AdamOptimizer(
                learning_rate=EG_learning_rate,
                beta1=beta1
            ).minimize(
                loss=self.loss_Dz,
                var_list=self.D_z_variables
            )

            # optimizer for discriminator on image
            self.D_img_optimizer = tf.train.AdamOptimizer(
                learning_rate=EG_learning_rate,
                beta1=beta1
            ).minimize(
                loss=self.loss_Di,
                var_list=self.D_img_variables
            )
        

        # -- TENSORBOARD SUMMARY ----------------------------------------------------------
        # ---------------------------------------------------------------------------------        
        with tf.device('/device:CPU:0'):
            self.EG_learning_rate_summary = tf.summary.scalar('EG_learning_rate', EG_learning_rate)
            self.summary = tf.summary.merge([
                self.z_summary, self.z_prior_summary,
                self.D_z_loss_z_summary, self.D_z_loss_prior_summary,
                self.D_z_logits_summary, self.D_z_prior_logits_summary,
                self.EG_loss_summary, self.E_z_loss_summary,
                self.D_img_loss_input_summary, self.D_img_loss_G_summary,
                self.G_img_loss_summary, self.EG_learning_rate_summary,
                self.D_G_logits_summary, self.D_input_logits_summary,
                self.vgg_loss_summary
            ])
            self.writer = tf.summary.FileWriter(os.path.join(save_dir, 'summary'), self.session.graph)
        
        

        # ************* get some random samples as testing data to visualize the learning process *********************
        sample_files = file_names[0:size_batch]

        file_names[0:size_batch] = []

        sample = [load_image(
            image_path=sample_file,
            image_size=size_image,
            image_value_range=image_value_range,
            is_gray=False,
        ) for sample_file in sample_files]

        sample_images = np.array(sample).astype(np.float32)

        sample_label_valence = np.asarray([[int(x.split('s')[2]) / 1000] for x in sample_files])
        sample_label_arousal = np.asarray([[int(x.split('s')[3][:-4]) / 1000] for x in sample_files])


        # ******************************************* training *******************************************************
        print('\n\tPreparing for training ...')

        # initialize the graph
        tf.global_variables_initializer().run()

        # load check point
        if use_trained_model:
            if self.load_checkpoint():
                print("\tSUCCESS ^_^")
            else:
                print("\tFAILED >_<!")

        # epoch iteration
        num_batches = len(file_names) // size_batch
        for epoch in range(num_epochs):
            if enable_shuffle:
                np.random.shuffle(file_names)
            for ind_batch in range(num_batches):
                start_time = time.time()
                # read batch images and labels
                batch_files = file_names[ind_batch*size_batch:(ind_batch+1)*size_batch]
                batch = [load_image(
                    image_path=batch_file,
                    image_size=size_image,
                    image_value_range=image_value_range,
                    is_gray=False,
                ) for batch_file in batch_files]

                batch_images = np.array(batch).astype(np.float32)

                batch_label_valence =np.asarray([[int(x.split('s')[2]) / 1000] for x in batch_files])
                batch_label_arousal = np.asarray([[int(x.split('s')[3][:-4]) / 1000] for x in batch_files])

                # prior distribution on the prior of z
                batch_z_prior = np.random.uniform(
                    image_value_range[0],
                    image_value_range[-1],
                    [size_batch, num_z_channels]
                ).astype(np.float32)

                # update
                _, _, _, EG_err, Ez_err, Dz_err, Dzp_err, Gi_err, DiG_err, Di_err, TV, vgg = self.session.run(
                    fetches = [
                        self.EG_optimizer,
                        self.D_z_optimizer,
                        self.D_img_optimizer,
                        self.EG_loss,
                        self.E_z_loss,
                        self.D_z_loss_z,
                        self.D_z_loss_prior,
                        self.G_img_loss,
                        self.D_img_loss_G,
                        self.D_img_loss_input,
                        self.tv_loss,
                        self.vgg_loss
                    ],
                    feed_dict={
                        self.input_image: batch_images,
                        self.valence: batch_label_valence,
                        self.arousal: batch_label_arousal,
                        self.z_prior: batch_z_prior
                    }
                )
                print("\nEpoch: [%3d/%3d] Batch: [%3d/%3d]\n\tEG_err=%.4f\tVGG=%.4f" %
                    (epoch+1, num_epochs, ind_batch+1, num_batches, EG_err, vgg))
                print("\tEz=%.4f\tDz=%.4f\tDzp=%.4f" % (Ez_err, Dz_err, Dzp_err))
                print("\tGi=%.4f\tDi=%.4f\tDiG=%.4f" % (Gi_err, Di_err, DiG_err))

                # estimate left run time
                elapse = time.time() - start_time
                time_left = ((num_epochs - epoch - 1) * num_batches + (num_batches - ind_batch - 1)) * elapse
                print("\tTime left: %02d:%02d:%02d" %
                      (int(time_left / 3600), int(time_left % 3600 / 60), time_left % 60))


                # add to summary
                summary = self.summary.eval(
                    feed_dict={
                        self.input_image: batch_images,
                        self.valence: batch_label_valence,
                        self.arousal: batch_label_arousal,
                        self.z_prior: batch_z_prior
                    }
                )
                self.writer.add_summary(summary, self.EG_global_step.eval())

                if ind_batch%500 == 0:
                    # save sample images for each epoch
                    name = '{:02d}_{:02d}'.format(epoch+1, ind_batch)
                    self.sample(sample_images, sample_label_valence, sample_label_arousal, name+'.png')
                    # TEST
                    test_dir = os.path.join(save_dir, 'test')
                    if not os.path.exists(test_dir):
                        os.makedirs(test_dir)
                    self.test(sample_images, test_dir, name+'.png')

            # save checkpoint for each epoch
            # VALIDATE
            name = '{:02d}_model'.format(epoch+1)
            self.validate(name)
            self.save_checkpoint(name=name)

        # save the trained model
        #self.save_checkpoint()
        # close the summary writer
        #self.writer.close()

    def save_checkpoint(self, name=''):
        checkpoint_dir = os.path.join(save_dir, 'checkpoint')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(
            sess=self.session,
            save_path=os.path.join(checkpoint_dir, name)
        )

    def load_checkpoint(self):
        print("\n\tLoading pre-trained model ...")
        checkpoint_dir = os.path.join(save_dir, 'checkpoint')
        checkpoints = tf.train.get_checkpoint_state(checkpoint_dir)
        if checkpoints and checkpoints.model_checkpoint_path:
            checkpoints_name = os.path.basename(checkpoints.model_checkpoint_path)
            self.saver.restore(self.session, os.path.join(checkpoint_dir, checkpoints_name))
            return True
        else:
            return False

    def sample(self, images, valence, arousal, name):
        sample_dir = os.path.join(save_dir, 'samples')
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        z, G = self.session.run(
            [self.z, self.G],
            feed_dict={
                self.input_image: images,
                self.valence: valence,
                self.arousal: arousal
            }
        )
        size_frame = int(np.sqrt(size_batch))
        save_batch_images(
            batch_images=G,
            save_path=os.path.join(sample_dir, name),
            image_value_range=image_value_range,
            size_frame=[size_frame, size_frame]
        )

        save_batch_images(
            batch_images=images,
            save_path=os.path.join(sample_dir, "input.png"),
            image_value_range=image_value_range,
            size_frame=[size_frame, size_frame]
        )

    def validate(self, name):
        # Create Validation Directory if needed
        val_dir = os.path.join(save_dir, 'validation')
        if not os.path.exists(val_dir):
            os.makedirs(val_dir)
        # Create Name Directory if needed
        name_dir = os.path.join(val_dir, name)
        if not os.path.exists(name_dir):
            os.makedirs(name_dir)

        # validate
        for image_path in self.validation_files:
            n = image_path.split("/")[3].split("s")[0]+ ".png"
            self.test(np.array([load_image(image_path, image_size=96)]), name_dir, n)

    def test(self, images, test_dir, name):
        images = images[:1, :, :, :]

        # valence
        valence = np.arange(0.75, -0.751, -0.25)
        valence = np.repeat(valence, 7).reshape((49, 1))

        # arousal
        arousal = [np.arange(0.75, -0.751, -0.25)]
        arousal = np.repeat(arousal, 7, axis=0)
        arousal = np.asarray([item for sublist in arousal for item in sublist]).reshape((49, 1))

        query_images = np.tile(images, (49, 1, 1, 1))

        z, G = self.session.run(
            [self.z, self.G],
            feed_dict={
                self.input_image: query_images,
                self.valence: valence,
                self.arousal: arousal
            }
        )

        save_output(
            input_image=images,
            output=G,
            path=os.path.join(test_dir, name),
            image_value_range = image_value_range,
            size_frame=[7, 10]
        )


    def fill_up_equally(self, X):
        sorted_samples = [[x for x in X if int(x.split('s')[1]) == r] for r in range(8)]

        amounts = [len(x) for x in sorted_samples]
        differences = [max(amounts) - a for a in amounts]

        for i, d in enumerate(differences):
            samples = sorted_samples[i]
            added = [samples[x] for x in np.random.choice(range(len(samples)), d)]
            sorted_samples[i] = sorted_samples[i] + added

        sorted_samples_flat = [item for sublist in sorted_samples for item in sublist]

        np.random.seed = 14031993
        np.random.shuffle(sorted_samples_flat)

        return sorted_samples_flat


class Logger(object):
    def __init__(self, output_file):
        self.terminal = sys.stdout
        self.log = open(output_file, "a")

    def write(self, message):
        self.terminal.write(message)
        if not self.log.closed:
            self.log.write(message)

    def close(self):
        self.log.close()

    def flush(self):
        self.close()
        # needed for python 3 compatibility
        pass