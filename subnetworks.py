"""
Implementations of the subnetworks of the CAAE:

- Encoder
- Generator
- Discriminator_Img
- Discriminator_Z
"""
import tensorflow as tf
import numpy as np
from layers import dense, conv2d, deconv2d, batch_norm
from config import size_batch, num_z_channels


# --HELPERS ---------------------------------------
# -------------------------------------------------
def lrelu(inp, leak=0.2):
    """
    Leaky Rectified Linear Unit (ReLu) activation.
    
    @param inp: input tensor
    
    @return: tensor of same size as input tensor
    """
    return tf.maximum(inp, leak*inp)


def concat_label(tensor, label, duplicate=1):
    """
    Duplicates label and concatenates it to tensor.
    
    @param tensor: input tensor 
                   (1) of size [batch_size, length]
                   (2) of size [batch_size, x, x, length]
    @param label: input tensor of size [batch_size, label_length]
    
    @return: (1) tensor of size [batch_size, length+duplicate*label_length]
             (2) tensor of size [batch_size, x, x, length+duplicate*label_length]
    """ 
    # duplicate the label to enhance its effect
    label = tf.tile(label, [1, duplicate])
    
    # get shapes of label and tensor
    tensor_shape = tensor.get_shape().as_list()    
    label_shape = label.get_shape().as_list()
    
    # CASE (1)
    if len(tensor_shape) == 2: return tf.concat([tensor, label], 1)
    
    # CASE (2)
    if len(tensor_shape) == 4:
        # reshape label to [batch_size, 1, 1, duplicate*label_length]
        label = tf.reshape(label, [tensor_shape[0], 1, 1, label_shape[-1]])
        # scale label to [batch_size, x, x, duplicate*label_length]
        label = label*tf.ones([tensor_shape[0], tensor_shape[1], tensor_shape[2], label_shape[-1]])
        # concatenate label and tensor
        return tf.concat([tensor, label], 3)


# --NETWORKS --------------------------------------
# -------------------------------------------------

def generator(z, valence, arousal, reuse_variables=False):
    """
    Creates generator network.
    
    @param z: tensor of size config.num_z_channels
    @param valence: tensor of size 1
    @param arousal: tensor of size 1
    
    @return: tensor of size 96x96x3
    """
    if reuse_variables:
        tf.get_variable_scope().reuse_variables()

    with tf.variable_scope("generator") as scope:

        # duplicate valence/arousal label and concatenate to z
        z = concat_label(z, valence, duplicate=num_z_channels)
        z = concat_label(z, arousal, duplicate=num_z_channels)

        # -- fc layer
        name = 'G_fc'
        current = dense(z, 1024*6*6, reuse=reuse_variables)
        # reshape
        current = tf.reshape(current, [-1, 6, 6, 1024])
        current = tf.nn.relu(current)

        # -- transposed convolutional layer 1-4
        for index, num_filters in enumerate([512, 256, 128, 64]):
            name = 'G_deconv' + str(index+1)
            current = deconv2d(current, num_filters, name=name, reuse=reuse_variables)
            current = tf.nn.relu(current)

        # -- transposed convolutional layer 5+6
        current = deconv2d(current, 32, stride=1, name='G_deconv5', reuse=reuse_variables)
        current = tf.nn.relu(current)

        current = deconv2d(current, 3, stride=1,  name='G_deconv6', reuse=reuse_variables)    
        return tf.nn.tanh(current)
        
def encoder(current, reuse_variables=False):
    """
    Creates encoder network.
    
    @param current: tensor of size 96x96x3
    
    @return: tensor of size config.num_z_channels
    """
    if reuse_variables:
        tf.get_variable_scope().reuse_variables()

    with tf.variable_scope("encoder") as scope:
        
        # -- transposed convolutional layer 1-4
        for index, num_filters in enumerate([64,128,256,512]):
            name = 'E_conv' + str(index)
            current = conv2d(current, num_filters, name=name, reuse=reuse_variables)
            current = tf.nn.relu(current)
             
        # reshape
        current = tf.reshape(current, [size_batch, -1])

        # -- fc layer
        name = 'E_fc'
        current = dense(current, num_z_channels, name=name, reuse=reuse_variables)
        return tf.nn.tanh(current)
    

def discriminator_img(current, valence, arousal, reuse_variables=False):
    """
    Creates discriminator network on generated image + desired emotion.

    @param current: tensor of size 96x96x3
    @param valence: tensor of size 1
    @param arousal: tensor of size 1

    @return:  sigmoid(output), output
              (output tensor is of size 1)
    """
    if reuse_variables:
        tf.get_variable_scope().reuse_variables()

    with tf.variable_scope("discriminator_img") as scope:

        # -- convolutional blocks (= convolution+batch_norm+relu) 1-4
        for index, num_filters in enumerate([16, 32, 64, 128]):

            # convolution
            name = 'D_img_conv' + str(index+1)
            current = conv2d(current, num_filters, name=name, reuse=reuse_variables)

            # batch normalization
            name = 'D_img_bn' + str(index+1)
            current = batch_norm(current, name, reuse=reuse_variables)
            # relu activation
            current = tf.nn.relu(current)

            if index==0:
                current = concat_label(current, valence, 16)
                current = concat_label(current, arousal, 16)

        # reshape
        current = tf.reshape(current, [size_batch, -1])

        # -- fc 1
        name = 'D_img_fc1'
        current = lrelu(dense(current,1024, name=name, reuse=reuse_variables))

        # -- fc 2
        name = 'D_img_fc2'
        current = dense(current,1, name=name, reuse=reuse_variables)
        return tf.nn.sigmoid(current), current


def discriminator_z(current, reuse_variables=False):
    if reuse_variables:
        tf.get_variable_scope().reuse_variables()

    with tf.variable_scope("discriminator_z") as scope:

        #  FC blocks 1-3 (full connection+batch_nom+relu)
        for index, num_filters in enumerate([64,32,16]):
            name = 'D_z_fc' + str(index+1)          
            current = dense(current, num_filters, name=name)
            # batch normalization
            name = 'D_z_bn' + str(index+1)
            current = batch_norm(current, name, reuse=reuse_variables)
            # relu activation
            current = tf.nn.relu(current)
            
        # FC block 4
        name = 'D_z_fc' + str(index+2)
        current = dense(current, 1, name=name)
        return tf.nn.sigmoid(current), current