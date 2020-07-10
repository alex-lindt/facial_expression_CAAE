"""
NETWORK CONFIG FILE
"""

# training data directory
training_data_path = "./data/train/"

# validation data directory
validation_data_path = "./data/train/"

# path to pre-trained vgg face model weights
vgg_face_path = './utils/vgg-face.mat'

# batch size
size_batch=49	

# size of hidden vector z
num_z_channels=50 	

# width and height of input image
size_image = 96

# path to save checkpoints, samples, and summary
save_dir='./save'  	

# value range of single pixels in an input image
image_value_range = (-1, 1) 

