import sys

import tensorflow as tf
from model import Model, Logger

def main(_):

    # create log file
    open('logfile.txt', 'a').close()

    config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=False)

    with tf.Session(config=config) as session:
	
        # Set Logger Output File
        sys.stdout = Logger(output_file='logfile.txt')

        model = Model(session)

        print('\n\t Start Training')
        model.train()

if __name__ == '__main__':
    tf.app.run()