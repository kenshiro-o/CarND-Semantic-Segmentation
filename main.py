#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
from tqdm import tqdm


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """    
    #   Use tf.saved_model.loader.load to load the model and weights    
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)    

    tensor_names = [vgg_input_tensor_name, vgg_keep_prob_tensor_name, 
                    vgg_layer3_out_tensor_name, vgg_layer4_out_tensor_name,
                    vgg_layer7_out_tensor_name]
    fn = tf.get_default_graph().get_tensor_by_name
    tensors = tuple(map(fn, tensor_names))
        
    return tensors[0], tensors[1], tensors[2], tensors[3], tensors[4]
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes, training=False):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # First finish the encoder
    fc8_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding="SAME", 
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    
    # fc8_1x1_bn = tf.layers.batch_normalization(fc8_1x1, training=training)


    pool4_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding="SAME", 
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))


    pool3_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding="SAME", 
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))                                
    
    
    # pool3_1x1_bn = tf.layers.batch_normalization(pool3_1x1, training=training)                                 
    # pool4_1x1_bn = tf.layers.batch_normalization(pool4_1x1, training=training)                                 


    tr_conv_fc8_1x1_2x = tf.layers.conv2d_transpose(fc8_1x1, num_classes, 4, strides=2, padding="SAME",
                                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    # tr_conv_fc8_1x1_2x_bn = tf.layers.batch_normalization(tr_conv_fc8_1x1_2x, training=training)

    pool4_1x_add_tr_conv_fc8_1x1_2x = tf.add(pool4_1x1, tr_conv_fc8_1x1_2x)
    
    # fcn_16x_final = tf.layers.conv2d_transpose(pool4_1x_add_tr_conv_fc8_1x1_2x, num_classes, 4, strides=16, padding="SAME",
    #                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    intermediate_fcn16x = tf.layers.conv2d_transpose(pool4_1x_add_tr_conv_fc8_1x1_2x, num_classes, 4, strides=2, padding="SAME",
                                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # intermediate_fcn16x_bn = tf.layers.batch_normalization(intermediate_fcn16x, training=training)                                                                                     

    pool3_1x1_add_intermedia_fcn16x = tf.add(pool3_1x1, intermediate_fcn16x)                                                     

    fcn8x_final = tf.layers.conv2d_transpose(pool3_1x1_add_intermedia_fcn16x, num_classes, 16, strides=8, padding="SAME",
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    
    return fcn8x_final
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """    

    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))

    beta = 0.001
    regularizer_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
    
    # Don't forget to add the L2 loss to our cross entropy loss
    final_loss = tf.reduce_mean(cross_entropy_loss + sum(regularizer_losses) * beta)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_operation = optimizer.minimize(final_loss)

    
    return logits, training_operation, final_loss
tests.test_optimize(optimize)



def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, dropout_keep_value=1.0, train_phase=None, learning_rate_value=0.0001):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """        
    # Never forget to initialise variables
    sess.run(tf.global_variables_initializer())

    for i in  tqdm(range(epochs)):
        total_loss = 0.0
        num_images = 0
        batch = get_batches_fn(batch_size)
        for X_train, Y_train in batch:
            fdict = {input_image: X_train, correct_label: Y_train, keep_prob: dropout_keep_value, learning_rate: learning_rate_value}
            if train_phase is not None:                
                fdict[train_phase] = True
            
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict=fdict)
            
            # Compute weighted loss per batch
            batch_size = len(X_train)
            num_images += batch_size
            total_loss += loss * batch_size
            print("[EPOCH {}] Final loss on batch = {}".format(i, loss))
                        
        avg_loss = total_loss / num_images            
        print("***** EPOCH {}: Average loss = {} *****".format(i, avg_loss))

tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer3_out, layer4_out,  layer7_out = load_vgg(sess, vgg_path)        

        is_training = tf.placeholder(tf.bool, name="is_training")
        
        last_layer = layers(layer3_out, layer4_out, layer7_out, num_classes, training=is_training)

        for v in tf.trainable_variables():
            # Print the trainable vbariables of the network
            print(v)        
        
        # Create placeholders for output
        y = tf.placeholder(tf.int32, shape=[None, None, None, num_classes])
        l_rate = tf.placeholder(tf.float32, name="learning_rate")
        logits, train_ops, ce_loss = optimize(last_layer, y, l_rate, num_classes)

        
        # Train
        train_nn(sess, 75, 30, get_batches_fn, train_ops, ce_loss, input_image, y, keep_prob, l_rate, 
                 dropout_keep_value=0.65, train_phase=is_training, learning_rate_value=0.001)

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':    
    run()
