import os
import os.path as path

import keras
from keras.models import load_model
from keras import backend as K
K.set_learning_phase(0)
from keras.applications import VGG16, ResNet50, VGG19, InceptionResNetV2, DenseNet201, Xception, InceptionV3

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

def export_model(saver, model, input_node_names, output_node_name):
    tf.train.write_graph(K.get_session().graph_def, 'out', \
        MODEL_NAME + '_graph.pbtxt')

    saver.save(K.get_session(), 'out/' + MODEL_NAME + '.chkp')

    freeze_graph.freeze_graph('out/' + MODEL_NAME + '_graph.pbtxt', None, \
        False, 'out/' + MODEL_NAME + '.chkp', output_node_name, \
        "save/restore_all", "save/Const:0", \
        'out/frozen_' + MODEL_NAME + '.pb', True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_' + MODEL_NAME + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, input_node_names, [output_node_name],
            tf.float32.as_datatype_enum)

    # for node in gd.node:
    #     if node.op == "Switch":
    #         node.op = "Identity"
    #         del node.input[1]

    with tf.gfile.FastGFile('out/opt_' + MODEL_NAME + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("graph saved!")


MODEL_NAME = 'VGG16_animal_ver1'
if not path.exists('out'):
    os.mkdir('out')

model = load_model('VGG16_ver1.h5')
model.summary()
[node.op.name for node in model.inputs] # ['input_1'], ['image_input'],  ['conv2d_1_input']
[node.op.name for node in model.outputs][0] # 'dense_2/Softmax','dense_3/Softmax'


# model = VGG16(weights='imagenet')
# VGG16_.summary()
export_model(tf.train.Saver(), model, [node.op.name for node in model.inputs], [node.op.name for node in model.outputs][0])
