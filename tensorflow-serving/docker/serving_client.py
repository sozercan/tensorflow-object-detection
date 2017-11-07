#!/usr/bin/env python2.7

"""Send JPEG image to tensorflow_model_server
"""

from __future__ import print_function
from grpc.beta import implementations
import tensorflow as tf
import numpy as np
import os
from io import BytesIO
import requests
from PIL import Image
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util

tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image', '', 'url to image in JPEG format')
tf.app.flags.DEFINE_string('label_map_path', './pascal_label_map.pbtxt', 'path to label map path')
tf.app.flags.DEFINE_string('save_path', './', 'save path for output image')
tf.app.flags.DEFINE_string('model_name', 'serving', 'model name')
tf.app.flags.DEFINE_string('signature_name', 'serving_default', 'signature name')
tf.app.flags.DEFINE_string('num_classes', '1', 'num classes')
FLAGS = tf.app.flags.FLAGS

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def main(_):
  host, port = FLAGS.server.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

  # Send request
  response = requests.get(FLAGS.image, stream=True)

  if response.status_code == 200:
    request = predict_pb2.PredictRequest()
    request.model_spec.name = FLAGS.model_name
    request.model_spec.signature_name = FLAGS.signature_name
    request.inputs['inputs'].CopyFrom(
        tf.contrib.util.make_tensor_proto(response.content, shape=[1]))
    result = stub.Predict(request, 10.0)  # 10 secs timeout

    image = Image.open(BytesIO(response.content))
    image_np = load_image_into_numpy_array(image)
    boxes = np.array(result.outputs['detection_boxes'].float_val).reshape(
        result.outputs['detection_boxes'].tensor_shape.dim[0].size,
        result.outputs['detection_boxes'].tensor_shape.dim[1].size,
        result.outputs['detection_boxes'].tensor_shape.dim[2].size
    )
    classes = np.array(result.outputs['detection_classes'].float_val)
    scores = np.array(result.outputs['detection_scores'].float_val)

    label_map = label_map_util.load_labelmap(FLAGS.label_map_path)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=FLAGS.num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)
    vis_util.save_image_array_as_png(image_np, FLAGS.save_path+"/output-"+FLAGS.image.split('/')[-1])

if __name__ == '__main__':
  tf.app.run()
