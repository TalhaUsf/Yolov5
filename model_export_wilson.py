# %%
from codecs import ignore_errors
import tensorflow as tf
from rich.console import Console
from keras_flops import get_flops

# %%
detection_model = tf.keras.models.load_model(
    filepath="/home/fsuser/AI_ENGINE/yolov5_tf_original/Yolov5/Ibrahim_Bhaai_model/fdet-d0", custom_objects=None, compile=True, options=None
)

# full model
Console().print(detection_model.summary())


# %%

class Scaling(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()
        # Create a non-trainable weight.
        self.scaling = tf.keras.layers.Rescaling(scale=1./127.5, offset=-1)

    def call(self, inputs):

        return self.scaling(inputs)

# FIXME this might needs removal
class Resizing(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()
        # Create a non-trainable weight.
        self.resizing = tf.keras.layers.Resizing(
            height=512,
            width=512,
            interpolation='bicubic',
            crop_to_aspect_ratio=False,
        )

    def call(self, inputs):
        return self.resizing(inputs)


class AnchorBox:
    """Generates anchor boxes.

    This class has operations to generate anchor boxes for feature maps at
    strides `[8, 16, 32, 64, 128]`. Where each anchor each box is of the
    format `[x, y, width, height]`.

    Attributes:
      aspect_ratios: A list of float values representing the aspect ratios of
        the anchor boxes at each location on the feature map
      scales: A list of float values representing the scale of the anchor boxes
        at each location on the feature map.
      num_anchors: The number of anchor boxes at each location on feature map
      areas: A list of float values representing the areas of the anchor
        boxes for each feature map in the feature pyramid.
      strides: A list of float value representing the strides for each feature
        map in the feature pyramid.
    """

    def __init__(self):
        self.aspect_ratios = [0.5, 1.0, 2.0]
        self.scales = [2 ** x for x in [0, 1 / 3, 2 / 3]]

        self._num_anchors = len(self.aspect_ratios) * len(self.scales)
        self._strides = [2 ** i for i in range(3, 8)]
        self._areas = [x ** 2 for x in [32.0, 64.0, 128.0, 256.0, 512.0]]
        self._anchor_dims = self._compute_dims()

    def _compute_dims(self):
        """Computes anchor box dimensions for all ratios and scales at all levels
        of the feature pyramid.
        """
        anchor_dims_all = []
        for area in self._areas:
            anchor_dims = []
            for ratio in self.aspect_ratios:
                anchor_height = tf.math.sqrt(area / ratio)
                anchor_width = area / anchor_height
                dims = tf.reshape(
                    tf.stack([anchor_width, anchor_height], axis=-1), [1, 1, 2]
                )
                for scale in self.scales:
                    anchor_dims.append(scale * dims)
            anchor_dims_all.append(tf.stack(anchor_dims, axis=-2))
        return anchor_dims_all

    def _get_anchors(self, feature_height, feature_width, level):
        """Generates anchor boxes for a given feature map size and level

        Arguments:
          feature_height: An integer representing the height of the feature map.
          feature_width: An integer representing the width of the feature map.
          level: An integer representing the level of the feature map in the
            feature pyramid.

        Returns:
          anchor boxes with the shape
          `(feature_height * feature_width * num_anchors, 4)`
        """
        rx = tf.range(feature_width, dtype=tf.float32) + 0.5
        ry = tf.range(feature_height, dtype=tf.float32) + 0.5
        centers = tf.stack(tf.meshgrid(rx, ry), axis=-1) * \
            self._strides[level - 3]
        centers = tf.expand_dims(centers, axis=-2)
        centers = tf.tile(centers, [1, 1, self._num_anchors, 1])
        dims = tf.tile(
            self._anchor_dims[level - 3], [feature_height, feature_width, 1, 1]
        )
        anchors = tf.concat([centers, dims], axis=-1)
        return tf.reshape(
            anchors, [feature_height * feature_width * self._num_anchors, 4]
        )

    def get_anchors(self, image_height, image_width):
        """Generates anchor boxes for all the feature maps of the feature pyramid.

        Arguments:
          image_height: Height of the input image.
          image_width: Width of the input image.

        Returns:
          anchor boxes for all the feature maps, stacked as a single tensor
            with shape `(total_anchors, 4)`
        """
        anchors = [
            self._get_anchors(
                tf.math.ceil(image_height / 2 ** i),
                tf.math.ceil(image_width / 2 ** i),
                i,
            )
            for i in range(3, 8)
        ]
        return tf.concat(anchors, axis=0)


def convert_to_corners(boxes):
    """Changes the box format to corner coordinates

    Arguments:
      boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[x, y, width, height]`.

    Returns:
      converted boxes with shape same as that of boxes.
    """
    return tf.concat(
        [boxes[..., :2] - boxes[..., 2:] / 2.0,
            boxes[..., :2] + boxes[..., 2:] / 2.0],
        axis=-1,
    )


class DecodePredictions(tf.keras.layers.Layer):
    """A Keras layer that decodes predictions of the RetinaNet model.

    Attributes:
      num_classes: Number of classes in the dataset
      confidence_threshold: Minimum class probability, below which detections
        are pruned.
      nms_iou_threshold: IOU threshold for the NMS operation
      max_detections_per_class: Maximum number of detections to retain per
       class.
      max_detections: Maximum number of detections to retain across all
        classes.
      box_variance: The scaling factors used to scale the bounding box
        predictions.
    """

    def __init__(
        self,
        num_classes=80,
        confidence_threshold=0.05,
        nms_iou_threshold=0.5,
        max_detections_per_class=100,
        max_detections=100,
        box_variance=[0.1, 0.1, 0.2, 0.2],
        **kwargs
    ):
        super(DecodePredictions, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections_per_class = max_detections_per_class
        self.max_detections = max_detections

        self._anchor_box = AnchorBox()
        self._box_variance = tf.convert_to_tensor(
            [0.1, 0.1, 0.2, 0.2], dtype=tf.float32
        )

    def _decode_box_predictions(self, anchor_boxes, box_predictions):
        boxes = box_predictions * self._box_variance
        boxes = tf.concat(
            [
                boxes[:, :, :2] * anchor_boxes[:, :, 2:] +
                anchor_boxes[:, :, :2],
                tf.math.exp(boxes[:, :, 2:]) * anchor_boxes[:, :, 2:],
            ],
            axis=-1,
        )
        boxes_transformed = convert_to_corners(boxes)
        return boxes_transformed

    def call(self, images, predictions):
        image_shape = tf.cast(tf.shape(images), dtype=tf.float32)
        anchor_boxes = self._anchor_box.get_anchors(
            image_shape[1], image_shape[2])
        box_predictions = predictions[:, :, :4]
        cls_predictions = tf.nn.sigmoid(predictions[:, :, 4:])
        boxes = self._decode_box_predictions(
            anchor_boxes[None, ...], box_predictions)

        return tf.image.combined_non_max_suppression(
            tf.expand_dims(boxes, axis=2),
            cls_predictions,
            self.max_detections_per_class,
            self.max_detections,
            self.nms_iou_threshold,
            self.confidence_threshold,
            clip_boxes=False,
        )


# %%


inp = tf.keras.Input(
    shape=(None, None, 3),
    name="input",
    dtype=tf.float32,
)
x = Resizing()(inp)
x = Scaling()(x)
predictions = detection_model(x)
# modified_model = tf.keras.Model(inputs=inp, outputs=out)
detections = DecodePredictions(confidence_threshold=0.5)(inp, predictions)
modified_model = tf.keras.Model(inputs=inp, outputs=detections)


# %%

modified_model.summary()
# Calculae FLOPS
flops = get_flops(modified_model, batch_size=1)
print(f"FLOPS: {flops / 10 ** 9:.03} G")

# %%

modified_model.save(
    filepath="/home/fsuser/AI_ENGINE/yolov5_tf_original/Yolov5/Ibrahim_Bhaai_model/modified_model_nms",
    overwrite=True,
    include_optimizer=True,
    save_format='tf',
    save_traces=True
)

# %%

# 👉 convet to tflite
converter = tf.lite.TFLiteConverter.from_keras_model(modified_model)
# optimize for speed and size
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# set fp16 to true to use float16 quantization
converter.target_spec.supported_types = [tf.float16]
# set opset to allow for extended ops
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
    # tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    # tf.lite.OpsSet.TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
]
# convert to tflite
tflite_model = converter.convert()
from pathlib import Path

Path("/home/fsuser/AI_ENGINE/yolov5_tf_original/Yolov5/Ibrahim_Bhaai_model/tflite_model").joinpath("model_fp16.tflite").write_bytes(tflite_model)


# %%
