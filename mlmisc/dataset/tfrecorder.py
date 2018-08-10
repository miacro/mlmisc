import tensorflow as tf
import logging
import numpy
from .annotation import METADATA


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=None if value is None else [tf.compat.as_bytes(value)]))


def bytes_list_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=None if value is None else
            [tf.compat.as_bytes(line) for line in value]))


def writer(filename,
           compression_type=tf.python_io.TFRecordCompressionType.GZIP):
    return tf.python_io.TFRecordWriter(
        filename,
        options=tf.python_io.TFRecordOptions(
            compression_type=compression_type))


def metadata2feature(metadata):
    feature = {"parser": None, "converter": None}
    typedict = {
        "int64": tf.int64,
        "float32": tf.float32,
        "string": tf.string,
    }
    converter = {
        "int64": int64_feature,
        "float32": float_feature,
        "string": bytes_feature,
    }
    listconverter = {
        "int64": int64_list_feature,
        "float32": float_list_feature,
        "string": bytes_list_feature,
    }
    if metadata["type"] == "string":
        default_value = ""
    else:
        default_value = 0
    if metadata["shape"] is None:
        feature["parser"] = tf.VarLenFeature(typedict[metadata["type"]])
        feature["converter"] = listconverter[metadata["type"]]
    else:
        feature["parser"] = tf.FixedLenFeature(
            shape=metadata["shape"],
            dtype=typedict[metadata["type"]],
            default_value=default_value)
        if not metadata["shape"]:
            feature["converter"] = converter[metadata["type"]]
        else:
            feature["converter"] = listconverter[metadata["type"]]
    return feature


FEATURE = {key: metadata2feature(value) for key, value in METADATA.items()}


def example(feature):
    feature = {
        name: value
        for name, value in feature.items() if value is not None
    }
    if "image" in feature and hasattr(feature["image"], "shape"):
        image_shape = feature["image"].shape
        if len(image_shape) > 1:
            for index, name in enumerate(("image/height", "image/width",
                                          "image/depth")):
                if name not in feature:
                    if len(image_shape) > index:
                        feature[name] = image_shape[index]
                    else:
                        feature[name] = 1
        feature["image"] = numpy.reshape(feature["image"], [-1])
        feature["image"] = feature["image"].astype(numpy.int64)

    logging.debug("logging example begins:")
    for name, value in feature.items():
        if name not in FEATURE:
            raise NameError("unexcepted feature name '{}'".format(name))
        if name not in ("image"):
            logging.debug("{}: {}".format(name, value))
    logging.debug("logging example ends.\n")
    feature = {
        name: FEATURE[name]["converter"](value)
        for name, value in feature.items()
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def parse_single_example(example_proto):
    feature = tf.parse_single_example(
        example_proto,
        {name: value["parser"]
         for name, value in FEATURE.items()})
    for name in FEATURE:
        if name in feature:
            if FEATURE[name]["parser"].dtype == tf.string:
                default_value = ""
            else:
                default_value = 0
            if isinstance(FEATURE[name]["parser"], tf.VarLenFeature):
                feature[name] = tf.sparse_tensor_to_dense(
                    feature[name], default_value=default_value)
    if "image" in feature:
        feature["image"] = tf.reshape(
            feature["image"],
            tf.cast([
                feature["image/height"], feature["image/width"],
                feature["image/depth"]
            ], tf.int32))
    return feature


def dataset(filenames, compression_type="GZIP"):
    return tf.data.TFRecordDataset(
        filenames=filenames,
        compression_type=compression_type).map(parse_single_example)
