METADATA = {
    "image": {
        "type": "int64",
        "shape": None,
    },
    "image/height": {
        "type": "int64",
        "shape": [],
    },
    "image/width": {
        "type": "int64",
        "shape": [],
    },
    "image/depth": {
        "type": "int64",
        "shape": [],
    },
    "image/text": {
        "type": "string",
        "shape": [],
    },
    "image/text/label": {
        "type": "int64",
        "shape": None,
    },
    "image/text/length": {
        "type": "int64",
        "shape": [],
    },
    "image/filename": {
        "type": "string",
        "shape": [],
    },
    "image/class/text": {
        "type": "string",
        "shape": [],
    },
    "image/class/label": {
        "type": "int64",
        "shape": [],
    },
    "image/object/bbox/xmin": {
        "type": "float32",
        "shape": None,
    },
    "image/object/bbox/ymin": {
        "type": "float32",
        "shape": None,
    },
    "image/object/bbox/xmax": {
        "type": "float32",
        "shape": None,
    },
    "image/object/bbox/ymax": {
        "type": "float32",
        "shape": None,
    },
    "image/object/class/text": {
        "type": "string",
        "shape": None,
    },
    "image/object/class/label": {
        "type": "int64",
        "shape": None,
    },
    "image/object/area": {
        "type": "float32",
        "shape": None,
    },
    "image/object/language": {
        "type": "string",
        "shape": None,
    },
    "image/object/text": {
        "type": "string",
        "shape": None,
    },
    "image/object/difficulty": {
        "type": "int64",
        "shape": None,
    },
}


def rollover(annotation):
    annotation = {
        key: value
        for key, value in annotation.items() if value is not None
    }
    image_width = None
    if "image" in annotation:
        annotation["image"] = annotation["image"][:, ::-1, :]
        image_width = annotation["image"].shape[1]
    if "image/width" in annotation:
        image_width = annotation["image/width"]
    if "image/text" in annotation:
        annotation["image/text"] = annotation["image/text"][::-1]
    if "image/text/label" in annotation:
        annotation["image/text/label"] = annotation["image/text/label"][::-1]

    if (("image/object/bbox/xmin" in annotation)
            and ("image/object/bbox/xmax" in annotation)
            and (image_width is not None)):
        length = min([
            len(annotation["image/object/bbox/xmax"]),
            len(annotation["image/object/bbox/xmin"])
        ])
        for index in range(length):
            (xmin, xmax) = (
                image_width - annotation["image/object/bbox/xmax"][index] - 1,
                image_width - annotation["image/object/bbox/xmin"][index] - 1)
            annotation["image/object/bbox/xmin"][index] = xmin
            annotation["image/object/bbox/xmax"][index] = xmax

    if "image/object/text" in annotation:
        annotation["image/object/text"] = [
            item[::-1] for item in annotation["image/object/text"]
        ]
    return annotation
