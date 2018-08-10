from mltools.thirdparty.coco_text.coco_text import COCO_Text
import os
import cv2
import logging
from mltools.dataset.datasettools import fromlabels


def annotations(annofile, imagedir=None, labels=None):
    """
    image["annotation"]["language"] in ("not english", "na", "english")
    image["annotation"]["legibility"] in ("legible", "illegible")
    image["annotation"]["class"] in
        ("others", "handwritten", "machine printed")
    """
    if isinstance(annofile, list):
        if len(annofile) > 1:
            logging.warn("only the first annofile({}) will be used".format(
                annofile[0]))
        annofile = annofile[0]
    coco = COCO_Text(annofile)
    if imagedir is not None:
        imagedir = os.path.expanduser(os.path.expandvars(imagedir))
    image_map = {}
    for annId in coco.getAnnIds():
        ann = coco.loadAnns(ids=[annId])[0]
        image_id = ann["image_id"]
        if image_id not in image_map:
            annotation = {
                "xmin": [],
                "ymin": [],
                "xmax": [],
                "ymax": [],
                "class/text": [],
                "class/label": [],
                "area": [],
                "difficulty": [],
                "language": [],
                "text": [],
            }
            image_map[image_id] = annotation
        else:
            annotation = image_map[image_id]

        (x, y, width, height) = tuple(ann["bbox"])
        if width <= 0 or height <= 0:
            continue
        annotation["xmin"].append(x)
        annotation["ymin"].append(y)
        annotation["xmax"].append(x + width)
        annotation["ymax"].append(y + height)
        annotation["language"].append(ann["language"])
        annotation["class/text"].append(ann["class"])
        if labels is not None:
            label = fromlabels(ann["class"], labels, update=True)
            annotation["class/label"].append(label)
        else:
            annotation["class/label"].append(None)
        annotation["area"].append(ann["area"])
        annotation["difficulty"].append(1 if ann["legibility"] == "illegible"
                                        else 0)
        annotation["text"].append(ann["utf8_string"]
                                  if "utf8_string" in ann else "")
    for image_id in image_map:
        if len(image_map[image_id]) <= 0:
            continue
        image_anno = coco.loadImgs(ids=[image_id])[0]
        if imagedir is not None and image_anno["file_name"]:
            image = cv2.imread(
                os.path.join(imagedir, image_anno["file_name"]))
        else:
            image = None
        annotation = image_map[image_id]
        yield {
            "image": image,
            "image/filename": image_anno["file_name"],
            "image/object/bbox/xmin": annotation["xmin"],
            "image/object/bbox/ymin": annotation["ymin"],
            "image/object/bbox/xmax": annotation["xmax"],
            "image/object/bbox/ymax": annotation["ymax"],
            "image/object/class/text": annotation["class/text"],
            "image/object/class/label": annotation["class/label"],
            "image/object/area": annotation["area"],
            "image/object/difficulty": annotation["difficulty"],
            "image/object/language": annotation["language"],
            "image/object/text": annotation["text"],
        }
