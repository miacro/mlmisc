import cv2
import numpy


def normalize_shape(image, max_height=3200):
    if image.shape[0] <= max_height:
        return image
    scale = max_height / image.shape[0]
    # INTER_AREA is better when zoom out an image
    return cv2.resize(
        image,
        dsize=(0, 0),
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )


def detect_edges(image, low_threshold=100):
    # Reduce noise with a kernel 3x3
    image = cv2.blur(image, (3, 3))
    # Canny recommended a upper:lower ratio between 2:1 and 3:1
    ratio = 2.5
    return cv2.Canny(
        image,
        threshold1=low_threshold,
        threshold2=low_threshold * ratio,
        apertureSize=3,
    )


def detect_lines(
        image,
        rho=1,
        theta=numpy.pi / 180 / 3,
        threshold=30,
        minLineLength=100,
        maxLineGap=32,
):
    image = detect_edges(image)
    lines = cv2.HoughLinesP(
        image,
        rho=rho,
        theta=theta,
        threshold=threshold,
        minLineLength=minLineLength,
        maxLineGap=maxLineGap,
    )
    return lines


def tilt_correction(image):
    lines = detect_lines(image)
    if lines is None:
        return image
    lines = [line for item in lines for line in item]
    thetas = []
    for x1, y1, x2, y2 in lines:
        if x1 != x2:
            theta = numpy.arctan((y1 - y2) / (x1 - x2))
            if numpy.abs(theta) < numpy.pi / 4:
                thetas.append(theta)
    if not thetas:
        return image
    theta = numpy.average(thetas)
    height, width = image.shape[:2]
    M = cv2.getRotationMatrix2D((width / 2, height / 2),
                                theta / numpy.pi * 180, 1)
    return cv2.warpAffine(
        image,
        M,
        (width, height),
        borderValue=255,
        borderMode=cv2.BORDER_CONSTANT,
    )


def binarize(image):
    # determines the optimal threshold value using the Otsu’s algorithm
    threshold, image = cv2.threshold(
        image,
        thresh=0,
        maxval=255,
        type=cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV,
    )
    return threshold, image


def remove_lines(
        image,
        element_size=32,
):
    """
    Extract horizontal and vertical lines by using morphological operations
    """
    threshold, gray_image = cv2.threshold(
        image,
        thresh=0,
        maxval=255,
        type=cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
    horizontal_image = cv2.morphologyEx(
        gray_image,
        op=cv2.MORPH_OPEN,
        kernel=cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (element_size * 2, 1),
        ),
    )
    horizontal_image = cv2.dilate(
        horizontal_image,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, int(element_size / 8)),
        ),
    )
    cv2.imwrite("image-hor-morphology.png", horizontal_image)

    vertical_image = cv2.morphologyEx(
        gray_image,
        op=cv2.MORPH_OPEN,
        kernel=cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, element_size * 2),
        ),
    )
    vertical_image = cv2.dilate(
        vertical_image,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (int(element_size / 8), 1),
        ),
    )
    cv2.imwrite("image-ver-morphology.png", vertical_image)

    image = cv2.bitwise_or(image, horizontal_image)
    image = cv2.bitwise_or(image, vertical_image)
    return image


def denoise(image):
    return cv2.fastNlMeansDenoising(
        image,
        h=10,
        templateWindowSize=7,
        searchWindowSize=21,
    )


def merge_bboxes(
        bboxes,
        min_gap=16,
        min_area=None,
        max_area=None,
        min_area_perimeter_ratio=4,
):
    def overlap(bbox1, bbox2):
        def axis_overlap(min1, max1, min2, max2):
            distance = abs((min1 + max1) / 2 - (min2 + max2) / 2)
            if (distance - (max1 - min1 + max2 - min2) / 2) > min_gap:
                return False
            return True

        return (axis_overlap(
            bbox1[0],
            bbox1[2],
            bbox2[0],
            bbox2[2],
        ) and axis_overlap(
            bbox1[1],
            bbox1[3],
            bbox2[1],
            bbox2[3],
        ))

    if min_area is not None:
        tiny_threshold = int(numpy.sqrt(min_area) / 4)
        bboxes = [
            bbox for bbox in bboxes
            if tiny_threshold < (bbox[2] - bbox[0]) and tiny_threshold <
            (bbox[3] - bbox[1])
        ]

    groups = []
    while bboxes:
        group = [bboxes.pop()]
        grouped_indexes = []
        for current in group:
            for index, box in enumerate(bboxes):
                if index in grouped_indexes:
                    continue
                if overlap(box, current):
                    grouped_indexes.append(index)
                    group.append(box)
        groups.append(group)
        bboxes = [
            bboxes[index] for index in range(len(bboxes))
            if index not in grouped_indexes
        ]
    bboxes = []
    for group in groups:
        xmin = min([item[0] for item in group])
        ymin = min([item[1] for item in group])
        xmax = max([item[2] for item in group])
        ymax = max([item[3] for item in group])
        area = (xmax - xmin) * (ymax - ymin)
        perimeter = 2 * (xmax - xmin + ymax - ymin)
        if min_area is not None and area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        if (min_area_perimeter_ratio is not None
                and area / perimeter < min_area_perimeter_ratio):
            continue
        bboxes.append((xmin, ymin, xmax, ymax))
    return bboxes


def detect_regions(image):
    image_edges = detect_edges(image)
    _, contours, hierarchy = cv2.findContours(
        image_edges,
        mode=cv2.RETR_TREE,
        method=cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return []
    contours = [
        contours[index] for index, item in enumerate(hierarchy[0])
        if item[3] == -1
    ]
    bboxes = [cv2.boundingRect(item) for item in contours]
    bboxes = [(bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
              for bbox in bboxes]
    return bboxes


if __name__ == "__main__":
    import sys
    import logging
    logging.basicConfig(
        format="[%(asctime)s]:%(levelname)s:%(name)s:%(message)s",
        level=logging.INFO,
    )
    logging.getLogger().name = "imagetest"
    image = cv2.imread(sys.argv[1])
    logging.info("start:")
    image = normalize_shape(image, max_height=3200)
    cv2.imwrite("image-resize.png", image)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("image-gray.png", image)

    logging.info("image tilt_correction")
    image = tilt_correction(image)
    cv2.imwrite("image-rotation.png", image)

    logging.info("image removelines")
    image = remove_lines(image)
    cv2.imwrite("image-removelines.png", image)

    # logging.info("image denoise")
    # image = denoise(image)
    # cv2.imwrite("image-denoise.png", image)

    logging.info("image detect object")
    bboxes = detect_regions(image)
    bboxes = merge_bboxes(
        bboxes,
        min_area=8 * 8,
    )
    for bbox in bboxes:
        cv2.rectangle(image, bbox[0:2], bbox[2:4], (0, 0, 0), 3)
    cv2.imwrite("image-end.png", image)
    logging.info("end.")
