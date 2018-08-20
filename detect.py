import caffe
import cv2
import numpy as np


net_path = 'MobileNetSSD_deploy.prototxt'
weights_path = 'MobileNetSSD_deploy.caffemodel'
voc_names = [
    'background',
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
]


class Detector:
    """
    Detect person use MobileNet SSD
    Neet to use ssdcaffe: https://github.com/weiliu89/caffe/tree/ssd
    """

    def __init__(self, net_path=net_path, net_weights_path=weights_path, conf_thresh=0.35):
        self.net_path = net_path
        self.net_weights_path = net_weights_path
        self.net = caffe.Net(net_path, net_weights_path, caffe.TEST)

        self.height = 400
        self.width = 600

        self._needed_classes = ['person', 'bus', 'car']
        self._confidence_threshold = conf_thresh

    def _convert(self, img):
        """
        :param img: input BGR image, uint8
        :return: input blob data for CNN
        """
        img = cv2.resize(img, (self.width, self.height))
        img = img.transpose((2, 0, 1))
        img = img.astype(np.float32)
        img -= 127.0
        img *= (1.0 / 127.0)
        img = img[np.newaxis, :, :, :]
        return img

    def _parse_output(self, output, width, height):
        # return bounding boxes in the original scale.
        # in OpenCV format (x, y, w, h)
        bounding_boxes = []
        counting = []
        for _ in self._needed_classes:
            counting.append(0)

        for det in output:
            label_id = int(det[1])
            label_name = voc_names[label_id]

            if label_name not in self._needed_classes:
                continue

            # the last layer of CNN already filter out
            # the most low-confidence detection,
            # You may use this threshold if the threshold
            # in the .prototxt file is too low.
            confidence = det[2]
            if confidence < self._confidence_threshold:
                continue

            x_min = int(det[3] * width)
            y_min = int(det[4] * height)
            x_max = int(det[5] * width)
            y_max = int(det[6] * height)

            bounding_box, valid = self.get_valid_bounding_box(x_min, y_min, x_max, y_max, width, height)

            if valid:
                idx = self._needed_classes.index(label_name)
                bounding_boxes.append(bounding_box)
                counting[idx] += 1

        return bounding_boxes, counting

    def get_classes(self):
        return self._needed_classes

    @staticmethod
    def get_valid_bounding_box(x_min, y_min, x_max, y_max, width, height):
        x_min = max(x_min, 0)
        y_min = max(y_min, 0)
        x_max = min(width - 1, x_max)
        y_max = min(height - 1, y_max)

        if x_max <= x_min or y_max <= y_min:
            return None, False

        width = x_max - x_min + 1
        height = y_max - y_min + 1
        return (x_min, y_min, width, height), True

    def infer(self, img):
        assert(img.shape[2] == 3)
        height = img.shape[0]
        width = img.shape[1]
        input_data = self._convert(img)
        self.net.blobs['data'].reshape(*input_data.shape)
        self.net.blobs['data'].data[...] = input_data

        output = self.net.forward()
        output = output['detection_out'][0][0]

        return self._parse_output(output, width, height)
