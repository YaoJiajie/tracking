import cv2
from detect import Detector
import numpy as np
import sys
from sort.sort import Sort
import caffe


colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],
          [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85],
          [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
          [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255],
          [255, 0, 170], [255, 0, 85]]


input_height = 400
input_width = 600


def draw_bounding_boxes(frame, bounding_boxes, color):
    for bb in bounding_boxes:
        x1, y1 = bb[0], bb[1]
        x2, y2 = bb[0] + bb[2] - 1, bb[1] + bb[3] - 1
        cv2.rectangle(frame, (x1, y1), (x2,  y2), color, 3)


def track_info_to_roi(track_info):
    x1 = int(track_info[0])
    y1 = int(track_info[1])
    x2 = int(track_info[2])
    y2 = int(track_info[3])
    roi = (x1, y1, (x2 - x1 + 1), (y2 - y1 + 1))
    return roi


def draw_tracking_results(track_results, image):
    for track_result in track_results:
        trk_id = int(track_result[4])
        roi = track_info_to_roi(track_result)
        color = colors[trk_id % len(colors)]
        draw_bounding_boxes(image, [roi, ], color)
        # draw the number
        cv2.putText(image, str(trk_id), (roi[0] + 3, roi[1] + 3),
                    cv2.FONT_HERSHEY_COMPLEX, 1.5, color, 3)


def track(video_path, use_gpu=False):
    video = cv2.VideoCapture(video_path)
    ret, frame = video.read()
    if ret:
        frame = cv2.resize(frame, (input_width, input_height))

    if use_gpu:
        caffe.set_mode_gpu()

    tracker = Sort(max_age=10)
    detector = Detector()
    classes = detector.get_classes()

    while ret:
        frame_disp = np.copy(frame)
        bounding_boxes, counting = detector.infer(frame)
        class_counting = zip(classes, counting)

        for pair in class_counting:
            print('{:s} {:03d}'.format(*pair))
        print('')

        if len(bounding_boxes) > 0:
            bounding_boxes = np.array(bounding_boxes, np.int32)

            # convert (x, y, w, h) to (x1, y1, x2, y2)
            bounding_boxes[:, 2:4] += bounding_boxes[:, 0:2]
            bounding_boxes[:, 2:4] -= 1

        track_results = tracker.update(bounding_boxes)
        draw_tracking_results(track_results, frame_disp)

        cv2.imshow('tracking', frame_disp)

        key = cv2.waitKey(1)
        if key == 27:
            return

        ret, frame = video.read()
        if ret:
            frame = cv2.resize(frame, (input_width, input_height))


if __name__ == '__main__':
    video_path = sys.argv[1]
    use_gpu = False

    if len(sys.argv) >= 3 and sys.argv[2] == '1':
        use_gpu = True

    track(video_path, use_gpu)
