from yolov7_package import Yolov7Detector
import cv2, time, os


def test_train():  # not implemented now
    ...
    """det.train('save',
                  'yolov7_package/data/coco.yaml',
                  'yolov7_package/cfg/training/yolov7.yaml',
                  'yolov7_package/data/hyp.scratch.p5.yaml')"""


def test(img_path, traced):
    img = cv2.imread(img_path)
    det = Yolov7Detector(traced=traced)

    _ = det.detect(img)
    _ = det.detect(img)
    start = time.time()
    classes, boxes, scores = det.detect([img, img.copy()])
    print(f'Time: {time.time() - start}')
    img = det.draw_on_image(img, boxes[0], scores[0], classes[0])

    cv2.imshow("image", img)
    cv2.waitKey()


if __name__ == '__main__':
    # test detection
    test('img.jpg', False)

    # test traced model
    test('img.jpg', True)
