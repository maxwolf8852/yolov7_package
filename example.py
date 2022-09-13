from yolov7_package import Yolov7Detector
import cv2, time, os

if __name__ == '__main__':

    img = cv2.imread('img.jpg')

    det = Yolov7Detector(traced=True, weights=['yolov7.torchscript_16.pt']) # traced=True, weights=['yolov7.torchscript.pt']

    _ = det.detect(img)
    _ = det.detect(img)
    start = time.time()
    classes, boxes, scores = det.detect([img, img.copy()])
    print(classes, boxes, scores, time.time() - start)
    img = det.draw_on_image(img, boxes[0], scores[0], classes[0])
    print(img.shape)

    cv2.imshow("image", img)
    cv2.waitKey()


