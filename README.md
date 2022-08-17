<h3 align="center">
  WongKinYiu/yolov7 as independent package
</h3>

Bindings for yolov7 project (https://github.com/WongKinYiu/yolov7)

Example of usage:
```Python
from detect import Yolov7Detector
import cv2

if __name__ == '__main__':
    img = cv2.imread('img.jpg')
    #img = cv2.resize(img, [640, 640])
    det = Yolov7Detector()
    classes, boxes, scores = det.detect(img)
    print(classes, boxes, scores)
    img = det.draw_on_image(img, boxes[0], scores[0], classes[0])
    print(img.shape)

    cv2.imshow("image", img)
    cv2.waitKey()
```
