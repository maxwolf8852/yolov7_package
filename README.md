![GitHub](https://img.shields.io/github/license/maxwolf8852/yolov7_package?style=plastic)
[![PyPI version](https://badge.fury.io/py/yolov7_package.svg)](https://badge.fury.io/py/yolov7_package)
![PyPI - Downloads](https://img.shields.io/pypi/dm/yolov7_package?style=plastic)

# This package is deprecated. Now use ![DetExecutor](https://github.com/maxwolf8852/DetExecutor)

<h3 align="center">
  WongKinYiu/yolov7 as independent package
</h3>

Bindings for yolov7 project (https://github.com/WongKinYiu/yolov7)

Example of usage:
```Python
from yolov7_package import Yolov7Detector
import cv2

if __name__ == '__main__':
    img = cv2.imread('img.jpg')
    det = Yolov7Detector(traced=False)
    classes, boxes, scores = det.detect(img)
    img = det.draw_on_image(img, boxes[0], scores[0], classes[0])

    cv2.imshow("image", img)
    cv2.waitKey()
```

You can use <b>traced=True</b> option to download and infer traced FP16 version of yolov7 model! 

PYPI link: https://pypi.org/project/yolov7-package/
