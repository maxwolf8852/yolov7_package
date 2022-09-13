import argparse
import colorsys
import os, sys

import cv2
import torch
import numpy as np

from .models.experimental import attempt_load
from .utils.general import non_max_suppression, non_max_suppression_kpt
from .utils.torch_utils import TracedModel


def conf2color(conf):
    if (conf <= 0.6):
        color = colorsys.hsv_to_rgb(15 / 360.0, 1, 1)
    else:
        color = colorsys.hsv_to_rgb((25 + 100 * (2 * conf - 1.2)) / 360.0, 1, 1)
    return [k * 255 for k in reversed(color)]

coco_names = ["person",
              "bicycle",
              "car",
              "motorbike",
              "aeroplane",
              "bus",
              "train",
              "truck",
              "boat",
              "traffic light",
              "fire hydrant",
              "stop sign",
              "parking meter",
              "bench",
              "bird",
              "cat",
              "dog",
              "horse",
              "sheep",
              "cow",
              "elephant",
              "bear",
              "zebra",
              "giraffe",
              "backpack",
              "umbrella",
              "handbag",
              "tie",
              "suitcase",
              "frisbee",
              "skis",
              "snowboard",
              "sports ball",
              "kite",
              "baseball bat",
              "baseball glove",
              "skateboard",
              "surfboard",
              "tennis racket",
              "bottle",
              "wine glass",
              "cup",
              "fork",
              "knife",
              "spoon",
              "bowl",
              "banana",
              "apple",
              "sandwich",
              "orange",
              "broccoli",
              "carrot",
              "hot dog",
              "pizza",
              "donut",
              "cake",
              "chair",
              "sofa",
              "pottedplant",
              "bed",
              "diningtable",
              "toilet",
              "tvmonitor",
              "laptop",
              "mouse",
              "remote",
              "keyboard",
              "cell phone",
              "microwave",
              "oven",
              "toaster",
              "sink",
              "refrigerator",
              "book",
              "clock",
              "vase",
              "scissors",
              "teddy bear",
              "hair drier",
              "toothbrush"]



class Yolov7Detector:
    def __init__(self,
                 weights=None,
                 img_size=None,
                 conf_thres=0.25,
                 iou_thres=0.45,
                 augment=False,
                 agnostic_nms=False,
                 device='cuda:0',
                 traced=False):
        if img_size is None:
            img_size = [640, 640]
        if weights is None:
            weights = ['yolov7.pt']

        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        cpu = device.lower() == 'cpu'
        cuda = not cpu and torch.cuda.is_available()
        self.device = torch.device(device if cuda else 'cpu')

        self.half = self.device.type != 'cpu'

        sys.path.append(os.path.join(os.path.dirname(__file__), ""))
        load_err = False

        if not traced:
            self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        else:
            self.model = torch.jit.load(weights[0], map_location=self.device).float().eval()

        self.traced = traced

        self.img_size = img_size  # check_img_size(img_size, s=self.stride)  # check img_size


        #self.stride = int(self.model.stride.max())  # model stride

        if self.half:
            self.model.half()

        self.names = coco_names
        self.model.eval()

    def detect(self, x) -> (list, list, list):
        """
        :param x: list of numpy images (e.g. after cv2.imread) or numpy image
        :return: []classes, []boxes, []confidences
        """
        if type(x) != list:
            x = [x]
        for i, img in enumerate(x):
            if type(img) != np.array:
                x[i] = np.array(img)
        with torch.no_grad():
            classes = []
            boxes = []
            confs = []
            img_sizes = []
            y = []
            for i, img in enumerate(x):
                img_sizes.append(img.shape)
                img_inf = cv2.resize(img, self.img_size)
                # img_inf = letterbox(img, self.img_size, stride=self.stride)[0]
                img_inf = img_inf[:, :, ::-1].transpose(2, 0, 1)
                img_inf = np.ascontiguousarray(img_inf)

                img_inf = torch.from_numpy(img_inf).to(self.device)
                img_inf = img_inf.half() if self.half else img_inf.float()  # uint8 to fp16/32
                img_inf /= 255.0  # 0 - 255 to 0.0 - 1.0
                y.append(img_inf)

            y = torch.stack(y)
            pred = self.model(y, )[0] # augment=self.augment
            #print(pred.shape)
            #if self.traced:
            #    pred = pred[0].unsqueeze(0)

            #print(pred.shape)

            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=None,
                                       agnostic=self.agnostic_nms)
            #print(pred)
            for i, det in enumerate(pred):  # detections per image
                local_classes = []
                local_boxes = []
                local_confs = []
                old_shape = img_sizes[i]
                #gn = torch.tensor(old_shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    dx = old_shape[0] / self.img_size[0]
                    dy = old_shape[1] / self.img_size[1]
                    #print(det[:, :4])
                    #det[:, :4] = scale_coords(self.img_size, det[:, :4], old_shape[:2]).round()
                    #print(det[:, :4])

                    for *xyxy, conf, cls in reversed(det):
                        coords = torch.tensor(xyxy).tolist()
                        xyxy_scaled = [coords[0] * dy, coords[1] * dx, coords[2] * dy, coords[3] * dx]
                        #print(, conf, cls)
                        #xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        local_classes.append(int(cls.cpu().item()))
                        local_boxes.append(xyxy_scaled)
                        local_confs.append(float(conf.cpu().item()))

                classes.append(local_classes)
                boxes.append(local_boxes)
                confs.append(local_confs)


            return classes, boxes, confs

    def draw_on_image(self, img, boxes: list, scores: list, class_ids: list, thickness=1):
        for i, box in enumerate(boxes):
            name = self.names[class_ids[i]]
            color = conf2color(scores[i])
            c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            img = cv2.rectangle(img, c1, c2, color, thickness=thickness)
            img = cv2.putText(img, name, (c1[0], c1[1] - 2), 0, 1, color, thickness=thickness)
        return img
