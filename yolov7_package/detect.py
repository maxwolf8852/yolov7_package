import argparse
import colorsys
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

def conf2color(conf):
    if (conf <= 0.6):
        color = colorsys.hsv_to_rgb(15 / 360.0, 1, 1)
    else:
        color = colorsys.hsv_to_rgb((25 + 100 * (2 * conf - 1.2)) / 360.0, 1, 1)
    return [k * 255 for k in reversed(color)]

class Yolov7Detector:
    def __init__(self,
                 weights=None,
                 img_size=None,
                 conf_thres=0.25,
                 iou_thres=0.45,
                 augment=False,
                 agnostic_nms=False,
                 device='cuda:0'):
        if img_size is None:
            img_size = [640, 640]
        if weights is None:
            weights = ['yolov7_package/model_weights/yolov7.pt']

        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        cpu = device.lower() == 'cpu'
        cuda = not cpu and torch.cuda.is_available()
        self.device = torch.device('cuda:0' if cuda else 'cpu')

        self.half = self.device.type != 'cpu'

        self.model = attempt_load(weights, map_location=device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.img_size = img_size  # check_img_size(img_size, s=self.stride)  # check img_size

        if self.half:
            self.model.half()

        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.model.eval()

    def detect(self, x) -> (list, list, list):
        """
        :param x: list of numpy images (e.g. after cv2.imread) or numpy image
        :return: []classes, []boxes, []confidences
        """
        if type(x) != list:
            x = [x]
        with torch.no_grad():
            classes = []
            boxes = []
            confs = []
            img_sizes = []
            for i, img in enumerate(x):
                img_sizes.append(img.shape)
                img_inf = cv2.resize(img, self.img_size)
                # img_inf = letterbox(img, self.img_size, stride=self.stride)[0]
                img_inf = img_inf[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                img_inf = np.ascontiguousarray(img_inf)

                img_inf = torch.from_numpy(img_inf).to(self.device)
                img_inf = img_inf.half() if self.half else img_inf.float()  # uint8 to fp16/32
                img_inf /= 255.0  # 0 - 255 to 0.0 - 1.0
                x[i] = img_inf

            x = torch.stack(x)
            pred = self.model(x, augment=self.augment)[0]
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=None,
                                       agnostic=self.agnostic_nms)
            for i, det in enumerate(pred):  # detections per image
                local_classes = []
                local_boxes = []
                local_confs = []
                old_shape = img_sizes[i]
                print(old_shape[:2], x.shape[2:], )
                gn = torch.tensor(old_shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    dx =  old_shape[0] / self.img_size[0]
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
