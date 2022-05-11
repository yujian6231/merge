#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : image_demo.py
#   Author      : YunYang1994
#   Created date: 2019-01-20 16:06:06
#   Description :
#
# ================================================================

import cv2
import numpy as np
import core.utils as utils
import tensorflow as tf
from PIL import Image
import os


class yolo_2d:

    def __init__(self):
        #self.image = original_image
        self.return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0",
                                "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
        self.pb_file = "./checkpoint/yolov3_coco.pb"
        self.graph = tf.Graph()
        self.return_tensors = utils.read_pb_return_tensors(
            self.graph, self.pb_file, self.return_elements)

    def yolo_2d(self, original_image,pred_sbbox,pred_mbbox,pred_lbbox):
        # original_image=self.image
        # return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0",
        #                    "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
        # pb_file = "./checkpoint/yolov3_coco.pb"
        #image_path = "./docs/images/"
        num_classes = 80
        input_size = 416
        # graph = tf.Graph()

        #all_image = sorted(os.listdir(image_path))
        # for i in images:
        #original_image = cv2.imread(image_path+i)
        #original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        # cv2.imshow('eee',original_image)
        # cv2.waitKey(0)
        #original_image= original_image.astype(np.float32, copy=False)
        original_image_size = original_image.shape[:2]
        # image_data = utils.image_preporcess(
        #     np.copy(original_image), [input_size, input_size])
        # image_data = image_data[np.newaxis, ...]

        # return_tensors = utils.read_pb_return_tensors(
        #     self.graph, self.pb_file, self.return_elements)

        # with tf.Session(graph=self.graph) as sess:
        #     pred_sbbox, pred_mbbox, pred_lbbox = sess.run([self.return_tensors[1], self.return_tensors[2],self.return_tensors[3]],feed_dict={self.return_tensors[0]: image_data})

        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                    np.reshape(
                                        pred_mbbox, (-1, 5 + num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

        bboxes = utils.postprocess_boxes(
            pred_bbox, original_image_size, input_size, 0.3)
        bboxes = utils.nms(bboxes, 0.45, method='nms')

        #txt = np.savetxt(original_image.replace('png', 'txt'), bboxes, fmt='%d')
        return bboxes

        # image = utils.draw_bbox(original_image, bboxes)
        # #image = Image.fromarray(image)
        # result = np.asarray(image)
        # result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # #image.show()
        # cv2.imshow("result", result)
        # cv2.waitKey(0)
        # #if cv2.waitKey(1) & 0xFF == ord('q'): break
