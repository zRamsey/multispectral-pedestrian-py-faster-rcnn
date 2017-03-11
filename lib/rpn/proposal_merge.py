# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import caffe
import numpy as np
import yaml
from fast_rcnn.config import cfg
from fast_rcnn.nms_wrapper import nms

DEBUG = False

class ProposalMergeLayer(caffe.Layer):

	def setup(self, bottom, top):
		'''
		bottom[0]: proposal boxes channel 1 (N*1)
		bottom[1]: proposal scores channel 1 (N*5)
		bottom[2]: proposal boxes channel 2
		bottom[3]: proposal scores channel 2
		'''
		assert(bottom[0].data.shape[1]==bottom[2].data.shape[1])
		assert(bottom[0].data.shape[0]==bottom[1].data.shape[0])
		assert(bottom[2].data.shape[0]==bottom[3].data.shape[0])
		top[0].reshape(1, 5)

		if len(top) > 1:
			top[1].reshape(1, 1, 1, 1)

	def forward(self, bottom, top):

		cfg_key = str(self.phase) # either 'TRAIN' or 'TEST'
		post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
		nms_thresh    = cfg[cfg_key].RPN_NMS_THRESH

		boxes 	 = bottom[0].data
		scores   = bottom[1].data
		boxes_p   = bottom[2].data
		scores_p  = bottom[3].data

		scores_merge = np.vstack((scores, scores_p))
		boxes_merge  = np.vstack((boxes, boxes_p))
		ind = np.argsort(scores_merge,axis=0)[::-1]

		boxes_out = boxes_merge[ind,:][:,0,:]
		scores_out = scores_merge[ind,:][:,0,:]

		'''
		keep = nms(np.hstack((boxes_out[:,1:], scores_out)), nms_thresh)

		if post_nms_topN > 0:
			keep = keep[:post_nms_topN]

		boxes_out = boxes_out[keep, :]
		scores_out = scores_out[keep]
		'''
		boxes_out = boxes_out[0:post_nms_topN,:]
		scores_out = scores_out[0:post_nms_topN,:]

		# print scores_out.shape
		
		top[0].reshape(*(boxes_out.shape))
		top[0].data[...] = boxes_out

		if len(top) > 1:
			top[1].reshape(*(scores_out.shape))
			top[1].data[...] = scores_out

	def backward(self, top, propagate_down, bottom):
		"""This layer does not propagate gradients."""
		pass

	def reshape(self, bottom, top):
		"""Reshaping happens during the call to forward."""
		pass


