from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import ops, models
from torchvision.ops import boxes as box_ops

from detection.layers import FrozenBatchNorm2d, smooth_l1_loss
from detection.layers import cat
from detection.modeling.utils import BalancedPositiveNegativeSampler, BoxCoder, Matcher


class VGG16BoxPredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        pool_size = cfg.MODEL.ROI_BOX_HEAD.POOL_RESOLUTION

        self.classifier = nn.Sequential(
            nn.Linear(in_channels * pool_size ** 2, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
        )

        self.cls_score = nn.Linear(4096, num_classes)
        self.bbox_pred = nn.Linear(4096, num_classes * 4)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

    def forward(self, box_features):
        box_features = box_features.view(box_features.size(0), -1)
        box_features = self.classifier(box_features)
        class_logits = self.cls_score(box_features)
        box_regression = self.bbox_pred(box_features)
        return class_logits, box_regression


class ResNetBoxPredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES

        resnet = models.resnet.__dict__[cfg.MODEL.BACKBONE.NAME](pretrained=True, norm_layer=FrozenBatchNorm2d)
        self.extractor = resnet.layer4
        del resnet

        in_channels = self.extractor[-1].conv3.out_channels
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

    def forward(self, box_features):
        box_features = self.extractor(box_features)
        box_features = torch.mean(box_features, dim=(2, 3))
        class_logits = self.cls_score(box_features)
        box_regression = self.bbox_pred(box_features)
        return class_logits, box_regression


BOX_PREDICTORS = {
    'vgg16_predictor': VGG16BoxPredictor,
    'resnet101_predictor': ResNetBoxPredictor,
}


def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    labels = cat(labels, dim=0)
    regression_targets = cat(regression_targets, dim=0)

    classification_loss = F.cross_entropy(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, -1, 4)

    box_loss = smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1,
        size_average=False,
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss


class BoxHead(nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()
        # fmt:off
        batch_size           = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
        score_thresh         = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        nms_thresh           = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        detections_per_img   = cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG

        box_predictor        = cfg.MODEL.ROI_BOX_HEAD.BOX_PREDICTOR
        spatial_scale        = cfg.MODEL.ROI_BOX_HEAD.POOL_SPATIAL_SCALE
        pool_size            = cfg.MODEL.ROI_BOX_HEAD.POOL_RESOLUTION
        pool_type            = cfg.MODEL.ROI_BOX_HEAD.POOL_TYPE
        # fmt:on

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

        if pool_type == 'align':
            pooler = partial(ops.roi_align, output_size=(pool_size, pool_size), spatial_scale=spatial_scale, sampling_ratio=2)
        elif pool_type == 'pooling':
            pooler = partial(ops.roi_pool, output_size=(pool_size, pool_size), spatial_scale=spatial_scale)
        else:
            raise ValueError('Unknown pool type {}'.format(pool_type))
        self.pooler = pooler

        self.box_predictor = BOX_PREDICTORS[box_predictor](cfg, in_channels)
        self.box_coder = BoxCoder(weights=(10., 10., 5., 5.))
        self.matcher = Matcher(0.5, 0.5, allow_low_quality_matches=False)
        self.fg_bg_sampler = BalancedPositiveNegativeSampler(batch_size, 0.25)

    def forward(self, features, proposals, img_metas, targets=None):
        if self.training:
            with torch.no_grad():
                proposals, labels, regression_targets = self.select_training_samples(proposals, targets)

        box_features = self.pooler(features, proposals)

        class_logits, box_regression = self.box_predictor(box_features)

        if self.training:
            classification_loss, box_loss = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            loss = {
                'rcnn_cls_loss': classification_loss,
                'rcnn_reg_loss': box_loss,
            }
            dets = []
        else:
            loss = {}
            dets = self.post_processor(class_logits, box_regression, proposals, img_metas)
        return dets, loss

    def post_processor(self, class_logits, box_regression, proposals, img_metas):
        num_classes = class_logits.shape[1]
        device = class_logits.device

        boxes_per_image = [box.shape[0] for box in proposals]
        proposals = cat([box for box in proposals])
        pred_boxes = self.box_coder.decode(
            box_regression.view(sum(boxes_per_image), -1), proposals
        )
        pred_boxes = pred_boxes.reshape(sum(boxes_per_image), -1, 4)

        pred_scores = F.softmax(class_logits, -1)

        # split boxes and scores per image
        if len(boxes_per_image) == 1:
            pred_boxes = (pred_boxes,)
            pred_scores = (pred_scores,)
        else:
            pred_boxes = pred_boxes.split(boxes_per_image, dim=0)  # (N, #CLS, 4)
            pred_scores = pred_scores.split(boxes_per_image, dim=0)  # (N, #CLS)

        results = []
        for scores, boxes, img_meta in zip(pred_scores, pred_boxes, img_metas):
            width, height = img_meta['img_shape']
            boxes = box_ops.clip_boxes_to_image(boxes, (height, width))

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            results.append((boxes, scores, labels))

        return results

    def select_training_samples(self, proposals, targets):
        labels = []
        regression_targets = []
        for batch_id in range(len(targets)):
            target = targets[batch_id]
            proposals_per_image = proposals[batch_id]

            match_quality_matrix = box_ops.box_iou(target['boxes'], proposals_per_image)
            matched_idxs = self.matcher(match_quality_matrix)

            matched_idxs_for_target = matched_idxs.clamp(0)

            target_boxes = target['boxes'][matched_idxs_for_target]
            target_labels = target['labels'][matched_idxs_for_target]
            labels_per_image = target_labels.to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                target_boxes, proposals_per_image
            )
            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)

        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        proposals = list(proposals)

        # distributed sampled proposals, that were obtained on all feature maps
        # concatenated via the fg_bg_sampler, into individual feature map levels
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            proposals[img_idx] = proposals[img_idx][img_sampled_inds]
            labels[img_idx] = labels[img_idx][img_sampled_inds]
            regression_targets[img_idx] = regression_targets[img_idx][img_sampled_inds]

        return proposals, labels, regression_targets
