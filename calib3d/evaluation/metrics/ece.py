# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Sequence

import mmengine
import numpy as np
import torch
import torch.nn.functional as F
from calib3d.registry import METRICS
from mmengine.evaluator import BaseMetric


@METRICS.register_module()
class ECEMetric(BaseMetric):
    """3D semantic segmentation evaluation metric.

    Args:
        collect_device (str, optional): Device name used for collecting
            results from different ranks during distributed training.
            Must be 'cpu' or 'gpu'. Defaults to 'cpu'.
        prefix (str): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Default: None.
        pklfile_prefix (str, optional): The prefix of pkl files, including
            the file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created. Default: None.
        submission_prefix (str, optional): The prefix of submission data.
            If not specified, the submission data will not be generated.
            Default: None.
    """

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 pklfile_prefix: str = None,
                 submission_prefix: str = None,
                 file_name: str = None,
                 **kwargs):
        self.pklfile_prefix = pklfile_prefix
        self.submission_prefix = submission_prefix
        self.file_name = file_name
        super(ECEMetric, self).__init__(
            prefix=prefix, collect_device=collect_device)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``,
        which will be used to compute the metrics when all batches
        have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """
        for data_sample in data_samples:
            pred_logit = data_sample['pts_seg_logits']
            eval_ann_info = data_sample['eval_ann_info']
            ece, bin_corrects, bin_scores = calculate_ece(pred_logit['pts_seg_logits'],
                                            eval_ann_info['pts_semantic_mask'],
                                            self.dataset_meta['ignore_index'])
            self.results.append((ece, bin_corrects, bin_scores))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        eces = []
        info = dict()
        for i, (ece, bin_correct, bin_score) in enumerate(self.results):
            eces.append(ece)
            info[i] = dict(bin_correct=bin_correct,
                           bin_score=bin_score)

        if self.file_name is not None:
            mmengine.dump(info, self.file_name)
        ret_dict = dict(ece=sum(eces) / len(eces))
        return ret_dict


def calculate_ece(logits, labels, ignore_index, n_bins=10):
    logits = logits.cpu()
    labels = torch.tensor(labels, dtype=torch.int64).to(device=logits.device)
    valid_index = labels != ignore_index
    valid_logits = logits[:, valid_index]
    valid_labels = labels[valid_index]

    softmaxes = F.softmax(valid_logits, dim=0)
    confidences, predictions = softmaxes.max(0)
    accuracies = torch.eq(predictions, valid_labels)

    bins = torch.linspace(0, 1, n_bins + 1)
    bin_indices = [confidences.gt(bin_lower) * confidences.lt(bin_upper) for bin_lower, bin_upper in zip(bins[:-1], bins[1:])]

    bin_corrects = np.array([torch.mean(accuracies[bin_index].float()) for bin_index in bin_indices])
    bin_scores = np.array([torch.mean(confidences[bin_index].float()) for bin_index in bin_indices])
    bin_corrects = np.nan_to_num(bin_corrects)
    bin_scores = np.nan_to_num(bin_scores)

    ece = _calculate_ece(valid_logits, valid_labels)

    return ece, bin_corrects, bin_scores

def _calculate_ece(logits, labels, n_bins=10):
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    softmaxes = F.softmax(logits, dim=0)
    confidences, predictions = torch.max(softmaxes, 0)
    accuracies = predictions.eq(labels)

    ece = torch.zeros(1, device=logits.device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(
            bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin -
                             accuracy_in_bin) * prop_in_bin
    return ece.item()
