import os
import time

import torch
import torch.distributed as dist

from detection import utils
from detection.data.evaluations import coco_evaluation, voc_evaluation
from detection.data.transforms import de_normalize
from detection.utils.dist_utils import is_main_process, all_gather, get_world_size


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    return predictions


def evaluation(model, data_loaders, device, types=('coco',), output_dir='./evaluations/', iteration=None, viz=False):
    if not isinstance(data_loaders, (list, tuple)):
        data_loaders = (data_loaders,)
    results = dict()
    for data_loader in data_loaders:
        dataset = data_loader.dataset
        _output_dir = os.path.join(output_dir, 'evaluations', dataset.dataset_name)
        os.makedirs(_output_dir, exist_ok=True)
        result = do_evaluation(model, data_loader, device, types=types, output_dir=_output_dir, iteration=iteration, viz=viz)
        results[dataset.dataset_name] = result
    return results


@torch.no_grad()
def do_evaluation(model, data_loader, device, types, output_dir, iteration=None, viz=False):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    dataset = data_loader.dataset
    header = 'Testing {}:'.format(dataset.dataset_name)
    results_dict = {}
    for images, img_metas, targets in metric_logger.log_every(data_loader, 10, header):
        assert len(targets) == 1
        images = images.to(device)

        model_time = time.time()
        boxes, scores, labels = model(images, img_metas)[0]
        model_time = time.time() - model_time

        img_meta = img_metas[0]
        scale_factor = img_meta['scale_factor']
        img_info = img_meta['img_info']

        if viz:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            plt.switch_backend('TKAgg')
            image = de_normalize(images[0], img_meta)
            plt.subplot(122)
            plt.imshow(image)
            plt.title('Predict')
            for i, ((x1, y1, x2, y2), label) in enumerate(zip(boxes.tolist(), labels.tolist())):
                if scores[i] > 0.65:
                    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, facecolor='none', edgecolor='g')
                    category_id = dataset.label2cat[label]
                    plt.text(x1, y1, '{}:{:.2f}'.format(dataset.CLASSES[category_id], scores[i]), color='r')
                    plt.gca().add_patch(rect)

            plt.subplot(121)
            plt.imshow(image)
            plt.title('GT')
            for i, ((x1, y1, x2, y2), label) in enumerate(zip(targets[0]['boxes'].tolist(), targets[0]['labels'].tolist())):
                category_id = dataset.label2cat[label]
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, facecolor='none', edgecolor='g')
                plt.text(x1, y1, '{}'.format(dataset.CLASSES[category_id]))
                plt.gca().add_patch(rect)
            plt.show()

        boxes /= scale_factor
        boxes = boxes.tolist()
        labels = labels.tolist()
        labels = [dataset.label2cat[label] for label in labels]
        scores = scores.tolist()

        results_dict.update({
            img_info['id']: (boxes, scores, labels)
        })
        metric_logger.update(model_time=model_time)

    if get_world_size() > 1:
        dist.barrier()

    predictions = _accumulate_predictions_from_multiple_gpus(results_dict)
    if not is_main_process():
        return {}
    results = {}
    if 'coco' in types:
        result = coco_evaluation(dataset, predictions, output_dir, iteration=iteration)
        results.update(result)
    if 'voc' in types:
        result = voc_evaluation(dataset, predictions, output_dir, iteration=iteration, use_07_metric=False)
        results.update(result)
    return results
