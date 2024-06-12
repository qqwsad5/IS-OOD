import numpy as np
import pandas as pd
from sklearn import metrics

def compute_all_metrics(conf, label, pred):
    np.set_printoptions(precision=3)
    recall = 0.95
    auroc, aupr_in, aupr_out, fpr = auc_and_fpr_recall(conf, label, recall)

    results = [fpr, auroc, aupr_in, aupr_out]

    return results

# auc
def auc_and_fpr_recall(conf, label, tpr_th):
    # following convention in ML we treat OOD as positive
    ood_indicator = np.zeros_like(label)
    ood_indicator[label == -1] = 1

    # in the postprocessor we assume ID samples will have larger
    # "conf" values than OOD samples
    # therefore here we need to negate the "conf" values
    fpr_list, tpr_list, thresholds = metrics.roc_curve(ood_indicator, -conf)
    fpr = fpr_list[np.argmax(tpr_list >= tpr_th)]

    precision_in, recall_in, thresholds_in \
        = metrics.precision_recall_curve(1 - ood_indicator, conf)

    precision_out, recall_out, thresholds_out \
        = metrics.precision_recall_curve(ood_indicator, -conf)

    auroc = metrics.auc(fpr_list, tpr_list)
    aupr_in = metrics.auc(recall_in, precision_in)
    aupr_out = metrics.auc(recall_out, precision_out)

    return auroc, aupr_in, aupr_out, fpr


def print_metrics(metrics):
    [fpr, auroc, aupr_in, aupr_out] = metrics

    # print ood metric results
    print('FPR@95: {:.2f}, AUROC: {:.2f}'.format(100 * fpr, 100 * auroc),
        end=' ',
        flush=True)
    print('AUPR_IN: {:.2f}, AUPR_OUT: {:.2f}'.format(
        100 * aupr_in, 100 * aupr_out),
        flush=True)
    print(u'\u2500' * 70, flush=True)
    print('', flush=True)
    

def eval_ood(net, postprocessor, dataloader_dict, progress: bool = True):

    # inference on imagenet1k
    print(f'Performing inference on ID test set...',
            flush=True)
    id_pred, id_conf, id_gt = postprocessor.inference(
        net, dataloader_dict['id']['test'], progress)

    # evaluate on imagenet21k
    print(f'Processing split level ood detection ...', flush=True)
    metrics_list = []
    for dataset_name, ood_dl in dataloader_dict['ood'].items():
        if ood_dl is None:
            ood_metrics = [0., 0., 0., 0.]
            metrics_list.append(ood_metrics)
            continue
        print(f'Performing inference on {dataset_name} dataset...',
                flush=True)
        ood_pred, ood_conf, ood_gt = postprocessor.inference(
            net, ood_dl, progress)

        ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood
        pred = np.concatenate([id_pred, ood_pred])
        conf = np.concatenate([id_conf, ood_conf])
        label = np.concatenate([id_gt, ood_gt])

        print(f'Computing metrics on {dataset_name} dataset...')
        ood_metrics = compute_all_metrics(conf, label, pred)
        metrics_list.append(ood_metrics)
        print_metrics(ood_metrics)

    # evaluate on SynIS
    print(f'Processing split level detection on SynIS ...', flush=True)
    for dataset_name, SynIS_dl in dataloader_dict['SynIS'].items():
        if SynIS_dl is None:
            SynIS_metrics = [0., 0., 0., 0.]
            metrics_list.append(SynIS_metrics)
            continue
        print(f'Performing inference on {dataset_name} dataset...',
                flush=True)
        SynIS_pred, SynIS_conf, SynIS_gt = postprocessor.inference(
            net, SynIS_dl, progress)

        SynIS_gt = -1 * np.ones_like(SynIS_gt)
        pred = np.concatenate([id_pred, SynIS_pred])
        conf = np.concatenate([id_conf, SynIS_conf])
        label = np.concatenate([id_gt, SynIS_gt])

        print(f'Computing metrics on {dataset_name} dataset...')
        SynIS_metrics = compute_all_metrics(conf, label, pred)
        metrics_list.append(SynIS_metrics)
        print_metrics(SynIS_metrics)

    all_metrics = np.array(metrics_list) * 100

    metrics = pd.DataFrame(
        all_metrics,
        index=list(dataloader_dict['ood'].keys()) + list(dataloader_dict['SynIS'].keys()),
        columns=['FPR@95', 'AUROC', 'AUPR_IN', 'AUPR_OUT'],
    )

    return metrics

