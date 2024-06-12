import argparse
import numpy as np
from scipy.stats import pearsonr

parser = argparse.ArgumentParser()
parser.add_argument('--postprocessor', default='msp', choices=['msp','knn','dice','rankfeat','odin','gradnorm','ash', 'mds'])
parser.add_argument('--metric', default='AUROC', choices=['FPR@95','AUROC','AUPR_IN','AUPR_OUT'])
args = parser.parse_args()

# set method and metric
tgt_method = args.postprocessor
target_metric = args.metric
csv_file = "./results/{}.csv".format(tgt_method)

for target_dataset in ["imagenet21k", "SynIS"]:

    with open(csv_file, 'r') as fp:
        all_result = fp.readlines()
    first_line = all_result[0]
    title = [t.strip() for t in first_line.split(',')]

    if target_dataset=="imagenet21k":
        results = all_result[1:-64]
    elif target_dataset=="SynIS":
        results = all_result[-64:]
    else:
        raise TypeError

    # load results into dict
    value_index = title.index(target_metric)
    output_dict = {}
    for result in results:
        result_info = [t.strip() for t in result.split(',')]
        cov_level = result_info[0][:5]
        value = result_info[value_index]
        if cov_level not in output_dict.keys():
            output_dict[cov_level] = [value]
        else:
            output_dict[cov_level].append(value)

    # print metrics on diverse semantic and covariate shifts
    print("method: {}".format(tgt_method))
    print("target_dataset: {}".format(target_dataset))
    for i in range(8):
        print("\tsem_{}".format(i), end='')
    print()
    all_values = []
    for cov_level in output_dict.keys():
        print(cov_level, end='')
        for value in output_dict[cov_level]:
            print('\t{}'.format(value), end='')
        print()
        all_values.append([float(value) for value in output_dict[cov_level]])

    def cal_sensitivity(mean_array):
        i_list = np.array([i+1 for i in range(mean_array.size)])
        mean_array = mean_array - mean_array.mean()
        i_list = i_list - i_list.mean()
        corr = (mean_array*i_list).sum()
        sensitivity = corr / (i_list*i_list).sum()

        corr_pearsonr, _ = pearsonr(mean_array, i_list)

        return sensitivity, corr_pearsonr

    # print correlation and sensitivity
    all_values = np.array(all_values)
    cov_mean = all_values.sum(axis=1) / np.greater(all_values, 1e-16).sum(axis=1)
    sem_mean = all_values.sum(axis=0) / np.greater(all_values, 1e-16).sum(axis=0)

    sensitivity, corr_pearsonr = cal_sensitivity(sem_mean)
    print('\tcorr\tsens\t\tcorr\tsens')
    print('sem\t{:.2f}\t{:.2f}\t'.format(corr_pearsonr, abs(sensitivity)), end='')
    sensitivity, corr_pearsonr = cal_sensitivity(cov_mean)
    print('cov\t{:.2f}\t{:.2f}'.format(corr_pearsonr, abs(sensitivity)))