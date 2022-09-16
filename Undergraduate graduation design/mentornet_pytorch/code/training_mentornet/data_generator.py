"""Generates training data for learning/updating MentorNet."""

import csv
import itertools
import os
import pickle
import models
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='[data_generator]')

parser.add_argument('--outdir', type=str, default='', 
                    help='Directory to the save training data.')
parser.add_argument('--vstar_fn', default='', help='the vstar function to use.')
parser.add_argument('vstar_gamma', default='', help='the hyper_parameter for the vstar_fn')
parser.add_argument('--sample_size', type=int, default=100000, 
                    help='size to of the total generated data set.')
parser.add_argument('--input_csv_filename', default='')
parser.add_argument('--tr_rate', type=int, default=0.8)

args = parser.parse_args()

def generate_pretrain_defined(vstar_fn, outdir, sample_size):
    """Generates a trainable dataset given a vstar_fn.

    Args:
        vstar_fn: the name of the variable star function to use.
        outdir: directory to save the training data.
        sample_size: size of the sample.
    """
    batch_l = np.concatenate((np.arange(0, 10, 0.1), np.arange(10, 30, 1)))
    batch_diff = np.arange(-5, 5, 0.1)
    batch_y = np.array([0])
    batch_e = np.arange(0,100,1)

    data = []

    for t in itertools.product(batch_l, batch_diff, batch_y, batch_e):
        # 类似笛卡尔积，四个都遍历
        data.append(t)
    data = np.array(data)

    v = vstar_fn(data)
    v = v.reshape([-1, 1])
    data = np.hstack((data, v))

    perm = np.arange(data.shape[0])
    np.random.shuffle(perm)
    data = data[perm[0:min(sample_size, len(perm))], ]

    tr_size = int(data.shape[0] * args.tr_rate)
    tr = data[0:tr_size]
    ts = data[(tr_size+1):data.shape[0]]

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    print('training_shape={} test_shape={}'.format(tr.shape, ts.shape))

    with open(os.path.join(outdir, 'tr.p'), 'wb') as outfile:
        pickle.dump(tr, outfile)

    with open(os.path.join(outdir, 'ts.p'), 'wb') as outfile:
        pickle.dump(ts, outfile)


def generate_data_driven(input_csv_filename,
                         outdir,
                         percentile_range='40,50,60,75,80,90'):
    """Generates a data-driven trainable dataset, given a CSV.

  Refer to README.md for details on how to format the CSV.

  Args:
    input_csv_filename: the path of the CSV file. The csv file format
      0: epoch_percentage
      1: noisy label
      2: clean label
      3: loss
    outdir: directory to save the training data.
    percentile_range: the percentiles used to compute the moving average.
    """
    raw = read_from_csv(input_csv_filename)

    raw = np.array(raw.values())
    dataset_name = os.path.splitext(os.path.basename(input_csv_filename))[0]

    percentile_range = percentile_range.split(',')
    percentile_range = [int(x) for x in percentile_range]

    for percentile in percentile_range:
        percentile = int(percentile)
        p_perncentile = np.percentile(raw[:, 3], percentile)

        v_star = np.float32(raw[:, 1] == raw[:, 2])

        l = raw[:, 3]
        diff = raw[:, 3] - p_perncentile
        # label not used in the current version.
        y = np.array([0] * len(v_star))
        epoch_percentage = raw[:, 0]

        data = np.vstack((l, diff, y, epoch_percentage, v_star))
        data = np.transpose(data)

        perm = np.arange(data.shape[0])
        np.random.shuffle(perm)
        data = data[perm,]

        tr_size = int(data.shape[0] * 0.8)

        tr = data[0:tr_size]
        ts = data[(tr_size + 1):data.shape[0]]

        cur_outdir = os.path.join(
            outdir, '{}_percentile_{}'.format(dataset_name, percentile))
        if not os.path.exists(cur_outdir):
            os.makedirs(cur_outdir)

        print('training_shape={} test_shape={}'.format(tr.shape, ts.shape))
        print(cur_outdir)
        with open(os.path.join(cur_outdir, 'tr.p'), 'wb') as outfile:
            pickle.dump(tr, outfile)

        with open(os.path.join(cur_outdir, 'ts.p'), 'wb') as outfile:
            pickle.dump(ts, outfile)


def read_from_csv(input_csv_file):
    """Reads Data from an input CSV file.

  Args:
    input_csv_file: the path of the CSV file.

  Returns:
    a numpy array with different data at each index:
    """
    data = {}
    with open(input_csv_file, 'r') as csv_file_in:
        reader = csv.reader(csv_file_in)
    for row in reader:
        for (_, cell) in enumerate(row):
            rdata = cell.strip().split(' ')
            rid = rdata[0]
            rdata = [float(t) for t in rdata[1:]]
            data[rid] = rdata
    csv_file_in.close()
    return data

def main():
    if args.vstar_fn == 'data_driven':
        generate_data_driven(args.input_csv_filename, args.outdir)
    elif args.vstar_fn in dir(models):
        generate_pretrain_defined(
            getattr(models, args.vstar_fn), args.outdir, args.sample_size
        )
    else:
        print(f'{args.vstar_fn}is not defined in models.py')

if __name__ == '__main__':
    main()