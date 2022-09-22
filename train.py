# @Time   : 2020/7/20
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE
# @Time   : 2020/10/3, 2020/10/1
# @Author : Yupeng Hou, Zihan Lin
# @Email  : houyupeng@ruc.edu.cn, zhlin@ruc.edu.cn
import os

import argparse

from quick_start import run_recbole, load_data_and_model

parameter_dict = {
    # 'neg_sampling': "{'uniform':1}",
    'neg_sampling': None,
    'loss_type': 'CE',
    'topk': [5, 10, 20, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'valid_metric': 'Recall@20',
    'stopping_step': 10,
    'seed': 2000,
    'user_inter_num_interval': '[5,inf)',
    'item_inter_num_interval': '[5,inf)',
    'eval_batch_size': 1024,
    'train_batch_size': 128,
    'MAX_ITEM_LIST_LENGTH': 100,
    # 'load_col': '{inter: "[user_id, item_id, timestamp, item_length, item_id_list, timestamp_list]"}',
    # 'USER_ID_FIELD': 'session_id',
    'gpu_id': 0,
    # 'split': {'RS': [0.9, 0.05, 0.05]},
    # 'benchmark_filename': ['part1', 'part2', 'part3']
}

if __name__ == '__main__':
    # print('dropout=0.5')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='CCLSRec', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='wechat', help='name of datasets')
    parser.add_argument('--config_files', type=str, default=None, help='config files')

    args, _ = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    run_recbole(model=args.model, dataset=args.dataset, config_file_list=config_file_list,
                config_dict=parameter_dict)
