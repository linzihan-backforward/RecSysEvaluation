"""
Put this file into the `RecBole/` directory.

This file is used to search best hyper parameters one by one
"""

import argparse
import numpy as np

from recbole.quick_start import objective_function
from tqdm import tqdm

stop_step = 2
rand_seed = 2020


def hyper_space(file):
    space = {}
    random_param = {}
    with open(file, 'r') as fp:
        for line in fp:
            para_list = line.strip().split(' ')
            if len(para_list) < 3:
                continue
            para_name, para_type, para_value = para_list[0], para_list[1], "".join(para_list[2:])
            if para_type == 'choice':
                para_value = eval(para_value)
                space[para_name] = para_value
            else:
                raise Exception
            random_param[para_name] = np.random.choice(para_value)

    return space, random_param


if __name__ == '__main__':
    np.random.seed(rand_seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='BPR', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='ml-100k', help='name of datasets')
    parser.add_argument('--config_files', type=str, default=None, help='config files')
    parser.add_argument('--params_file', type=str, default=None, help='params file')

    args, _ = parser.parse_known_args()
    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    searching_space, param_value = hyper_space(args.params_file)

    for key in searching_space.keys():
        print("Searching {}".format(key))
        best_index = -1
        best_valid_result = -np.inf
        iter_data = tqdm(
            enumerate(searching_space[key]),
            total=len(searching_space[key]),
            desc=f"{key}",
        )
        for idx, value in iter_data:
            param_value[key] = value
            print("running parameters:", param_value)
            result = objective_function(config_file_list=config_file_list, config_dict=param_value)
            print("valid result:", result['best_valid_result'], "\n")
            if result['best_valid_score'] > best_valid_result:
                best_valid_result = result['best_valid_score']
                test_result = result['test_result']
                iter_data.set_postfix(valid_result=best_valid_result)
                best_index = idx

            if idx - best_index >= stop_step:
                print("{} reach stop step".format(key))
                break

        param_value[key] = searching_space[key][best_index]
        print("{} search done. Best parameter is {} \n".format(key, param_value[key]))

    print("Done. Best valid result is {}. Best parameters are {}. "
          "Test result is {}".format(best_valid_result, param_value, test_result))
