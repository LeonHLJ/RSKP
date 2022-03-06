import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import interp1d


def str2ind(categoryname, classlist):
    return [i for i in range(len(classlist)) if categoryname == classlist[i].decode('utf-8')][0]


def strlist2indlist(strlist, classlist):
    return [str2ind(s, classlist) for s in strlist]


def strlist2multihot(strlist, classlist):
    return np.sum(np.eye(len(classlist))[strlist2indlist(strlist, classlist)], axis=0)


def idx2multihot(id_list, num_class):
    return np.sum(np.eye(num_class)[id_list], axis=0)


def random_choose(v_len, num_seg):
    start_ind = np.random.randint(0, v_len - num_seg)
    random_p = np.arange(start_ind, start_ind + num_seg)
    return random_p.astype(int)


def random_perturb(v_len, num_seg):
    random_p = np.arange(num_seg) * v_len / num_seg
    for i in range(num_seg):
        if i < num_seg - 1:
            if int(random_p[i]) != int(random_p[i + 1]):
                random_p[i] = np.random.choice(range(int(random_p[i]), int(random_p[i + 1]) + 1))
            else:
                random_p[i] = int(random_p[i])
        else:
            if int(random_p[i]) < v_len - 1:
                random_p[i] = np.random.choice(range(int(random_p[i]), v_len))
            else:
                random_p[i] = int(random_p[i])
    return random_p.astype(int)


def uniform_sampling(v_len, num_seg):
    u_sample = np.arange(num_seg) * v_len / num_seg
    u_sample = np.floor(u_sample)
    return u_sample.astype(int)

def write_results_to_eval_file(args, dmap, avg, itr1, itr2):
    file_folder = './ckpt/' + args.dataset_name + '/eval/'
    file_name = args.dataset_name + '-results.log'
    fid = open(file_folder + file_name, 'a+')
    string_to_write = str(itr1)
    string_to_write += ' ' + str(itr2)
    for item in dmap:
        string_to_write += ' ' + '%.2f' % item
    string_to_write += ' ' + '%.2f' % avg    
    fid.write(string_to_write + '\n')
    fid.close()


def write_results_to_file(args, dmap, avg, cmap, itr):
    file_folder = './ckpt/' + args.dataset_name + '/' + str(args.model_id) + '/'
    file_name = args.dataset_name + '-results.log'
    fid = open(file_folder + file_name, 'a+')
    string_to_write = str(itr)
    for item in dmap:
        string_to_write += ' ' + '%.2f' % item
    string_to_write += ' ' + '%.2f' % avg    
    string_to_write += ' ' + '%.2f' % cmap
    fid.write(string_to_write + '\n')
    fid.close()

def write_settings_to_file(args):
    file_folder = './ckpt/' + args.dataset_name + '/' + str(args.model_id) + '/'
    file_name = args.dataset_name + '-results.log'
    fid = open(file_folder + file_name, 'a+')
    string_to_write = '#' * 80 + '\n'
    for arg in vars(args):
        string_to_write += str(arg) + ': ' + str(getattr(args, arg)) + '\n'
    string_to_write += '*' * 80 + '\n'
    fid.write(string_to_write)
    fid.close()