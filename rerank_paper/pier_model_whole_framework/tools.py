# -*- coding: utf-8 -*-
import datetime
import time
import os

import math
import random

class tick_tock:
    def __init__(self, process_name, verbose=1):
        self.process_name = process_name
        self.verbose = verbose

    def __enter__(self):
        if self.verbose:
            print(("*" * 50 + " {} START!!!! ".format(self.process_name) + "*" * 50))
            self.begin_time = time.time()

    def __exit__(self, type, value, traceback):
        if self.verbose:
            end_time = time.time()
            duration_seconds = end_time - self.begin_time
            duration = str(datetime.timedelta(seconds=duration_seconds))

            print(("#" * 50 + " {} END... time lapsing {}  ".format(self.process_name, duration) + "#" * 50))


class FeatureInfo:
    def __init__(self, feature_info_str):
        self.feature_info_str = feature_info_str

        self.feature_name = "NonFeaName"
        self.feature_size = 0
        self.feature_mask = 1
        self.parse_info_flag = False
        self.part_num = 3

        self._parse_info()

    def _parse_info(self):
        infoList = self.feature_info_str.split()

        if len(infoList) == self.part_num:
            self.feature_name = infoList[0]
            self.feature_size = int(infoList[1])
            self.feature_mask = int(infoList[2])
            self.parse_info_flag = True


def parse_mask_file(feature_mask_file):
    try:
        if not os.path.exists(feature_mask_file):
            print("parse_mask_file fail - file not exists:", feature_mask_file)
            return [], False
        # feature_name_list = []
        feature_mask_list = []
        feature_hold_cnt = 0

        with open(feature_mask_file) as f:
            str_list = f.readlines()

        for i in range(0, len(str_list)):
            str_list[i] = str_list[i].strip('\n').strip()
            if str_list[i] == "":
                continue

            info = FeatureInfo(str_list[i])
            if not info.parse_info_flag:
                print("parse_mask_file fail - parse_info fail:", str_list[i])
                parse_mask_flag = False
                return [], parse_mask_flag

            for j in range(info.feature_size):
                feature_mask_list.append(info.feature_mask)
                if info.feature_mask != 0:
                    feature_hold_cnt += 1
                # if info.feature_size > 1:
                #     feature_name_list.append(info.feature_name + "_" + str(j))
                # else:
                #     feature_name_list.append(info.feature_name)

        parse_mask_flag = True
        return feature_mask_list, parse_mask_flag, feature_hold_cnt
    except Exception as e:
        print("parse_mask_file fail - Exception:", e)
        return [], False

import itertools


# 利用itertools库中的permutations函数,给定一个排列,输出他的全排列
def allPermutation(n):
    permutation = []
    # 首先需要初始化一个1-n的排列
    for i in range(n):
        permutation.append(i)
    # itertools.permutations返回的只是一个对象,需要将其转化成list
    # 每一种排列情况以元组类型存储
    all_permutation = list(itertools.permutations(permutation))
    return all_permutation


def random_vector():
    print([[random.random() for x in list(range(0,8))] for y in list(range(0,5))])




if __name__ == "__main__":
    # feature_mask_list, parse_feature_mask_flag, feature_hold_cnt = parse_mask_file("feature_mask")
    # print(feature_mask_list)
    # print(len(feature_mask_list))
    # print(parse_feature_mask_flag)
    # print(feature_hold_cnt)
    # print(allPermutation(5))
    random_vector()
