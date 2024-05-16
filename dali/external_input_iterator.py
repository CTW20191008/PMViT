import json
import numpy as np
from random import shuffle

import torch


class ExternalInputIterator(object):
    def __init__(self, file_path, batch_size, num_instances=1, shuffled=False):
        # 这一块其实与 dateset 都比较像
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.shuffled = shuffled
        self.img_seq_length = num_instances

        with open(file_path) as f:
            images_dict = json.load(f)

        self.images_dict = images_dict

        self.list_of_pids = list(images_dict.keys())
        self._num_classes = len(self.list_of_pids)
        self.all_indexs = list(range(len(self.list_of_pids)))
        self.n = self.__len__()

    def __iter__(self):
        self.i = 0
        if self.shuffled:
            shuffle(self.all_indexs)
        return self

    def __len__(self):
        return len(self.all_indexs)

    @staticmethod
    def image_open(path):
        return np.fromfile(path, dtype=np.uint8)

    def __next__(self):
        # 如果溢出了，就终止
        if self.i >= self.n:
            self.__iter__()
            raise StopIteration

        batch_images = []
        batch_labels = []

        leave_num = self.n - self.i
        current_batch_size = min(
            self.batch_size, leave_num)  # 保证最后一个 batch 不溢出
        for _ in range(current_batch_size):
            tmp_index = self.all_indexs[self.i]
            p_id = self.list_of_pids[tmp_index]
            images_dict = self.images_dict[p_id]

            images = images_dict["images"]  # 取 n 个
            # 分别读取为 numpy，也可以是 batch
            images = list(map(self.image_open, images))
            # 这一块都比较像，但是不作 transform 处理
            label = images_dict["label"]

            batch_images.append(images)
            batch_labels.append(torch.tensor(label))

            self.i += 1

        """
        这一块非常重要
        把 上面的 images ，换为 以 batch 为计量单位的形式;
        label 不用换，因为他本身只有一个，所以他的 batch_labels 就是 batch * 1
        但是 batch_images 是 batch * n (n = num_instance)，即 n 个图是连续的
        所以必须把 batch * n 转换为 (batch * 1) * n，即 一个图的 batch 个数据时连续的
        这一点非常重要（再次强调）
        因为 dali 取数据时是 取 一个 batch，所以一个图的 batch 个数据必须连续，这样才能对应上
        """
        batch_data = []
        for ins_i in range(self.num_instances):
            elem = []
            for batch_idx in range(current_batch_size):
                elem.append(batch_images[batch_idx][ins_i])
            batch_data.append(elem)
        # 其实这块也可以通过 tensor 的 permute 实现？我之前没有注意，大家有兴趣可以试试

        return batch_data, batch_labels

    next = __next__
    len = __len__
