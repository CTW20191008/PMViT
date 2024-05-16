import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn

class ExternalSourcePipeline(Pipeline):
    def __init__(self, external_data, batch_size, num_instances, num_threads, device_id):
        super(ExternalSourcePipeline, self).__init__(batch_size, num_threads, device_id, seed=0)
        self.external_data = external_data # ExternalInputIterator
        self.iterator = iter(self.external_data)
        self.images_length = num_instances

        # inputs 对 image 进行读取，因为有 n 个图，所以就构建 n 个读取，每个读取读 连续 batch 的数据
        self.inputs = [ops.ExternalSource() for _ in range(self.images_length)]
        # label 只有 1 个，所以只需要一个
        self.input_label = ops.ExternalSource()
        self.transforms = self.build_transform()

    def build_transform(self):
        res = []
        size_train = self.cfg.DATA.INPUT_SIZE
        resize = ops.Resize(device="gpu", resize_x=size_train[0], resize_y=size_train[1],
                            interp_type=types.INTERP_TRIANGULAR)
        res.append(resize)
        return res

    def define_graph(self):
        # 定义图，数据解码、transform 都在这里
        batch_data = [i() for i in self.inputs]
        self.images = batch_data
        self.labels = self.input_label()

        out = fn.decoders.image(self.images, device="mixed", output_type=types.RGB)
        out = [out_elem.gpu() for out_elem in out]

        for trans in self.transforms:
            out = trans(out)
        # 注意，这里是 *out ，即最后拿到的是一个 list
        # 所以 iter_setup 里也是循环 n 次进行 feed_input
        # 把 images 序列有 list 方式弹出；
        return (*out, self.labels)

    def iter_setup(self):
        try:
            batch_data, labels = self.iterator.next()  # 拿到一个Batch的数据，对应上面的 ExternalInputIterator 的 next 拿到的结果
            # batch_data 中的每个元素都构建一个 feed_input，每个feed_input 操作 batch 个数据
            for i in range(self.images_length):
                self.feed_input(self.images[i], batch_data[i])
            self.feed_input(self.labels, labels)
        except StopIteration:
            self.iterator = iter(self.external_data)
            raise StopIteration