from nvidia.dali.plugin.pytorch import DALIGenericIterator
import torch

class CustomDALIGenericIterator(DALIGenericIterator):
    def __init__(self, length, num_instances, pipelines, **argw):
        self._len = length # dataloader 的长度
        self.num_instances = num_instances
        output_map = [f"image_{i}" for i in range(num_instances)]
        super().__init__(pipelines, output_map, **argw)

    def __next__(self):
        batch = super().__next__()
        return self.parse_batch(batch)

    def __len__(self):
        return self._len

    def parse_batch(self, batch):
        images = [batch[f"images_{i}"] for i in range(self.num_instances)]  # bs * n * 3 * h * w
        images = torch.stack(images)
        labels = batch["labels"]  # bs * 1
        return {"images": images, "labels": labels}