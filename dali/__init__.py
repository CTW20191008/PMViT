from .external_input_iterator import ExternalInputIterator
from .external_source_pipeline import ExternalSourcePipeline
from .custom_dali_genericIterator import CustomDALIGenericIterator
from nvidia.dali.plugin.pytorch import LastBatchPolicy

def build_loader(config, logger):
    model_type = config.MODEL.TYPE
    if model_type == 'pvit':
        batch_size = config.DATA.BATCH_SIZE
        num_instances = 1
        file_path = config.DATA.DATA_PATH
        eii = ExternalInputIterator(file_path, batch_size=batch_size, num_instances=num_instances, shuffled=True)
        pipe = ExternalSourcePipeline(external_data=eii, batch_size=batch_size, num_instances=num_instances, num_threads=0, device_id=0)
        pipe.build()

        # 直接使用自己的iter
        dali_iter = CustomDALIGenericIterator(
            len(eii) // batch_size + 1, num_instances, [pipe], auto_reset=True, dynamic_shape=True, last_batch_padded=True, last_batch_policy=LastBatchPolicy.PARTIAL)
        
        return dali_iter