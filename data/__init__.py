from .data_simmim import build_loader_simmim
from .data_finetune import build_loader_finetune
# from .data_pvit import build_load_pvit, build_load_pvit_single
from .data_pvit_plus import build_load_pvit, build_load_pvit_single
from .data_cvit import build_load_cvit

def build_loader(config, logger, is_pretrain):
    model_type = config.MODEL.TYPE
    if is_pretrain:
        if model_type == 'pvit':
            if config.TRAIN.USE_DISTRIBUTED:
                return build_load_pvit(config, logger)
            else:
                return build_load_pvit_single(config, logger)
        elif model_type == 'cvit':
            return build_load_cvit(config, logger)
        else:
            return build_loader_simmim(config, logger)
    else:
        return build_loader_finetune(config, logger)
    # if model_type == 'pvit':
    #     if is_pretrain:
    #         if config.TRAIN.USE_DISTRIBUTED:
    #             return build_load_pvit(config, logger)
    #         else:
    #             return build_load_pvit_single(config, logger)
    #     else:
    #         return build_loader_finetune(config, logger)
    # elif model_type == 'cvit':
    #     if is_pretrain:
    #         return build_load_cvit(config, logger)
    #     else:
    #         return build_loader_finetune(config, logger)
    # else:
    #     if is_pretrain:
    #         return build_loader_simmim(config, logger)
    #     else:
    #         return build_loader_finetune(config, logger)