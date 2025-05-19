from models.P2AT import P2AT
#from models.CSNet import CSNet

def build_model(model_type, mode, name_or_cfg, num_classes=None):
    """
    Build a model for training or prediction.

    Args:
        model_type (str): 'P2AT' or 'CSNet'
        mode (str): 'train' or 'pred'
        name_or_cfg: model name (str) for pred, or cfg object for train
        num_classes (int, optional): Number of classes (required for pred)

    Returns:
        model: Instantiated model
    """
    if model_type == 'P2AT':
        ModelClass = P2AT
    #elif model_type == 'CSNet':
    #    ModelClass = CSNet
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if mode == 'train':
        name = name_or_cfg.MODEL.NAME
        num_classes = name_or_cfg.DATASET.NUM_CLASSES
        is_train = True
    else:
        name = name_or_cfg
        is_train = False

    if 's' in name:
        backbone = "resnet18"
    elif 'm' in name:
        backbone = "resnet34"
    else:
        backbone = "resnet50"

    return ModelClass(backbone=backbone, pretrained=True, num_classes=num_classes, is_train=is_train)