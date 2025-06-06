import torch
from torch import nn
from model_zoo import resnet

def generate_model(MODEL_TYPE, MODEL_DEPTH,
                   INPUT_W, INPUT_H, INPUT_D, NUM_CLASSES,
                   NEW_LAYER_NAMES, PRETRAIN_PATH,
                   RESNET_SHORTCUT, NO_CUDA, GPU_ID,
                   model_phase='train'):
    assert MODEL_DEPTH in [
        'resnet'
    ]

    if MODEL_TYPE == 'resnet':
        assert MODEL_DEPTH in [10, 18, 34, 50, 101, 152, 200]
        
        if MODEL_DEPTH == 10:
            model = resnet.resnet10(
                sample_input_W=INPUT_W,
                sample_input_H=INPUT_H,
                sample_input_D=INPUT_D,
                shortcut_type=RESNET_SHORTCUT,
                no_cuda=NO_CUDA,
                num_seg_classes=NUM_CLASSES)
        elif MODEL_DEPTH == 18:
            model = resnet.resnet18(
                sample_input_W=INPUT_W,
                sample_input_H=INPUT_H,
                sample_input_D=INPUT_D,
                shortcut_type=RESNET_SHORTCUT,
                no_cuda=NO_CUDA,
                num_seg_classes=NUM_CLASSES)
        elif MODEL_DEPTH == 34:
            model = resnet.resnet34(
                sample_input_W=INPUT_W,
                sample_input_H=INPUT_H,
                sample_input_D=INPUT_D,
                shortcut_type=RESNET_SHORTCUT,
                no_cuda=NO_CUDA,
                num_seg_classes=NUM_CLASSES)
        elif MODEL_DEPTH == 50:
            model = resnet.resnet50(
                sample_input_W=INPUT_W,
                sample_input_H=INPUT_H,
                sample_input_D=INPUT_D,
                shortcut_type=RESNET_SHORTCUT,
                no_cuda=NO_CUDA,
                num_seg_classes=NUM_CLASSES)
        elif MODEL_DEPTH == 101:
            model = resnet.resnet101(
                sample_input_W=INPUT_W,
                sample_input_H=INPUT_H,
                sample_input_D=INPUT_D,
                shortcut_type=RESNET_SHORTCUT,
                no_cuda=NO_CUDA,
                num_seg_classes=NUM_CLASSES)
        elif MODEL_DEPTH == 152:
            model = resnet.resnet152(
                sample_input_W=INPUT_W,
                sample_input_H=INPUT_H,
                sample_input_D=INPUT_D,
                shortcut_type=RESNET_SHORTCUT,
                no_cuda=NO_CUDA,
                num_seg_classes=NUM_CLASSES)
        elif MODEL_DEPTH == 200:
            model = resnet.resnet200(
                sample_input_W=INPUT_W,
                sample_input_H=INPUT_H,
                sample_input_D=INPUT_D,
                shortcut_type=RESNET_SHORTCUT,
                no_cuda=NO_CUDA,
                num_seg_classes=NUM_CLASSES)
    
    if not NO_CUDA:
        if len(GPU_ID) > 1:
            model = model.cuda() 
            model = nn.DataParallel(model, device_ids=GPU_ID)
            net_dict = model.state_dict() 
        else:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"]=str(GPU_ID[0])
            model = model.cuda() 
            model = nn.DataParallel(model, device_ids=None)
            net_dict = model.state_dict()
    else:
        net_dict = model.state_dict()
    
    # load pretrain
    if model_phase != 'test' and PRETRAIN_PATH:
        print ('loading pretrained model {}'.format(PRETRAIN_PATH))
        pretrain = torch.load(PRETRAIN_PATH)
        pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
         
        net_dict.update(pretrain_dict)
        model.load_state_dict(net_dict)

        new_parameters = [] 
        for pname, p in model.named_parameters():
            for layer_name in NEW_LAYER_NAMES:
                if pname.find(layer_name) >= 0:
                    new_parameters.append(p)
                    break

        new_parameters_id = list(map(id, new_parameters))
        base_parameters = list(filter(lambda p: id(p) not in new_parameters_id, model.parameters()))
        parameters = {'base_parameters': base_parameters, 
                      'new_parameters': new_parameters}

        return model, parameters

    return model, model.parameters()
