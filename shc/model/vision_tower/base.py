import os

import torch
import torch.nn as nn

from transformers import PreTrainedModel
import torch.nn.functional as F
# from tinyllava.utils.data_utils import get_value_from_kwargs

def get_value_from_kwargs(kwargs, name):
    if name in kwargs:
        return kwargs.pop(name)
    else:
        return None

class VisionTower(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._vision_tower = None
        self._image_processor = None
        self.config = cfg
        # self.fusion_1 = nn.ModuleList([
        #     nn.Linear(self.config.hidden_size, self.config.hidden_size//2)
        #     for _ in range(4)
        # ])
        # self.fusion_2 = nn.ModuleList([
        #     nn.Linear(self.config.hidden_size, self.config.hidden_size//2)
        #     for _ in range(4)
        # ])
        # self.fusion_3 = nn.ModuleList([
        #     nn.Linear(self.config.hidden_size//2, self.config.hidden_size)
        #     for _ in range(4)
        # ])

    def load_model(self, vision_tower_name, **kwargs):
        self._load_model(vision_tower_name, **kwargs)
        self._vision_tower.requires_grad_(False)

        
    def _load_model(self, vision_tower_name, **kwargs):
        pretrained_vision_tower_path = get_value_from_kwargs(kwargs, 'pretrained_vision_tower_path')
        if isinstance(self._vision_tower, PreTrainedModel): # hf model
            if pretrained_vision_tower_path is not None:
                vision_tower_name = pretrained_vision_tower_path
            self._vision_tower = self._vision_tower.from_pretrained(vision_tower_name, **kwargs)      
        else: # nn.Module
            if pretrained_vision_tower_path is not None:
                vision_tower_weights = torch.load(os.path.join(pretrained_vision_tower_path, 'pytorch_model.bin'), map_location='cpu')
                def get_w(weights, keyword):
                    return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
                self._vision_tower.load_state_dict(vision_tower_weights)

        print("Loading vision tower from ", vision_tower_name)
        


    def forward(self, x, **kwargs):
        # image_features = self._vision_tower(x, output_hidden_states=True)
        # image_features = image_features.hidden_states[kwargs.get('vision_feature_layer', -2)]
        #
        # if kwargs.get('vision_feature_select_strategy', 'patch') == 'patch':
        #     image_features = image_features[:, 1:]
        # elif kwargs.get('vision_feature_select_strategy', 'patch') == 'cls_patch':
        #     image_features = image_features
        # else:
        #     raise ValueError(f"Unexpected select feature: {kwargs.get('vision_feature_select_strategy')}")
        #
        # return image_features
        image_features_1 = []
        image_features_2 = []
        image_features_3 = []
        image_features_4 = []
        image_features_mutli=[]
        image_features = self._vision_tower(x, output_hidden_states=True)
        image_fusion = image_features
        img_mid=image_features
        
        
        #二分层
        # for i in range(0, 13):
        #     image_features_1.append(image_fusion.hidden_states[i][:, :].to(x.dtype))
        # image_features_1 = torch.stack(image_features_1, dim=0)
        # image_features_1 = torch.sum(image_features_1, dim=0) / 13
        # image_features_mutli.append(image_features_1.unsqueeze(1))
        # for i in range(13, 26):
        #     image_features_2.append(image_fusion.hidden_states[i][:, :].to(x.dtype))
        # image_features_2 = torch.stack(image_features_2, dim=0)
        # image_features_2 = torch.sum(image_features_2, dim=0) / 13
        # image_features_mutli.append(image_features_2.unsqueeze(1))
        #
        # image_features_3 = image_features.hidden_states[kwargs.get('vision_feature_layer', -2)]
        # image_features_mutli.append(image_features_3.unsqueeze(1))
        # image_features = image_features.hidden_states[kwargs.get('vision_feature_layer', -1)]
        # if kwargs.get('vision_feature_select_strategy', 'patch') == 'patch':
        #     image_features = image_features[:, 1:]
        # elif kwargs.get('vision_feature_select_strategy', 'patch') == 'cls_patch':
        #     image_features = image_features
        # else:
        #     raise ValueError(f"Unexpected select feature: {kwargs.get('vision_feature_select_strategy')}")
        # return image_features, image_features_mutli

        


        #三分层
        # image_features_l2 = image_features.hidden_states[kwargs.get('vision_feature_layer', -2)]
        # image_features_mutli.append(image_features_l2.unsqueeze(1))
        # for i in range(0, 12):
        #     image_features_1.append(image_fusion.hidden_states[i][:, :].to(x.dtype))
        # image_features_1 = torch.stack(image_features_1, dim=0)
        # image_features_1 = torch.sum(image_features_1, dim=0) / 12
        # image_features_mutli.append(image_features_1.unsqueeze(1))
        # for i in range(6, 21):
        #     image_features_2.append(image_fusion.hidden_states[i][:, :].to(x.dtype))
        # image_features_2 = torch.stack(image_features_2, dim=0)
        # image_features_2 = torch.sum(image_features_2, dim=0) / 15
        # image_features_mutli.append(image_features_2.unsqueeze(1))
        # for i in range(18, 27):
        #     image_features_3.append(image_fusion.hidden_states[i][:, :].to(x.dtype))
        # image_features_3 = torch.stack(image_features_3, dim=0)
        # image_features_3 = torch.sum(image_features_3, dim=0) / 9
        # image_features_mutli.append(image_features_3.unsqueeze(1))
        # image_features = image_features.hidden_states[kwargs.get('vision_feature_layer', -1)]
        
        # if kwargs.get('vision_feature_select_strategy', 'patch') == 'patch':
        #     image_features = image_features[:, 1:]
        # elif kwargs.get('vision_feature_select_strategy', 'patch') == 'cls_patch':
        #     image_features = image_features
        # else:
        #     raise ValueError(f"Unexpected select feature: {kwargs.get('vision_feature_select_strategy')}")
        
        # return image_features,image_features_mutli


        
        # #四分层
        # image_features_l2 = image_features.hidden_states[kwargs.get('vision_feature_layer', -2)]
        # image_features_mutli.append(image_features_l2.unsqueeze(1))
        # for i in range(18, 27):
        #     image_features_1.append(image_fusion.hidden_states[i][:, :].to(x.dtype))
        # image_features_1 = torch.stack(image_features_1, dim=0)
        # image_features_1 = torch.sum(image_features_1, dim=0) / 9
        # image_features_mutli.append(image_features_1.unsqueeze(1))
        # for i in range(11, 24):
        #     image_features_2.append(image_fusion.hidden_states[i][:, :].to(x.dtype))
        # image_features_2 = torch.stack(image_features_2, dim=0)
        # image_features_2 = torch.sum(image_features_2, dim=0) / 13
        # image_features_mutli.append(image_features_2.unsqueeze(1))
        # for i in range(4, 17):
        #     image_features_3.append(image_fusion.hidden_states[i][:, :].to(x.dtype))
        # image_features_3 = torch.stack(image_features_3, dim=0)
        # image_features_3 = torch.sum(image_features_3, dim=0) / 13
        # image_features_mutli.append(image_features_3.unsqueeze(1))
        # for i in range(0, 10):
        #     image_features_4.append(image_fusion.hidden_states[i][:, :].to(x.dtype))
        # image_features_4 = torch.stack(image_features_4, dim=0)
        # image_features_4 = torch.sum(image_features_4, dim=0) / 10
        # image_features_mutli.append(image_features_4.unsqueeze(1))
        # image_features = image_features.hidden_states[kwargs.get('vision_feature_layer', -1)]
        
        # if kwargs.get('vision_feature_select_strategy', 'patch') == 'patch':
        #     image_features = image_features[:, 1:]
        # elif kwargs.get('vision_feature_select_strategy', 'patch') == 'cls_patch':
        #     image_features = image_features
        # else:
        #     raise ValueError(f"Unexpected select feature: {kwargs.get('vision_feature_select_strategy')}")
        # return image_features,image_features_mutli

        


    @property
    def vision_tower(self):
        return self._vision_tower
        
    @vision_tower.setter
    def vision_tower(self, vision_tower):
        self._vision_tower = vision_tower
        
    
