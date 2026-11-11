import torch
import torch.nn as nn
import torch.nn.functional as F




    
    

    
class Location_Net_stage_one(nn.Module):
    def __init__(self, visual_net, txt_net):
        super(Location_Net_stage_one, self).__init__()

        # backbone net
        self.visual_net = visual_net
        self.txt_net = txt_net



    def forward(self, v_input, text_ids, text_atts, return_intermediate=False):

        # visual pathway
        if return_intermediate:
            v_fea, image_embeds, intermediate_feat = self.visual_net(v_input, return_intermediate=True)
        else:
            v_fea, image_embeds = self.visual_net(v_input)
            intermediate_feat = None
        
         
        # audio pathway
        t_fea = self.txt_net(text_ids, text_atts, return_dict=True, mode='text').last_hidden_state
    
        if return_intermediate:
            return v_fea, t_fea, image_embeds, intermediate_feat
        else:
            return v_fea, t_fea, image_embeds


