import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer, BertTokenizerFast
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from PIL import Image
import requests
from util import trans_to_square,trans_to_square_inputImage
import copy
Clip_path="<path>/clip-vit-large-patch14"
Roberta_path="<path>/Taiyi-CLIP-Roberta-large-326M-Chinese"

class Taiyi_CLIP(nn.Module):
    def __init__(self):
        super(Taiyi_CLIP, self).__init__()
        self.clip_model=CLIPModel.from_pretrained(Clip_path)
        for name,parameter in self.clip_model.named_parameters():
            parameter.requires_grad=False
        self.vision_model = self.clip_model.vision_model
        self.visual_projection=self.clip_model.visual_projection
        for name,parameter in self.vision_model.named_parameters():
            parameter.requires_grad=True
        for name,parameter in self.visual_projection.named_parameters():
            parameter.requires_grad=True
#         self.
        self.processor = CLIPProcessor.from_pretrained(Clip_path)
        self.text_tokenizer = BertTokenizer.from_pretrained(Roberta_path)
        self.text_encoder = BertForSequenceClassification.from_pretrained(Roberta_path)
        for name,parameter in self.text_encoder.named_parameters():
            parameter.requires_grad=True
        self.logit_scale=self.clip_model.logit_scale
        self.logit_scale.requires_grad=True
#         del self.clip_model.text_model
        #设置require_gred
#         named_parameters = list(self.named_parameters())
#         train_params = [n for n, p in named_parameters if p.requires_grad]
#         freeze_params = [n for n, p in named_parameters if not p.requires_grad]
#         print('------------------leared parameters')
#         print(train_params)
#         print('------------------freeze parameters')
#         print(freeze_params)
    def encode_text(self, text):
        return self.text_encoder(text).logits
    def encode_image(self, image):
        vision_outputs = self.vision_model(
            pixel_values=image,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        )
        pooled_output = vision_outputs[1]  # pooled_output
        image_features = self.visual_projection(pooled_output)

        return image_features
#         return self.clip_model.get_image_features(pixel_values=image)
    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return image_features, text_features, self.logit_scale.exp()
    def freeze_vision_parameter(self):
        for name,parameter in self.vision_model.named_parameters():
            parameter.requires_grad=False
        for name,parameter in self.visual_projection.named_parameters():
            parameter.requires_grad=False
        self.logit_scale.requires_grad=True
        print('____________freeze vision encoder parameter without logit_scale')
def get_taiyi_tokenizer():
    text_tokenizer = BertTokenizer.from_pretrained(Roberta_path)
    return text_tokenizer

def get_taiyi_tokenizer_fast():
    text_tokenizer = BertTokenizerFast.from_pretrained(Roberta_path)
    return text_tokenizer
def get_taiyi_image_process():
    processor = CLIPProcessor.from_pretrained(Clip_path)
    return processor


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

image_taiyi_processer=get_taiyi_image_process()
def image_process_func(image_path):
    img_=Image.open(image_path).convert('RGB')
    image = image_taiyi_processer(images=img_, return_tensors="pt")
    image=image['pixel_values'][0] 
    return image
def extract_image_func_batch(model,batch_key,batch_tensor):
    batch_tensor=batch_tensor.to(device)
    image_features = model.encode_image(batch_tensor)
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    image_features_list=image_features.tolist()
    return batch_key,image_features_list
def extract_image_func(model,image_path):
    img_=Image.open(image_path).convert('RGB')
    image = model.processor(images=img_, return_tensors="pt")
    image=image['pixel_values'].to(device)
    image_features = model.encode_image(image)
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    image_features_list=image_features[0].tolist()
    return image_features_list
def extract_image_func_PIL(model,img_):
    image = model.processor(images=img_, return_tensors="pt")
    image=image['pixel_values'].to(device)
    image_features = model.encode_image(image)
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    image_features_list=image_features[0].tolist()
    return image_features_list
def extract_text_func(model,text):
    text_key=[text]
    text = model.text_tokenizer.encode(str(text),max_length=32,truncation=True,padding='max_length')
    text=torch.tensor(text).unsqueeze(0)
    text=text.to(device)
    text_features =model.encode_text(text)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    text_features_list=text_features[0].tolist()
    return text_features_list
def inti_model():
    # load taiyi model
    model=Taiyi_CLIP()
    model.float()
    model.eval()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model=model.to(device)
    return model

def clip_guide_func():
    model=inti_model()
    def clip_guide(segments,source_prompt,raw_image):
        # get best mask according the text input
        all_sub_image=[]
        with torch.no_grad():
            text_featrue=np.array(extract_text_func(model,source_prompt),dtype=np.float32)
        for mask_i in range(len(segments)):
                mask_matri=segments[mask_i]['segmentation']
                a,b=np.where(mask_matri==True)
                # print(min(a),max(a),min(b),max(b))
                # bbox=mask_matri['bbox']
                arround_true=mask_matri[min(a):max(a)+1,min(b):max(b)+1]
                image_sub=raw_image[min(a):max(a)+1,min(b):max(b)+1,:]
                # image_sub_=Image.fromarray(copy.deepcopy(image_sub))
                image_sub=image_sub[:,:,(2,1,0)]
                h_,w_=arround_true.shape
                for i_ in range(h_):
                    for j_ in range(w_):
                        if not arround_true[i_][j_]:
                            image_sub[i_][j_]=np.array([255,255,255],dtype=np.uint8)
                image_sub_=Image.fromarray(copy.deepcopy(image_sub))
                copy_img_=image_sub_.convert('RGB')
                with torch.no_grad():
                    img_pil_square=trans_to_square_inputImage(copy_img_)
                    image_featrue=extract_image_func_PIL(model,img_pil_square)
                sim_=score=np.dot(image_featrue,text_featrue.T)
                all_sub_image.append((img_pil_square,sim_,mask_i))
        all_sub_image.sort(key=lambda x:x[1],reverse=True)
        return all_sub_image
    return clip_guide
if __name__=='__main__':
    model=Taiyi_CLIP()
    model.float()
    model.eval()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model=model.to(device)
    image_path_test='xxx.jpg'
    text_test='<text>'
    image_featrue=extract_image_func(model,image_path_test)
    text_featrue=extract_text_func(model,text_test)
    image_process_func(image_path_test)
    len(image_featrue),len(text_featrue)
    
