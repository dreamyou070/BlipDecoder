import os
from models.blip import blip_decoder
import torch
import torch.nn as nn
from models.med import BertConfig, BertLMHeadModel
from transformers import BertTokenizer
from models.vit import VisionTransformer

def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    return tokenizer

vision_width = 768
visual_encoderckpt_layer = VisionTransformer(img_size=224, patch_size=16, embed_dim=vision_width, depth=12,
                                   num_heads=12,
                                   use_grad_checkpointing=False,
                                   ckpt_layer=0,
                                   drop_path_rate=0)
class BLIP_Decoder(nn.Module):
    def __init__(self,
                 med_config='configs/med_config.json',
                 image_size=384,
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 prompt='a picture of ',
                 ):
        super().__init__()

        self.visual_encoder, vision_width = visual_encoderckpt_layer, 768
        self.tokenizer = init_tokenizer()
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_decoder = BertLMHeadModel(config=med_config)

        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids) - 1

    def forward(self, image, caption):

        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        text = self.tokenizer(caption, padding='longest', truncation=True, max_length=40, return_tensors="pt").to(image.device)
        text.input_ids[:, 0] = self.tokenizer.bos_token_id

        decoder_targets = text.input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100)
        decoder_targets[:, :self.prompt_length] = -100

        decoder_output = self.text_decoder(text.input_ids,
                                           attention_mask=text.attention_mask,

                                           encoder_hidden_states=image_embeds,
                                           encoder_attention_mask=image_atts,

                                           labels=decoder_targets,
                                           return_dict=True,)
        loss_lm = decoder_output.loss
        return loss_lm


from PIL import Image
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

img_dir = '/home/dreamyou070/MyData/anomal_source/dtd_images/banded/banded_0002.jpg'
image = Image.open(img_dir)
inputs = processor(text=["a photo of a cat", "a photo of a dog"],
                   images=image, return_tensors="pt", padding=True)
caption_moel = BLIP_Decoder()
result = caption_moel(**inputs)