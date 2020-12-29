#!/usr/bin/env python
# COPYRIGHT 2020. Fred Fung. Boston University.
"""
Baseline Siamese model for vehicle retrieval task on CityFlow-NL
"""
import torch
import torch.nn.functional as F
from torchvision.models import resnet50
from transformers import BertTokenizer, BertModel


class SiameseBaselineModel(torch.nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.resnet50 = resnet50(pretrained=False,
                                 num_classes=model_cfg.OUTPUT_SIZE)
        state_dict = torch.load(self.model_cfg.RESNET_CHECKPOINT,
                                map_location=lambda storage, loc: storage.cpu())
        del state_dict["fc.weight"]
        del state_dict["fc.bias"]
        self.resnet50.load_state_dict(state_dict, strict=False)
        self.bert_tokenizer = BertTokenizer.from_pretrained(
            self.model_cfg.BERT_CHECKPOINT)
        self.bert_model = BertModel.from_pretrained(
            self.model_cfg.BERT_CHECKPOINT)
        self.lang_fc = torch.nn.Linear(768, model_cfg.OUTPUT_SIZE)

    def forward(self, track):
        nl = track["nl"]
        tokens = self.bert_tokenizer.batch_encode_plus(nl, padding='longest',
                                                       return_tensors='pt')
        outputs = self.bert_model(tokens['input_ids'].cuda(),
                                  attention_mask=tokens[
                                      'attention_mask'].cuda())
        lang_embeds = torch.mean(outputs.last_hidden_state, dim=1)
        lang_embeds = self.lang_fc(lang_embeds)
        crops = track["crop"].cuda()
        visual_embeds = self.resnet50(crops)
        d = F.pairwise_distance(visual_embeds, lang_embeds)
        similarity = torch.exp(-d)
        return similarity

    def compute_loss(self, track):
        similarity = self.forward(track)
        loss = F.binary_cross_entropy(similarity, track["label"][:, 0].cuda())
        return loss

    def compute_lang_embed(self, nls, rank):
        with torch.no_grad():
            tokens = self.bert_tokenizer.batch_encode_plus(nls,
                                                           padding='longest',
                                                           return_tensors='pt')
            outputs = self.bert_model(tokens['input_ids'].cuda(rank),
                                      attention_mask=tokens[
                                          'attention_mask'].cuda(rank))
            lang_embeds = torch.mean(outputs.last_hidden_state, dim=1)
            lang_embeds = self.lang_fc(lang_embeds)
        return lang_embeds

    def compute_similarity_on_frame(self, track, lang_embeds, rank):
        with torch.no_grad():
            crops = track["crops"][0].cuda(rank)
            visual_embeds = self.resnet50(crops)
            similarity = 0.
            for lang_embed in lang_embeds:
                d = F.pairwise_distance(visual_embeds, lang_embed)
                similarity += torch.mean(torch.exp(-d))
            similarity = similarity / len(lang_embeds)
        return similarity
