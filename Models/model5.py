# 交叉注意力机制
import torch
import torch.nn as nn
from transformers import AutoModel
from torchvision.models import resnet50
class TextModel(nn.Module):

    def __init__(self, config):
        super(TextModel, self).__init__()

        self.bert = AutoModel.from_pretrained(config.bert_name)
        self.trans = nn.Sequential(
            nn.Dropout(config.bert_dropout),
            nn.Linear(self.bert.config.hidden_size, config.middle_hidden_size),
            nn.ReLU(inplace=True)
        )

    def forward(self, bert_inputs, masks, token_type_ids=None):
        assert bert_inputs.shape == masks.shape, 'error! bert_inputs and masks must have same shape!'
        bert_out = self.bert(input_ids=bert_inputs, token_type_ids=token_type_ids, attention_mask=masks)
        pooler_out = bert_out['pooler_output']

        return self.trans(pooler_out)


class ImageModel(nn.Module):

    def __init__(self, config):
        super(ImageModel, self).__init__()
        self.full_resnet = resnet50(pretrained=True)
        self.resnet = nn.Sequential(
            *(list(self.full_resnet.children())[:-1]),
            nn.Flatten()
        )

        self.trans = nn.Sequential(
            nn.Dropout(config.resnet_dropout),
            nn.Linear(self.full_resnet.fc.in_features, config.middle_hidden_size),
            nn.ReLU(inplace=True)
        )

    def forward(self, imgs):
        feature = self.resnet(imgs)
        return self.trans(feature)

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        # text
        self.text_model = TextModel(config)
        # image
        self.img_model = ImageModel(config)

        # 交叉注意力层
        self.cross_attention = nn.MultiheadAttention(embed_dim=config.middle_hidden_size, num_heads=config.attention_nhead)

        # 全连接分类器
        self.classifier = nn.Sequential(
            nn.Dropout(config.fuse_dropout),
            nn.Linear(config.middle_hidden_size, config.num_labels)
        )

        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, texts, texts_mask, imgs, labels=None):
        # 处理文本信息
        text_features = self.text_model(texts, texts_mask)

        # 处理图像信息
        img_features = self.img_model(imgs)

        # 使用交叉注意力融合文本和图像的特征
        attention_out, _ = self.cross_attention(text_features, img_features, img_features)

        # 最终分类
        output = self.classifier(attention_out.squeeze(0))

        pred_labels = torch.argmax(output, dim=1)

        if labels is not None:
            loss = self.loss_func(output, labels)
            return pred_labels, loss
        else:
            return pred_labels

