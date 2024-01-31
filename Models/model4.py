# 多层融合
import torch
import torch.nn as nn
from transformers import AutoModel
from torchvision.models import resnet50

# 定义基于文本的模型
class TextModel(nn.Module):

    def __init__(self, config):
        super(TextModel, self).__init__()

        # 载入预训练的BERT模型
        self.bert = AutoModel.from_pretrained(config.bert_name)

        # 定义对BERT输出进行转换的Sequential模块
        self.trans = nn.Sequential(
            nn.Dropout(config.bert_dropout),
            nn.Linear(self.bert.config.hidden_size, config.middle_hidden_size),
            nn.ReLU(inplace=True)
        )

    def forward(self, bert_inputs, masks, token_type_ids=None):
        assert bert_inputs.shape == masks.shape, '错误！bert_inputs和masks必须具有相同的形状！'

        # 通过BERT模型进行前向传播
        bert_out = self.bert(input_ids=bert_inputs, token_type_ids=token_type_ids, attention_mask=masks)

        # 从BERT中提取池化输出
        pooler_out = bert_out['pooler_output']

        # 在池化输出上应用转换
        return self.trans(pooler_out)


# 定义基于图像的模型
class ImageModel(nn.Module):

    def __init__(self, config):
        super(ImageModel, self).__init__()

        # 使用预训练的ResNet50模型
        self.full_resnet = resnet50(pretrained=True)

        # 定义对ResNet输出进行转换的Sequential模块
        self.resnet = nn.Sequential(
            *(list(self.full_resnet.children())[:-1]),
            nn.Flatten()
        )

        # 定义对ResNet特征进行转换的Sequential模块
        self.trans = nn.Sequential(
            nn.Dropout(config.resnet_dropout),
            nn.Linear(self.full_resnet.fc.in_features, config.middle_hidden_size),
            nn.ReLU(inplace=True)
        )

    def forward(self, imgs):
        # 提取图像特征
        feature = self.resnet(imgs)

        # 在图像特征上应用转换
        return self.trans(feature)


# 定义整体模型，包括文本模型、图像模型、多层融合层和全连接分类器
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        # 文本模型
        self.text_model = TextModel(config)

        # 图像模型
        self.img_model = ImageModel(config)

        # 多层融合层
        self.fusion_layers = nn.ModuleList()
        for _ in range(config.num_fusion_layers):
            fusion_layer = nn.Sequential(
                nn.Linear(config.middle_hidden_size + config.middle_hidden_size, config.middle_hidden_size + config.middle_hidden_size),
                nn.ReLU(inplace=True)
            )
            self.fusion_layers.append(fusion_layer)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(config.fuse_dropout),
            nn.Linear(config.middle_hidden_size * 2, config.out_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(config.fuse_dropout),
            nn.Linear(config.out_hidden_size, config.num_labels)
        )

        # 交叉熵损失函数
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, texts, texts_mask, imgs, labels=None):
        # 获取文本特征
        text_features = self.text_model(texts, texts_mask)

        # 获取图像特征
        img_features = self.img_model(imgs)

        # 文本和图像特征的初始融合
        fused_features = torch.cat([text_features, img_features], dim=1)

        # 多层次融合
        for fusion_layer in self.fusion_layers:
            fused_features = fusion_layer(fused_features)

        # 最终分类
        prob_vec = self.classifier(fused_features)

        # 预测标签
        pred_labels = torch.argmax(prob_vec, dim=1)

        # 如果提供了标签，计算损失并返回
        if labels is not None:
            loss = self.loss_func(prob_vec, labels)
            return pred_labels, loss
        else:
            # 否则，只返回预测的标签
            return pred_labels
