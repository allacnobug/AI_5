import os
import json
import chardet
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score, classification_report, accuracy_score
from transformers import AutoTokenizer
from torchvision import transforms

# 将文本和标签格式化成一个json
def format_data(input_path, data_dir, output_path):
    data = []
    with open(input_path) as f:
        for line in tqdm(f.readlines(), desc='Formating'):
            guid, label = line.replace('\n', '').split(',')
            text_path = os.path.join(data_dir, (guid + '.txt'))
            if guid == 'guid': continue
            with open(text_path, 'rb') as textf:
                text_byte = textf.read()
                encode = chardet.detect(text_byte)
                try:
                    text = text_byte.decode(encode['encoding'])
                except:
                    try:
                        text = text_byte.decode('iso-8859-1').encode('iso-8859-1').decode('gbk')
                    except:
                        print('not is0-8859-1', guid)
                        continue
            text = text.strip('\n').strip('\r').strip(' ').strip()
            data.append({
                'guid': guid,
                'label': label,
                'text': text
            })
    with open(output_path, 'w') as wf:
        json.dump(data, wf, indent=4)

# 读取数据，返回[(guid, text, img, label)]元组列表
def read_data_from_file(path, data_dir, only=None):
    data = []
    with open(path) as f:
        json_file = json.load(f)
        for d in tqdm(json_file, desc='Loading'):
            guid, label, text = d['guid'], d['label'], d['text']
            if guid == 'guid': continue

            if only == 'text': img = Image.new(mode='RGB', size=(224, 224), color=(0, 0, 0))
            else:
                img_path = os.path.join(data_dir, (guid + '.jpg'))
                # img = cv2.imread(img_path)
                img = Image.open(img_path)
                img.load()

            if only == 'img': text = ''

            data.append((guid, text, img, label))
        f.close()

    return data

class MyDataset(Dataset):

    def __init__(self, guids, texts, imgs, labels) -> None:
        self.guids = guids
        self.texts = texts
        self.imgs = imgs
        self.labels = labels

    def __len__(self):
        return len(self.guids)

    def __getitem__(self, index):
        return self.guids[index], self.texts[index], \
            self.imgs[index], self.labels[index]

    def collate_fn(self, batch):
        guids = [b[0] for b in batch]
        texts = [torch.LongTensor(b[1]) for b in batch]
        imgs = torch.FloatTensor([np.array(b[2]).tolist() for b in batch])
        labels = torch.LongTensor([b[3] for b in batch])

        texts_mask = [torch.ones_like(text) for text in texts]

        paded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
        paded_texts_mask = pad_sequence(texts_mask, batch_first=True, padding_value=0).gt(0)

        return guids, paded_texts, paded_texts_mask, imgs, labels

class LabelVocab:
    UNK = 'UNK'

    def __init__(self) -> None:
        self.label2id = {}
        self.id2label = {}

    def __len__(self):
        return len(self.label2id)

    def add_label(self, label):
        if label not in self.label2id:
            self.label2id.update({label: len(self.label2id)})
            self.id2label.update({len(self.id2label): label})

    def label_to_id(self, label):
        return self.label2id.get(label)

    def id_to_label(self, id):
        return self.id2label.get(id)


class Processor:

    def __init__(self, config) -> None:
        self.config = config
        self.labelvocab = LabelVocab()
        pass

    def __call__(self, data, params):
        return self.to_loader(data, params)

    def encode(self, data):
        self.labelvocab.add_label('positive')
        self.labelvocab.add_label('neutral')
        self.labelvocab.add_label('negative')
        self.labelvocab.add_label('null')

        tokenizer = AutoTokenizer.from_pretrained(self.config.text_model)

        def get_resize(image_size):
            for i in range(20):
                if 2 ** i >= image_size:
                    return 2 ** i
            return image_size

        img_transform = transforms.Compose([
            transforms.Resize(get_resize(self.config.image_size)),
            transforms.CenterCrop(self.config.image_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        guids, encoded_texts, encoded_imgs, encoded_labels = [], [], [], []
        for line in tqdm(data, desc='----- [Encoding]'):
            guid, text, img, label = line
            guids.append(guid)
            text.replace('#', '')
            tokens = tokenizer.tokenize('[CLS]' + text + '[SEP]')
            encoded_texts.append(tokenizer.convert_tokens_to_ids(tokens))
            encoded_imgs.append(img_transform(img))
            encoded_labels.append(self.labelvocab.label_to_id(label))

        return guids, encoded_texts, encoded_imgs, encoded_labels

    def decode(self, outputs):
        formated_outputs = ['guid,tag']
        for guid, label in tqdm(outputs, desc='Decoding'):
            formated_outputs.append((str(guid) + ',' + self.labelvocab.id_to_label(label)))
        return formated_outputs

    def metric(self, true_labels, pred_labels):
        print(classification_report(true_labels, pred_labels))
        return accuracy_score(true_labels, pred_labels)

    def to_dataset(self, data):
        dataset_inputs = self.encode(data)
        return MyDataset(*dataset_inputs)

    def to_loader(self, data, params):
        dataset = self.to_dataset(data)
        return DataLoader(dataset=dataset, **params, collate_fn=dataset.collate_fn)

