import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore")

import torch
import argparse
from datapre import read_data_from_file, Processor
from utils import train_val_split, save_model, write_to_file, Trainer
import matplotlib.pyplot as plt

class config:
    root_path = os.getcwd()
    data_dir = os.path.join(root_path, './datadata/data/')
    train_data_path = os.path.join(root_path, 'datadata/train.json')
    test_data_path = os.path.join(root_path, 'datadata/test.json')
    output_path = os.path.join(root_path, 'output')
    output_test_path = os.path.join(output_path, 'test.txt')
    load_model_path = None

    epoch = 1
    learning_rate = 3e-5
    weight_decay = 0
    num_labels = 3

    only = None
    middle_hidden_size = 64
    attention_nhead = 8
    attention_dropout = 0.4
    fuse_dropout = 0.5
    out_hidden_size = 128
    fuse_model_type = '1'

    bert_name = 'bert-base-uncased'
    bert_learning_rate = 5e-6
    bert_dropout = 0.2

    image_size = 224
    resnet_learning_rate = 5e-6
    resnet_dropout = 0.2
    img_hidden_seq = 64

    checkout_params = {'batch_size': 4, 'shuffle': False}
    train_params = {'batch_size': 16, 'shuffle': True, 'num_workers': 2}
    val_params = {'batch_size': 16, 'shuffle': False, 'num_workers': 2}
    test_params = {'batch_size': 8, 'shuffle': False, 'num_workers': 2}

    num_fusion_layers = 3

# 参数
parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', default=True, help='训练')
parser.add_argument('--test', action='store_true', default=False, help='测试')

parser.add_argument('--text_model', default='bert-base-uncased', help='文本分析模型', type=str)
parser.add_argument('--model_type', default='5', help='模型类别', type=str)
parser.add_argument('--text', action='store_true', default=False, help='仅用文本')
parser.add_argument('--img', action='store_true', default=False, help='仅用图像')

parser.add_argument('--lr', default=5e-5, help='学习率', type=float)
parser.add_argument('--weight_decay', default=1e-2, help='权重衰减', type=float)
parser.add_argument('--epoch', default=5, help='训练轮数', type=int)

parser.add_argument('--load_model_path', default=None, help='已经训练好的模型路径', type=str)

args = parser.parse_args()

config.learning_rate = args.lr
config.weight_decay = args.weight_decay
config.epoch = args.epoch
config.text_model = args.text_model
config.model_type = args.model_type
config.load_model_path = args.load_model_path

config.only = 'img' if args.img else None
config.only = 'text' if args.text else None
if args.img and args.text : config.only = None
processor = Processor(config)

# 训练
def train():
    data = read_data_from_file(config.train_data_path, config.data_dir, config.only)
    train_data, val_data = train_val_split(data)
    train_loader = processor(train_data, config.train_params)
    val_loader = processor(val_data, config.val_params)
    best_acc = 0
    epoch = config.epoch
    loss_list = []
    for e in range(epoch):
        print('\nEpoch ' + str(e + 1))
        tloss, tloss_list = trainer.train(train_loader)
        loss_list.extend(tloss_list)
        print('Train Loss: {}'.format(tloss))
        vloss, vacc = trainer.valid(val_loader)
        print('Valid Loss: {}'.format(vloss))
        print('Valid Acc: {}'.format(vacc))
        if vacc > best_acc:
            best_acc = vacc
            save_model(config.output_path, config.model_type, model)
            print('Update best model!')
        print()
    plt.plot(loss_list, label='Training Loss')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_loss_plot'+config.model_type+'.png')

def valid():
    if config.load_model_path is not None:
        load_model_path = 'output/'+config.load_model_path+'/pytorch_model.bin'
        model.load_state_dict(torch.load(load_model_path))
    data = read_data_from_file(config.train_data_path, config.data_dir, config.only)
    train_data, val_data = train_val_split(data)
    val_loader = processor(val_data, config.val_params)
    vloss, vmetric = trainer.valid(val_loader)
    print('Valid Loss: {}'.format(vloss))
    print(vmetric)

# 测试
def test():
    test_data = read_data_from_file(config.test_data_path, config.data_dir, config.only)
    test_loader = processor(test_data, config.test_params)

    if config.load_model_path is not None:
        load_model_path = 'output/'+config.load_model_path+'/pytorch_model.bin'
        model.load_state_dict(torch.load(load_model_path))
    outputs = trainer.predict(test_loader)
    outputs = processor.decode(outputs)
    write_to_file(config.output_test_path, outputs)


# main
if __name__ == "__main__":
    for i in range(3, 6):
        config.model_type = str(i)
        print(config.model_type)
        if config.model_type == '1':
            from Models.model1 import Model
        elif config.fuse_model_type == '2':
            from Models.model2 import Model
        elif config.fuse_model_type == '3':
            from Models.model3 import Model
        elif config.fuse_model_type == '4':
            from Models.model4 import Model
        else:
            from Models.model5 import Model

        model = Model(config)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        trainer = Trainer(config, processor, model, device)
        train()


    # if args.train:
    #     train()
    # if args.test:
    #     test()