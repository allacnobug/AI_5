# AI_5
## 代码架构
    ├── Models
    │   ├── model1.py # 决策融合
    │   ├── model2.py # 特征简单融合
    │   ├── model3.py # 特征注意力融合
    │   ├── model4.py # 特征多层融合
    │   ├── model5.py # 交叉注意力融合
    ├── datapre.py # 处理数据
    ├── main.py # 主运行代码
    ├── utils.py # 其它工具
    ├── README.md

## 安装依赖
在运行代码之前，请确保已经安装了所需的依赖。你可以使用以下命令安装依赖：

        pip install -r requirements.txt
## 训练模型
如果你想进行模型训练，可以运行以下命令：

        python main.py --train --model_type 5 --lr 5e-5 --weight_decay 1e-2 --epoch 5
        
可选参数：

        --train：启动训练模式。
        --model_type：选择模型类别，默认为 '5'。
        --lr：学习率，默认为 5e-5。
        --weight_decay：权重衰减，默认为 1e-2。
        --epoch：训练轮数，默认为 5。
        
测试模型

如果你想进行模型测试，可以运行以下命令：

        python main.py --test --model_type 5
        
可选参数：

        --test：启动测试模式。
        --model_type：选择模型类别，默认为 '5'。
        
使用文本或图像，你可以选择只使用文本或图像进行测试，通过以下命令启动相应的模式：

仅用文本：

        python your_script.py --valid --text
        
仅用图像：

        python your_script.py --valid --img
        
其他参数

    --load_model_path：如果你有一个已经训练好的模型，可以通过此参数加载模型，例如：
    
        python your_script.py --test --model_type 5 --load_model_path /path/to/you
        
## Attribution
Parts of this code are based on the following repositories:

https://github.com/YeexiaoZheng/Multimodal-Sentiment-Analysis

https://github.com/Miaheeee/AI_lab5

https://github.com/joker-star-l/ai_lab5
