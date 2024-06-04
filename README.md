# 节点分类
```bash
# 对比
python node_classification/structure_lrgae.py --left 2 --right 2
python node_classification/feature_lrgae.py --left 2 --right 0 --device 1

python node_classification/maskgae.py
python node_classification/s2gae.py
python node_classification/graphmae.py
python node_classification/graphmae2.py
python node_classification/augmae.py
python node_classification/gigamae.py
```

# 链路预测
```bash
# 对比 一跳和一跳
python link_prediction/structure_lrgae.py --encoder_layers 1 --left 1 --right 1
python link_prediction/feature_lrgae.py --left 2 --right 0 --device 1
python link_prediction/graphmae.py
python link_prediction/graphmae2.py
python link_prediction/augmae.py
python link_prediction/maskgae.py --encoder_layers 1
python link_prediction/s2gae.py
python link_prediction/gigamae.py

```
