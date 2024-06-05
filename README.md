# lrGAE: graph autoencoders as contrastive learning architectures
![](imgs/cases.png)
![](imgs/cases2.png)

# Reproduction
### Node classification
+ Baselines
```bash
python node_classification/maskgae.py
python node_classification/s2gae.py
python node_classification/graphmae.py
python node_classification/graphmae2.py
python node_classification/augmae.py
python node_classification/gigamae.py
```
+ lrGAE
```bash
python node_classification/structure_lrgae.py --left 2 --right 2
python node_classification/feature_lrgae.py --left 2 --right 0
```

### Link prediction
+ Baselines
```bash
python link_prediction/graphmae.py
python link_prediction/graphmae2.py
python link_prediction/augmae.py
python link_prediction/maskgae.py --encoder_layers 1
python link_prediction/s2gae.py
python link_prediction/gigamae.py
```
+ lrGAE
```bash
python link_prediction/structure_lrgae.py --encoder_layers 1 --left 1 --right 1
python link_prediction/feature_lrgae.py --left 2 --right 0
```

