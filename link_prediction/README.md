+ lrGAE-5
```bash
python link_prediction/structure_lrgae.py --dataset Cora --left 1 --right 1 --encoder_layers 1  --view AA
python link_prediction/structure_lrgae.py --dataset Citeseer --left 1 --right 1 --encoder_layers 1  --view AA
python link_prediction/structure_lrgae.py --dataset Pubmed --left 1 --right 1 --encoder_layers 1  --view AA  --encoder_dropout 0.2
```

+ lrGAE-6
```bash
python link_prediction/structure_lrgae.py --dataset Cora --left 2 --right 1 --encoder_layers 2  --view AA
python link_prediction/structure_lrgae.py --dataset Citeseer --left 2 --right 1 --encoder_layers 2  --view AA
python link_prediction/structure_lrgae.py --dataset Pubmed --left 2 --right 1 --encoder_layers 2  --view AA  --encoder_dropout 0.2
```
+ lrGAE-7
```bash
python link_prediction/structure_lrgae.py --dataset Cora --left 1 --right 1 --encoder_layers 1 --view AB
python link_prediction/structure_lrgae.py --dataset Citeseer --left 1 --right 1 --encoder_layers 1 --view AB
python link_prediction/structure_lrgae.py --dataset Pubmed --left 1 --right 1 --encoder_layers 1 --view AB  --encoder_dropout 0.2
```

+ lrGAE-8
```bash
python link_prediction/structure_lrgae.py --dataset Cora --left 2 --right 1 --encoder_layers 2 --view AB
python link_prediction/structure_lrgae.py --dataset Citeseer --left 2 --right 1 --encoder_layers 2 --view AB
python link_prediction/structure_lrgae.py --dataset Pubmed --left 2 --right 1 --encoder_layers 2 --view AB  --encoder_dropout 0.2
```

+ GAE$_f$
```bash
python link_prediction/feature_lrgae.py --dataset Cora --mask none --encoder_activation relu --encoder_dropout 0.5 --layer gcn
python link_prediction/feature_lrgae.py --dataset Citeseer --mask none --encoder_activation relu --encoder_dropout 0.5 --layer gcn
python link_prediction/feature_lrgae.py --dataset Pubmed --mask none --encoder_activation relu --encoder_dropout 0.5 --layer gcn
```


+ GAE
```bash
python link_prediction/structure_lrgae.py --dataset Cora --mask none --encoder_activation relu --encoder_dropout 0.5 --left 1 --right 1 --encoder_layers 2
python link_prediction/structure_lrgae.py --dataset Citeseer --mask none --encoder_activation relu --encoder_dropout 0.5 --left 1 --right 1 --encoder_layers 2
python link_prediction/structure_lrgae.py --dataset Pubmed --mask none --encoder_activation relu --encoder_dropout 0.5 --left 1 --right 1 --encoder_layers 2
```

+ S2GAE
python link_prediction/s2gae.py --dataset Cora
python link_prediction/s2gae.py --dataset Citeseer
python link_prediction/s2gae.py --dataset Pubmed

+ MaskGAE
python link_prediction/maskgae.py --dataset Cora
python link_prediction/maskgae.py --dataset Citeseer
python link_prediction/maskgae.py --dataset Pubmed --encoder_dropout 0.2

+ GraphMAE
python link_prediction/graphmae.py --dataset Cora
python link_prediction/graphmae.py --dataset Citeseer
python link_prediction/graphmae.py --dataset Pubmed

+ GraphMAE2
python link_prediction/graphmae2.py --dataset Cora
python link_prediction/graphmae2.py --dataset Citeseer
python link_prediction/graphmae2.py --dataset Pubmed

+ AUG-MAE
python link_prediction/augmae.py --dataset Cora
python link_prediction/augmae.py --dataset Citeseer
python link_prediction/augmae.py --dataset Pubmed

+ GiGaMAE
python link_prediction/gigamae.py --dataset Cora
python link_prediction/gigamae.py --dataset Citeseer
python link_prediction/gigamae.py --dataset Pubmed
