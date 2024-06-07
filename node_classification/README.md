+ lrGAE-5
```bash
python node_classification/structure_lrgae.py --dataset Cora --left 2 --right 2 --view AA
python node_classification/structure_lrgae.py --dataset Citeseer --left 2 --right 2 --view AA
python node_classification/structure_lrgae.py --dataset Pubmed --left 2 --right 2 --view AA
python node_classification/structure_lrgae.py --dataset Photo --left 2 --right 2 --view AA
python node_classification/structure_lrgae.py --dataset Computers --left 2 --right 2 --view AA
python node_classification/structure_lrgae.py --dataset CS --left 2 --right 2 --view AA
python node_classification/structure_lrgae.py --dataset Physics --left 2 --right 2 --view AA
```

+ lrGAE-6
```bash
python node_classification/structure_lrgae.py --dataset Cora --left 2 --right 1 --view AA
python node_classification/structure_lrgae.py --dataset Citeseer --left 2 --right 1 --view AA
python node_classification/structure_lrgae.py --dataset Pubmed --left 2 --right 1 --view AA
python node_classification/structure_lrgae.py --dataset Photo --left 2 --right 1 --view AA
python node_classification/structure_lrgae.py --dataset Computers --left 2 --right 1 --view AA
python node_classification/structure_lrgae.py --dataset CS --left 2 --right 1 --view AA
python node_classification/structure_lrgae.py --dataset Physics --left 2 --right 1 --view AA
```
+ lrGAE-7
```bash
python node_classification/structure_lrgae.py --dataset Cora --view AB
python node_classification/structure_lrgae.py --dataset Citeseer --view AB
python node_classification/structure_lrgae.py --dataset Pubmed --view AB
python node_classification/structure_lrgae.py --dataset Photo --view AB
python node_classification/structure_lrgae.py --dataset Computers --view AB
python node_classification/structure_lrgae.py --dataset CS --view AB
python node_classification/structure_lrgae.py --dataset Physics --view AB
```

+ lrGAE-8
```bash
python node_classification/structure_lrgae.py --dataset Cora --left 2 --right 1 --view AB
python node_classification/structure_lrgae.py --dataset Citeseer --left 2 --right 1 --view AB
python node_classification/structure_lrgae.py --dataset Pubmed --left 2 --right 1 --view AB
python node_classification/structure_lrgae.py --dataset Photo --left 2 --right 1 --view AB
python node_classification/structure_lrgae.py --dataset Computers --left 2 --right 1 --view AB
python node_classification/structure_lrgae.py --dataset CS --left 2 --right 1 --view AB
python node_classification/structure_lrgae.py --dataset Physics --left 2 --right 1 --view AB
```

+ GAE$_f$
```bash
python node_classification/feature_lrgae.py --dataset Cora --mask none --encoder_activation relu --encoder_dropout 0.5 --layer gcn
python node_classification/feature_lrgae.py --dataset Citeseer --mask none --encoder_activation relu --encoder_dropout 0.5 --layer gcn
python node_classification/feature_lrgae.py --dataset Pubmed --mask none --encoder_activation relu --encoder_dropout 0.5 --layer gcn
python node_classification/feature_lrgae.py --dataset Photo --mask none --encoder_activation relu --encoder_dropout 0.5 --layer gcn
python node_classification/feature_lrgae.py --dataset Computers --mask none --encoder_activation relu --encoder_dropout 0.5 --layer gcn
python node_classification/feature_lrgae.py --dataset CS --mask none --encoder_activation relu --encoder_dropout 0.5 --layer gcn
python node_classification/feature_lrgae.py --dataset Physics --mask none --encoder_activation relu --encoder_dropout 0.5 --layer gcn
```

+ GAE
```bash
python node_classification/structure_lrgae.py --dataset Cora --mask none --encoder_activation relu --encoder_dropout 0.5
python node_classification/structure_lrgae.py --dataset Citeseer --mask none --encoder_activation relu --encoder_dropout 0.5
python node_classification/structure_lrgae.py --dataset Pubmed --mask none --encoder_activation relu --encoder_dropout 0.5
python node_classification/structure_lrgae.py --dataset Photo --mask none --encoder_activation relu --encoder_dropout 0.5
python node_classification/structure_lrgae.py --dataset Computers --mask none --encoder_activation relu --encoder_dropout 0.5
python node_classification/structure_lrgae.py --dataset CS --mask none --encoder_activation relu --encoder_dropout 0.5
python node_classification/structure_lrgae.py --dataset Physics --mask none --encoder_activation relu --encoder_dropout 0.5
```

+ S2GAE
```bash
python node_classification/s2gae.py --dataset Cora
python node_classification/s2gae.py --dataset Citeseer
python node_classification/s2gae.py --dataset Pubmed
python node_classification/s2gae.py --dataset Photo
python node_classification/s2gae.py --dataset Computers
python node_classification/s2gae.py --dataset CS
python node_classification/s2gae.py --dataset Physics
```

+ MaskGAE
```bash
python node_classification/maskgae.py --dataset Cora
python node_classification/maskgae.py --dataset Citeseer
python node_classification/maskgae.py --dataset Pubmed
python node_classification/maskgae.py --dataset Photo
python node_classification/maskgae.py --dataset Computers
python node_classification/maskgae.py --dataset CS
python node_classification/maskgae.py --dataset Physics
```

+ GraphMAE
```bash
python node_classification/graphmae.py --dataset Cora
python node_classification/graphmae.py --dataset Citeseer
python node_classification/graphmae.py --dataset Pubmed --mode cat
python node_classification/graphmae.py --dataset Photo
python node_classification/graphmae.py --dataset Computers
python node_classification/graphmae.py --dataset CS
python node_classification/graphmae.py --dataset Physics
```

+ GraphMAE2
```bash
python node_classification/graphmae2.py --dataset Cora
python node_classification/graphmae2.py --dataset Citeseer
python node_classification/graphmae2.py --dataset Pubmed --mode cat
python node_classification/graphmae2.py --dataset Photo
python node_classification/graphmae2.py --dataset Computers
python node_classification/graphmae2.py --dataset CS
python node_classification/graphmae2.py --dataset Physics
```

+ AUG-MAE
```bash
python node_classification/augmae.py --dataset Cora
python node_classification/augmae.py --dataset Citeseer
python node_classification/augmae.py --dataset Pubmed --mode cat
python node_classification/augmae.py --dataset Photo
python node_classification/augmae.py --dataset Computers
python node_classification/augmae.py --dataset CS
python node_classification/augmae.py --dataset Physics
```

+ GiGaMAE
```bash
python node_classification/gigamae.py --dataset Cora
python node_classification/gigamae.py --dataset Citeseer
python node_classification/gigamae.py --dataset Pubmed --mode cat
python node_classification/gigamae.py --dataset Photo
python node_classification/gigamae.py --dataset Computers
python node_classification/gigamae.py --dataset CS
python node_classification/gigamae.py --dataset Physics
```