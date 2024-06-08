```bash
cd examples/node_classification
```


+ lrGAE-5
```bash
python structure_lrgae.py --dataset Cora --left 2 --right 2 --view AA
python structure_lrgae.py --dataset Citeseer --left 2 --right 2 --view AA
python structure_lrgae.py --dataset Pubmed --left 2 --right 2 --view AA
python structure_lrgae.py --dataset Photo --left 2 --right 2 --view AA
python structure_lrgae.py --dataset Computers --left 2 --right 2 --view AA
python structure_lrgae.py --dataset CS --left 2 --right 2 --view AA
python structure_lrgae.py --dataset Physics --left 2 --right 2 --view AA
```

+ lrGAE-6
```bash
python structure_lrgae.py --dataset Cora --left 2 --right 1 --view AA
python structure_lrgae.py --dataset Citeseer --left 2 --right 1 --view AA
python structure_lrgae.py --dataset Pubmed --left 2 --right 1 --view AA
python structure_lrgae.py --dataset Photo --left 2 --right 1 --view AA
python structure_lrgae.py --dataset Computers --left 2 --right 1 --view AA
python structure_lrgae.py --dataset CS --left 2 --right 1 --view AA
python structure_lrgae.py --dataset Physics --left 2 --right 1 --view AA
```
+ lrGAE-7
```bash
python structure_lrgae.py --dataset Cora --view AB
python structure_lrgae.py --dataset Citeseer --view AB
python structure_lrgae.py --dataset Pubmed --view AB
python structure_lrgae.py --dataset Photo --view AB
python structure_lrgae.py --dataset Computers --view AB
python structure_lrgae.py --dataset CS --view AB
python structure_lrgae.py --dataset Physics --view AB
```

+ lrGAE-8
```bash
python structure_lrgae.py --dataset Cora --left 2 --right 1 --view AB
python structure_lrgae.py --dataset Citeseer --left 2 --right 1 --view AB
python structure_lrgae.py --dataset Pubmed --left 2 --right 1 --view AB
python structure_lrgae.py --dataset Photo --left 2 --right 1 --view AB
python structure_lrgae.py --dataset Computers --left 2 --right 1 --view AB
python structure_lrgae.py --dataset CS --left 2 --right 1 --view AB
python structure_lrgae.py --dataset Physics --left 2 --right 1 --view AB
```

+ GAE$_f$
```bash
python feature_lrgae.py --dataset Cora --mask none --encoder_activation relu --encoder_dropout 0.5 --layer gcn
python feature_lrgae.py --dataset Citeseer --mask none --encoder_activation relu --encoder_dropout 0.5 --layer gcn
python feature_lrgae.py --dataset Pubmed --mask none --encoder_activation relu --encoder_dropout 0.5 --layer gcn
python feature_lrgae.py --dataset Photo --mask none --encoder_activation relu --encoder_dropout 0.5 --layer gcn
python feature_lrgae.py --dataset Computers --mask none --encoder_activation relu --encoder_dropout 0.5 --layer gcn
python feature_lrgae.py --dataset CS --mask none --encoder_activation relu --encoder_dropout 0.5 --layer gcn
python feature_lrgae.py --dataset Physics --mask none --encoder_activation relu --encoder_dropout 0.5 --layer gcn
```

+ GAE
```bash
python structure_lrgae.py --dataset Cora --mask none --encoder_activation relu --encoder_dropout 0.5
python structure_lrgae.py --dataset Citeseer --mask none --encoder_activation relu --encoder_dropout 0.5
python structure_lrgae.py --dataset Pubmed --mask none --encoder_activation relu --encoder_dropout 0.5
python structure_lrgae.py --dataset Photo --mask none --encoder_activation relu --encoder_dropout 0.5
python structure_lrgae.py --dataset Computers --mask none --encoder_activation relu --encoder_dropout 0.5
python structure_lrgae.py --dataset CS --mask none --encoder_activation relu --encoder_dropout 0.5
python structure_lrgae.py --dataset Physics --mask none --encoder_activation relu --encoder_dropout 0.5
```

+ S2GAE
```bash
python s2gae.py --dataset Cora
python s2gae.py --dataset Citeseer
python s2gae.py --dataset Pubmed
python s2gae.py --dataset Photo
python s2gae.py --dataset Computers
python s2gae.py --dataset CS
python s2gae.py --dataset Physics
```

+ MaskGAE
```bash
python maskgae.py --dataset Cora
python maskgae.py --dataset Citeseer
python maskgae.py --dataset Pubmed
python maskgae.py --dataset Photo
python maskgae.py --dataset Computers
python maskgae.py --dataset CS
python maskgae.py --dataset Physics
```

+ GraphMAE
```bash
python graphmae.py --dataset Cora
python graphmae.py --dataset Citeseer
python graphmae.py --dataset Pubmed --mode cat
python graphmae.py --dataset Photo
python graphmae.py --dataset Computers
python graphmae.py --dataset CS
python graphmae.py --dataset Physics
```

+ GraphMAE2
```bash
python graphmae2.py --dataset Cora
python graphmae2.py --dataset Citeseer
python graphmae2.py --dataset Pubmed --mode cat
python graphmae2.py --dataset Photo
python graphmae2.py --dataset Computers
python graphmae2.py --dataset CS
python graphmae2.py --dataset Physics
```

+ AUG-MAE
```bash
python augmae.py --dataset Cora
python augmae.py --dataset Citeseer
python augmae.py --dataset Pubmed --mode cat
python augmae.py --dataset Photo
python augmae.py --dataset Computers
python augmae.py --dataset CS
python augmae.py --dataset Physics
```

+ GiGaMAE
```bash
python gigamae.py --dataset Cora
python gigamae.py --dataset Citeseer
python gigamae.py --dataset Pubmed --mode cat
python gigamae.py --dataset Photo
python gigamae.py --dataset Computers
python gigamae.py --dataset CS
python gigamae.py --dataset Physics
```