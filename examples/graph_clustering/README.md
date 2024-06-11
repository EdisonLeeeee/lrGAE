```bash
cd examples/graph_clustering
```


+ lrGAE-5
```bash
python lrgae_vu.py --dataset Cora --left 2 --right 2 --view AA
python lrgae_vu.py --dataset Citeseer --left 2 --right 2 --view AA
python lrgae_vu.py --dataset Pubmed --left 2 --right 2 --view AA
python lrgae_vu.py --dataset Photo --left 2 --right 2 --view AA
python lrgae_vu.py --dataset Computers --left 2 --right 2 --view AA
python lrgae_vu.py --dataset CS --left 2 --right 2 --view AA
python lrgae_vu.py --dataset Physics --left 2 --right 2 --view AA
```

+ lrGAE-6
```bash
python lrgae_vu.py --dataset Cora --left 2 --right 1 --view AA
python lrgae_vu.py --dataset Citeseer --left 2 --right 1 --view AA
python lrgae_vu.py --dataset Pubmed --left 2 --right 1 --view AA
python lrgae_vu.py --dataset Photo --left 2 --right 1 --view AA
python lrgae_vu.py --dataset Computers --left 2 --right 1 --view AA
python lrgae_vu.py --dataset CS --left 2 --right 1 --view AA
python lrgae_vu.py --dataset Physics --left 2 --right 1 --view AA
```
+ lrGAE-7
```bash
python lrgae_vu.py --dataset Cora --view AB
python lrgae_vu.py --dataset Citeseer --view AB
python lrgae_vu.py --dataset Pubmed --view AB
python lrgae_vu.py --dataset Photo --view AB
python lrgae_vu.py --dataset Computers --view AB
python lrgae_vu.py --dataset CS --view AB
python lrgae_vu.py --dataset Physics --view AB
```

+ lrGAE-8
```bash
python lrgae_vu.py --dataset Cora --left 2 --right 1 --view AB
python lrgae_vu.py --dataset Citeseer --left 2 --right 1 --view AB
python lrgae_vu.py --dataset Pubmed --left 2 --right 1 --view AB
python lrgae_vu.py --dataset Photo --left 2 --right 1 --view AB
python lrgae_vu.py --dataset Computers --left 2 --right 1 --view AB
python lrgae_vu.py --dataset CS --left 2 --right 1 --view AB
python lrgae_vu.py --dataset Physics --left 2 --right 1 --view AB
```

+ GAE$_f$
```bash
python gae_f.py --dataset Cora
python gae_f.py --dataset Citeseer
python gae_f.py --dataset Pubmed
python gae_f.py --dataset Photo
python gae_f.py --dataset Computers
python gae_f.py --dataset CS
python gae_f.py --dataset Physics
```

+ GAE
```bash
python gae.py --dataset Cora
python gae.py --dataset Citeseer
python gae.py --dataset Pubmed
python gae.py --dataset Photo
python gae.py --dataset Computers
python gae.py --dataset CS
python gae.py --dataset Physics
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
python graphmae.py --dataset CS --encoder_channels 512
python graphmae.py --dataset Physics
```

+ GraphMAE2
```bash
python graphmae2.py --dataset Cora
python graphmae2.py --dataset Citeseer
python graphmae2.py --dataset Pubmed --mode cat
python graphmae2.py --dataset Photo
python graphmae2.py --dataset Computers
python graphmae2.py --dataset CS --encoder_channels 512
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