```bash
cd examples/link_prediction
```


+ lrGAE-5
```bash
python lrgae_vu.py --dataset Cora --left 1 --right 1 --encoder_layers 1  --view AA
python lrgae_vu.py --dataset Citeseer --left 1 --right 1 --encoder_layers 1  --view AA
python lrgae_vu.py --dataset Pubmed --left 1 --right 1 --encoder_layers 1  --view AA  --encoder_dropout 0.2
```

+ lrGAE-6
```bash
python lrgae_vu.py --dataset Cora --left 2 --right 1 --encoder_layers 2  --view AA
python lrgae_vu.py --dataset Citeseer --left 2 --right 1 --encoder_layers 2  --view AA
python lrgae_vu.py --dataset Pubmed --left 2 --right 1 --encoder_layers 2  --view AA  --encoder_dropout 0.2
```
+ lrGAE-7
```bash
python lrgae_vu.py --dataset Cora --left 1 --right 1 --encoder_layers 1 --view AB
python lrgae_vu.py --dataset Citeseer --left 1 --right 1 --encoder_layers 1 --view AB
python lrgae_vu.py --dataset Pubmed --left 1 --right 1 --encoder_layers 1 --view AB  --encoder_dropout 0.2
```

+ lrGAE-8
```bash
python lrgae_vu.py --dataset Cora --left 2 --right 1 --encoder_layers 2 --view AB
python lrgae_vu.py --dataset Citeseer --left 2 --right 1 --encoder_layers 2 --view AB
python lrgae_vu.py --dataset Pubmed --left 2 --right 1 --encoder_layers 2 --view AB  --encoder_dropout 0.2
```

+ GAE$_f$
```bash
python lrgae_vv.py --dataset Cora --mask none --encoder_activation relu --encoder_dropout 0.5 --layer gcn
python lrgae_vv.py --dataset Citeseer --mask none --encoder_activation relu --encoder_dropout 0.5 --layer gcn
python lrgae_vv.py --dataset Pubmed --mask none --encoder_activation relu --encoder_dropout 0.5 --layer gcn
```


+ GAE
```bash
python lrgae_vu.py --dataset Cora --mask none --encoder_activation relu --encoder_dropout 0.5 --left 1 --right 1 --encoder_layers 2
python lrgae_vu.py --dataset Citeseer --mask none --encoder_activation relu --encoder_dropout 0.5 --left 1 --right 1 --encoder_layers 2
python lrgae_vu.py --dataset Pubmed --mask none --encoder_activation relu --encoder_dropout 0.5 --left 1 --right 1 --encoder_layers 2
```

+ S2GAE
python s2gae.py --dataset Cora
python s2gae.py --dataset Citeseer
python s2gae.py --dataset Pubmed

+ MaskGAE
python maskgae.py --dataset Cora
python maskgae.py --dataset Citeseer
python maskgae.py --dataset Pubmed --encoder_dropout 0.2

+ GraphMAE
python graphmae.py --dataset Cora
python graphmae.py --dataset Citeseer
python graphmae.py --dataset Pubmed

+ GraphMAE2
python graphmae2.py --dataset Cora
python graphmae2.py --dataset Citeseer
python graphmae2.py --dataset Pubmed

+ AUG-MAE
python augmae.py --dataset Cora
python augmae.py --dataset Citeseer
python augmae.py --dataset Pubmed

+ GiGaMAE
python gigamae.py --dataset Cora
python gigamae.py --dataset Citeseer
python gigamae.py --dataset Pubmed
