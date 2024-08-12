```bash
cd examples/hetero_node_classification
```

+ lrGAE-5
```bash
python lrgae_vu.py --dataset DBLP --left 2 --right 2 --view AA
python lrgae_vu.py --dataset ACM --left 2 --right 2 --view AA
python lrgae_vu.py --dataset IMDB --left 2 --right 2 --view AA
```

+ lrGAE-6
```bash
python lrgae_vu.py --dataset DBLP --left 2 --right 1 --view AA
python lrgae_vu.py --dataset ACM --left 2 --right 1 --view AA
python lrgae_vu.py --dataset IMDB --left 2 --right 1 --view AA
```
+ lrGAE-7
```bash
python lrgae_vu.py --dataset DBLP --view AB
python lrgae_vu.py --dataset ACM --view AB
python lrgae_vu.py --dataset IMDB --view AB --encoder_dropout 0
```

+ lrGAE-8
```bash
python lrgae_vu.py --dataset DBLP --left 2 --right 1 --view AB
python lrgae_vu.py --dataset ACM --left 2 --right 1 --view AB
python lrgae_vu.py --dataset IMDB --left 2 --right 1 --view AB
```

+ GAE
```bash
python gae.py --dataset DBLP
python gae.py --dataset ACM
python gae.py --dataset IMDB --encoder_dropout 0
```

+ MaskGAE
```bash
python maskgae.py --dataset DBLP
python maskgae.py --dataset ACM
python maskgae.py --dataset IMDB --encoder_dropout 0
```

+ S2GAE
```bash
python s2gae.py --dataset DBLP
python s2gae.py --dataset ACM
python s2gae.py --dataset IMDB --encoder_dropout 0
```