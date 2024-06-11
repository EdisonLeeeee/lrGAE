```bash
cd examples/graph_classification
```


+ lrGAE-6
```bash
python lrgae_vu.py --dataset IMDB-BINARY --left 2 --right 1 --view AA --pooling mean
python lrgae_vu.py --dataset IMDB-MULTI --left 2 --right 1 --view AA --pooling mean
python lrgae_vu.py --dataset PROTEINS --left 2 --right 1 --view AA --pooling max
python lrgae_vu.py --dataset COLLAB --left 2 --right 1 --view AA --pooling max
python lrgae_vu.py --dataset MUTAG --left 2 --right 1 --view AA --pooling sum
python lrgae_vu.py --dataset REDDIT-BINARY --left 2 --right 1 --view AA --pooling max
python lrgae_vu.py --dataset NCI1 --left 2 --right 1 --view AA --pooling max
```
+ lrGAE-7
```bash
python lrgae_vu.py --dataset IMDB-BINARY --view AB --pooling mean
python lrgae_vu.py --dataset IMDB-MULTI --view AB --pooling mean
python lrgae_vu.py --dataset PROTEINS --view AB --pooling max
python lrgae_vu.py --dataset COLLAB --view AB --pooling max
python lrgae_vu.py --dataset MUTAG --view AB --pooling sum
python lrgae_vu.py --dataset REDDIT-BINARY --view AB --pooling max
python lrgae_vu.py --dataset NCI1 --view AB --pooling max
```

+ lrGAE-8
```bash
python lrgae_vu.py --dataset IMDB-BINARY --left 2 --right 1 --view AB --pooling mean
python lrgae_vu.py --dataset IMDB-MULTI --left 2 --right 1 --view AB --pooling mean
python lrgae_vu.py --dataset PROTEINS --left 2 --right 1 --view AB --pooling max
python lrgae_vu.py --dataset COLLAB --left 2 --right 1 --view AB --pooling max
python lrgae_vu.py --dataset MUTAG --left 2 --right 1 --view AB --pooling sum
python lrgae_vu.py --dataset REDDIT-BINARY --left 2 --right 1 --view AB --pooling max
python lrgae_vu.py --dataset NCI1 --left 2 --right 1 --view AB --pooling max
```

+ GAE$_f$
```bash
python gae_f.py --dataset IMDB-BINARY --pooling mean
python gae_f.py --dataset IMDB-MULTI --pooling mean
python gae_f.py --dataset PROTEINS --pooling max
python gae_f.py --dataset COLLAB --pooling max
python gae_f.py --dataset MUTAG --pooling sum
python gae_f.py --dataset REDDIT-BINARY --pooling max
python gae_f.py --dataset NCI1 --pooling max
```

+ GAE
```bash
python gae.py --dataset IMDB-BINARY --pooling mean
python gae.py --dataset IMDB-MULTI --pooling mean
python gae.py --dataset PROTEINS --pooling max
python gae.py --dataset COLLAB --pooling max
python gae.py --dataset MUTAG --pooling sum
python gae.py --dataset REDDIT-BINARY --pooling max
python gae.py --dataset NCI1 --pooling max
```

+ S2GAE
```bash
python s2gae.py --dataset IMDB-BINARY --pooling mean
python s2gae.py --dataset IMDB-MULTI --pooling mean
python s2gae.py --dataset PROTEINS --pooling max
python s2gae.py --dataset COLLAB --pooling max
python s2gae.py --dataset MUTAG --pooling sum
python s2gae.py --dataset REDDIT-BINARY --pooling max
python s2gae.py --dataset NCI1 --pooling max
```

+ MaskGAE
```bash
python maskgae.py --dataset IMDB-BINARY --pooling mean
python maskgae.py --dataset IMDB-MULTI --pooling mean
python maskgae.py --dataset PROTEINS --pooling max
python maskgae.py --dataset COLLAB --pooling max
python maskgae.py --dataset MUTAG --pooling sum
python maskgae.py --dataset REDDIT-BINARY --pooling max
python maskgae.py --dataset NCI1 --pooling max
```

+ GraphMAE
```bash
python graphmae.py --dataset IMDB-BINARY --pooling mean
python graphmae.py --dataset IMDB-MULTI --pooling mean
python graphmae.py --dataset PROTEINS --pooling max
python graphmae.py --dataset COLLAB --pooling max
python graphmae.py --dataset MUTAG --pooling sum
python graphmae.py --dataset REDDIT-BINARY --pooling max
python graphmae.py --dataset NCI1 --pooling max
```

+ GraphMAE2
```bash
python graphmae2.py --dataset IMDB-BINARY --pooling mean
python graphmae2.py --dataset IMDB-MULTI --pooling mean
python graphmae2.py --dataset PROTEINS --pooling max
python graphmae2.py --dataset COLLAB --pooling max
python graphmae2.py --dataset MUTAG --pooling sum
python graphmae2.py --dataset REDDIT-BINARY --pooling max
python graphmae2.py --dataset NCI1 --pooling max
```

+ AUG-MAE
```bash
python augmae.py --dataset IMDB-BINARY --pooling mean
python augmae.py --dataset IMDB-MULTI --pooling mean
python augmae.py --dataset PROTEINS --pooling max
python augmae.py --dataset COLLAB --pooling max
python augmae.py --dataset MUTAG --pooling sum
python augmae.py --dataset REDDIT-BINARY --pooling max
python augmae.py --dataset NCI1 --pooling max
```

