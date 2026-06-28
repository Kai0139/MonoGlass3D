# Implementation For MonoGlass3D

## Dataset 
Download from [Here](https://pan.baidu.com/s/1u0n4j7UffxtfkH0B88yLhw?pwd=mg3d)

## Trained Weights
Download from [Here](https://pan.baidu.com/s/1xfcLB5R6QGJDnEsVuB23UA?pwd=mg3d)

## Run Evaluation
First change the `weight_path` to the path of weight, then run
```bash
python3 evals/eval_3d_nn.py
```

## Train
First change the `root_dir` to directory of downloaded dataset, then run
```bash
python3 train/train3d_dam.py
```

