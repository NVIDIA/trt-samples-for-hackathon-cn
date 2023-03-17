#

## Steps to run

+ need Anaconda

```shell
conda install caffe # failed using "pip install"
python3 buildModelInTensorFlow1.py
mmconvert -sf tensorflow -in ./model.ckpt.meta -iw ./model.ckpt --dstNodeName y -df caffe -om model
# edit file model.prototxt, near Reshape node, insert two "dim: 1" between "dim: 1 dim: 3136", making it as "dim: 1 dim: 1 dim: 1 dim: 3136"
# if not inserting, error information is listed in result-Dim2.txt and result-Dim3.txt
python3 main.py
```

## Output for reference: result.log
