# Neural Network Training Repository

Various examples using tensorflow for building neural networks. 

# Installation

Pull latest tensorflow docker image:
```bash
docker pull tensorflow/tensorflow
```

Test if docker container runs:
```bash
docker run -it --rm tensorflow/tensorflow \
   python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```

# Run
Run tensorflow example 
```
./start.bat {directory}
```

Example with Xor-NN-Model:
```
./start.bat xor
```