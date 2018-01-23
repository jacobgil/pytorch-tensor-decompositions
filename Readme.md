# PyTorch Tensor Decompositions

This is an implementation of Tucker and CP decomposition of convolutional layers.
A blog post about this can be found [here](https://jacobgil.github.io/deeplearning/tensor-decompositions-deep-learning).

It depends on [TensorLy](https://github.com/tensorly/tensorly) for performing tensor decompositions.

# Usage

- Train a model based on fine tuning VGG16: ``python main.py --train``.
- There should be a dataset with two categories. One directory for each category. Training data should go into a directory called 'train'.  Testing data should go into a directory called 'test'. This can be controlled with the flags --train_path and --test_path.
- I used the [Kaggle Cats/Dogs dataset.](https://www.kaggle.com/c/dogs-vs-cats)
- The model is then saved into a file called "model".

- Perform a decomposition: 
``python main.py --decompose``
This saves the new model into "decomposed_model".
It uses the Tucker decomposition by default. To use CP decomposition, pass --cp.

- Fine tune the decomposed model: ``python main.py --fine_tune``

# References

- CP Decomposition for convolutional layers is described here: https://arxiv.org/abs/1412.6553
- Tucker Decomposition for convolutional layers is described here: https://arxiv.org/abs/1511.06530
- VBMF for rank selection is described here: http://www.jmlr.org/papers/volume14/nakajima13a/nakajima13a.pdf
- VBMF code was taken from here: https://github.com/CasvandenBogaard/VBMF
- Tensorly: https://github.com/tensorly/tensorly
