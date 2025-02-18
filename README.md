# Smooth Gating Functions

There's the differentiable top-k method:
- sum of activations is exactly k
- tau controls sparsity
put these together, you've got about k active and the rest close to zero.

The softmax method, 
- sum of activations is exactly 1
- tau controls sparsity

If tau is small then the DTk method will push two activations close to one and the rest close to zero. The softmax method pushes one value to a large value and the rest to zero.
