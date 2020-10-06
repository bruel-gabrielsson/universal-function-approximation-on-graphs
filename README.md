# Universal Function Approximation on Graphs

This repository is the official PyTorch implementation of the experiments in the following paper:

Rickard Brüel-Gabrielsson. Universal Function Approximation on Graphs

[arXiv](https://arxiv.org/pdf/2003.06706.pdf)

The code is based on the code of [How Powerful are Graph Neural Networks?](https://github.com/weihua916/powerful-gnns)

## Compiling C++ Extensions

If you haven't already, clone the repository

You are now ready to compile extensions.  PyTorch tutorial on extensions [here](https://pytorch.org/tutorials/advanced/cpp_extension.html)

*Important*: in environment, it seems like using the pytorch conda channel is important
```bash
source activate environment
conda install pytorch torchvision -c pytorch
```
Compilation uses python's `setuptools` module.

To complile (from home directory):
```bash
source activate environment
python setup.py install --record files.txt
```
You should now have the package available in your environment. You can run the above command any time you modify the source code, and the package on your path should update.

__MacOS Information:__
If PyTorch was compiled using `clang++`, you may run into issues if `pip` defaults to `g++`.  You can make `pip` use `clang++` by setting the `CXX` environment variable.  The `CPPFLAGS` environment variable also needs to be set to look at `libc++` to avoid compatibility issues with the PyTorch headers.  The `MACOSX_DEPLOYMENT_TARGET` environment variable may also need to be set (set the target to be whatever your OS version is).
```bash
export CXX=/usr/bin/clang++
export CPPFLAGS="-stdlib=libc++"
export MACOSX_DEPLOYMENT_TARGET=$(sw_vers -productVersion)
pip install --verbose git+https://github.com/bruel-gabrielsson/TopologyLayer.git
```
