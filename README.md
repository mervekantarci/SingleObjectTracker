# Single Object Tracking In Videos

Image features are extracted with pretrained VGG-16 [[1]](#1) network. The later layers (FC network) learns to predict object bounding boxes given the first object bounding box and frame.
See the implemetation details on the [report](https://github.com/mervekantarci/SingleObjectTracker/blob/main/report.pdf).

Feature tensor files, model files and GIFs are saved to the working directory. 
Please make sure there is enough space.

## Compatibility
This code is tested with Python 3.7. You should install compatible PyTorch version (CUDA strongly advised!).


## References
<a id="1">[1]</a> 
S. Liu and W. Deng, "Very deep convolutional neural network based image classification using small training sample size," 2015 3rd IAPR Asian Conference on Pattern Recognition (ACPR), Kuala Lumpur, Malaysia, 2015, pp. 730-734, doi: 10.1109/ACPR.2015.7486599.
