# Single Object Tracking In Videos

Image features are extracted with pretrained VGG-16 network.
FC network predicts object bounding boxes given the first object bounding box and frame.
See the implemetation details on the [report](https://github.com/mervekantarci/SingleObjectTracker/blob/main/report.pdf).

Feature tensor files, model files and GIFs are saved to the working directory. 
Please make sure there is enough space.

## Compatibility
This code is tested with only Python 3.7. CUDA support is strongly advised!
