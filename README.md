# Object detection in an urban environment [Submission]

## Tasks

- [x] Try 2 custom models (except EfficientNet) from the tensorflow model zoo and train them on the given waymo dataset and report their metrics.
- [x] Write a brief summary of the experiments and suggest the best model for this problem.
- [x] Try various data augmentation strategies to further improve your accuracy.
- [x] Add Tensorboard graphs in your notebook.
- [x] Update the model artificats path in the second notebook and generate the video output.

---
## Submission Write-Up

I've divided this into multiple parts that come chronologically in any machine learning project. Each sub-directory in the `project_writeup` directory contains a jupyter notebook that describes that step in detail along with figures and any related code snippet. This should help in reviewing each step of this project with clarity and ease.

The main code which trains the model on AWS can be found in the `1_model_training/solution_train_model.ipynb` notebook [here](1_model_training/solution_train_model.ipynb). It contains code to train 2 models, that are:

1. Faster R-CNN ResNet152 V1 640x640 - Pipeline config file for this model can be found [here](1_model_training/source_dir/faster_rcnn_pipeline.config) or at path `1_model_training/source_dir/faster_rcnn_pipeline.config`.
2. SSD MobileNet V2 FPNLite 640x640 - Pipeline config file for this model can be found [here](1_model_training/source_dir/ssd_mobilenet_v2_fpnlite.config) or at path `1_model_training/source_dir/ssd_mobilenet_v2_fpnlite.config`.

Each of the next section will go in-depth about my experiments and the outcomes that I got.

### Data Analysis

Since data is the core part of any machine learning based project, my first step was to extensively explore and analyze the data provided in the `train` and `val` folders of the S3 bucket, since this sets the direction of what sort of augmentations can be done along with what can be expected while running the trained model on the validation data.

Although this was not a requirement in the project rubric, this step helped me a lot as a starting point for training the models.

This writeup can be found here: [link](project_writeup/Exploratory-Data-Analysis/main.ipynb)

### Baseline Models

Here I selected `Faster R-CNN ResNet152 V1 640x640` and `SSD MobileNet V2 FPNLite 640x640` for custom training task. This section described the steps I did to train them and the metrics I got with the basic pipeline configuration that I selected from the tensorflow object detection sample configurations.

I touched a minimum required set of parameters that should be configured in order to kick off the training and get a baseline result. Furthermore, I compare the `mAP`, `recall` and `losses` amongst the models using tensorboard visualizations.

This writeup can be found here: [link](project_writeup/Baseline-Models/main.ipynb)

### Model Improvements

This section covers the `Optional 1` and `Optional 2` sections of the project rubric. Here I explore various data augmentations techniques and hyper parameters changes to improve upon the baseline perfomance I got in the last section.

Then there are tensorboard visualizations showing comparisions amongst the different versions of the models.

This writeup can be found here: [link](project_writeup/Model-Improvements/main.ipynb)
  
### Model Deployment
This section covers the task given in the **Model Deployment** rubric of the project where `2_deploy_model` notebook needs to be used for video generation.

I generated the videos using both the trained models. The one from `Faster R-CNN ResNet152 V1 640x640` was better in terms of detecting more objects in the scene and with more confidence, as well as in terms of inference times. The **average inference time** was 0.46101199865341186 seconds per invocation which I calculated using time module on each `predictor.predict()` call. The video can be found here - [link](./outputs/rcnn_output.avi).

Compared to the first model, the second one took higher average time which was unexpected. The **average inference time** for `SSD MobileNet V2 FPNLite 640x640` was 0.598960063457489 seconds. And performance was worse than the R-CNN as it had a hard time detecting pedestrians in the frames. The video can be found here - [link](./outputs/ssd_output.avi).