# UniTraj : last modified 27/05/24 for milestone 3
#### :student: **Authors**: 
- Victoria NGUYEN, 385211
- Valentin Perret, 327718


### :file_folder: Files modified and added by us

- `ptr.py` : We implemented the `temporal_attn_fn` function and the `spatial_attn_fn` function in the ptr.py file. 

- `train.py` : We added our metrics + shuffling the dataset at the beginning of each epoch.

- `base_dataset.py` : added transformations to the dataset.

- `config.yaml` and `ptr.yaml`: We added the hyperparameters we used for the training process + the path to the data.

- `PTR_dataset.py`
  
- `mmTransformer`: We implement a new model based on stacked transformer in the Unitraj framework. Link: https://arxiv.org/pdf/2103.11624

- `generate_prediction_mmTransformer.py`: We adapted the generate prediction file, for the mmTransformer model. To run this script, in the mmTransformer.py file, the Wandb related lines should be commented out, also loss should be set to 0, and the line with loss = get_loss commented, as well as the lines corresponding to the other loss function. This should enable to generate a csv file.

### :question:  What did we implemented ?

#### - mmTransformer

We have chosen to implement Multimodal Motion Prediction with Stacked Transformers (abbreviated MMTransformer).
The article is available at the following link: https://arxiv.org/pdf/2103.11624. Their github is available at the foloowing link: https://github.com/decisionforce/mmTransformer. We implemented their code for stacked transformers and adapted the dataset to it. We also added the loss function they mention in their paper. 


#### - Social and Temporal Attention

According the the unitraj original repo, the ptr.py file provided is based on a model called AUTOBOT. It is based on a transformer architecture with a temporal and spatial attention mechanism. The temporal attention mechanism is used to capture the dependencies between the past and future trajectories of the agents. The spatial attention mechanism is used to capture the dependencies between the different agents at the same time step.

We implemented the `temporal_attn_fn` function and the `spatial_attn_fn` function in the ptr.py file. We implemented our own version (commented), and we compare it to the original github repo as adviced by our TA : since the two implementations are equivalent, we chose the original one.




#### - Residual connections and layer normalization


- Effect of Layer Normalization on the validation metrics :

<img src="images/LN.png" width="1400" height="300">

- Effect of Residual connections on the validation metrics :

<img src="images/skipco.png" width="1400" height="300">




#### - Adding ponderated loss

There are a total of 8 possible trajectories in the test database, grouping together every possible movement. As it turns out, however, the train base does not cover all these cases uniformly. As a result, errors are mainly caused by the few trajectories in the train database. We can clearly see the dominance of one class, and a minority of rare cases.


To compensate and reduce the error caused by the lack of rare cases during training, we therefore tried to influence the loss: for each batch, we weighted the loss by the type of trajectory. In this way, rare trajectories will have a greater weighting in the loss: this impact will better penalize the network and therefore have an impact on learning. To calculate the weights, we used the frequency of each trajectory.


<img src="images/data.png" width="600" height="300">


#### - Adding transformations to the dataset

https://discuss.pytorch.org/t/understand-data-augmentation-in-pytorch/139720/8
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

In Pytorch, we can implement transformations to the dataset, while loading the data in the first place. To do this,

1) We modify the class used to load the data, which is derived from the `torch.utils.data.Dataset` class. We can add a `transform` parameter to the constructor of the class, which is a callable that takes a sample and returns the transformed sample. To be called in the PTR model, we also add the `transform` parameter to the class `PTRDataset` in the `ptr_dataset.py` file.

2) In pytorch, in order to implement data augmentation, we have to modify the `__getitem__` method of the class. This method is called when you index into the dataset (e.g. `dataset[i]`). Thus, every time that we iterate over the dataset, we apply the transformations to the data.

3) We created the `class Apply_noise_on_start_pos(object)` : this class is used to add noise to the starting position of the agents. The method `__call__` is called whenever a transformation is applied to the data. 

Here are some visualizations of the transformations we implemented :



- Horizontal Flip
<img src="images/flip.png" width="600" height="300">

- Random Lane Occlusion
<img src="images/occlusion.png" width="600" height="300">


### :gear: Training process

As you can see from the leaderboard, we made over twenty submissions. This is due to the fact that we tried to play on several hyperparameters, to get the best possible train on the one hand, and to reduce the training time on the other. 

**Main thing about the Hyperparameters**

- __scheduler__: We tried over 3 different schedulers, and we finally chose the `CosineAnnealingLR` scheduler, using this documentation : https://pytorch.org/docs/stable/optim.html. Details are provided in the report Milestone 1.

- __batch_size__: We tested different batch sizes, to have the best trade-off between training time and performance. We finally chose a batch size of 256.

- __learning_rate__: We tested different learning rates, and we finally chose a learning rate of 0.00075.


- __max_epochs__: We tested different number of epochs, and we finally chose 250 epochs.

The choices of the hyperparameters are detailed in the report Milestone 1.



### :chart_with_upwards_trend: Results

If we focus on the metric __minADE60__, we have a satisfying. During the training process, we reached a minADE60 of 1.16 on the validation set. It gave us a score of 1.006 on the leaderboard, which is our best score.


<img src="images/train_best.png" width="800" height="400">


### :computer: Visualization

Here is some predictions made by our model. We can see that the model is able to predict the future trajectory of the agents, even if the prediction is not perfect.



<img src="images/visualization_4.png" width="400" height="300">

<img src="images/visualization_12.png" width="400" height="300">



### :timer_clock: Training time

We use Scitas cluster for all our training. Our best score was obtained after **120 epochs**. The training time for 120 epochs was around **20 hours**.




**A Unified Framework for Cross-Dataset Generalization of Vehicle Trajectory Prediction**

UniTraj is a framework for vehicle trajectory prediction, designed by researchers from VITA lab at EPFL. 
It provides a unified interface for training and evaluating different models on multiple dataset, and supports easy configuration and logging. 
Powered by [Hydra](https://hydra.cc/docs/intro/), [Pytorch-lightinig](https://lightning.ai/docs/pytorch/stable/), and [WandB](https://wandb.ai/site), the framework is easy to configure, train and logging.
In this project, you will be using UniTraj to train and evalulate a model we call PTR (predictive transformer) on the data we have given to you.




___

## :books: Dataset


__input__ : A list of 1 element, which is a dictionnary, which represent 1 trajectory. The dictionnary contains the following keys : 

- key :  `scenario_id has size` :  (61,) ==> représente l'id du scénario. Il y a 61 scénarios dans le batch. 1 scénario = 1 vidéo

- key :  `obj_trajs has size` :  torch.Size([61, 15, 21, 39]) ==> représente les trajectoires des agents. Il y a 61 scénarios dans le batch. Il y a 15 agents dans chaque scénario. Chaque agent a 21 frames. Chaque frame contient 39 features
        
- key :  `obj_trajs_mask has size` :  torch.Size([61, 15, 21]) ==> représente le masque des trajectoires des agents. Il y a 61 scénarios dans le batch. Il y a 15 agents dans chaque scénario. Chaque agent a 21 frames
        
        
- key :  `track_index_to_predict` has size :  torch.Size([61]) ==> représente l'index de l'agent à prédire. Il y a 61 scénarios dans le batch
        
- key :  `obj_trajs_pos has size` :  torch.Size([61, 15, 21, 3]) ==> représente la position des agents. Il y a 61 scénarios dans le batch. Il y a 15 agents dans chaque scénario. Chaque agent a 21 frames. Chaque frame contient 3 features
        
- key :  `obj_trajs_last_pos` has size :  torch.Size([61, 15, 3]) ==> représente la dernière position des agents. obj_trajs_last_pos = obj_trajs_pos[:,-1,:]
        
        
- key :  `center_objects_world` has size :  torch.Size([61, 10]) ==> représente le centre des objets dans le monde. Il y a 61 scénarios dans le batch. Chaque scénario contient 10 objets
        
        
- key :  `center_objects_id` has size :  (61,)
        
- key :  `center_objects_type` has size :  (61,)
        
- key :  `map_center` has size :  torch.Size([61, 3]) ==> représente le centre de la carte. Il y a 61 scénarios dans le batch. Chaque scénario contient 3 features
        
        
- key :  `obj_trajs_future_state` has size :  torch.Size([61, 15, 60, 4]) ==> représente les états futurs des agents. Il y a 61 scénarios dans le batch. Il y a 15 agents dans chaque scénario. Chaque agent a 60 frames. Chaque frame contient 4 features : x, y, vx, vy
        
        
- key :  `obj_trajs_future_mask` has size :  torch.Size([61, 15, 60])
        
- key :  `center_gt_trajs` has size :  torch.Size([61, 60, 4]) ==> représente les trajectoires des agents. Il y a 61 scénarios dans le batch. Chaque scénario contient 60 frames. Chaque frame contient 4 features : x, y, vx, vy
        
- key :  `center_gt_trajs_mask` has size :  torch.Size([61, 60])
        
- key :  `center_gt_final_valid_idx `has size :  torch.Size([61])
        
        
- key :  `center_gt_trajs_src` has size :  torch.Size([61, 81, 10]) ==> represente le ce
        
- key :  `map_polylines` has size :  torch.Size([61, 256, 20, 9]) ==> représente les polylines de la carte. Il y a 61 scénarios dans le batch. Il y a 256 polylines dans chaque scénario. Chaque polyline a 20 points. Chaque point contient 9 features
        
- key :  `map_polylines_mask` has size :  torch.Size([61, 256, 20])
        
- key :  `map_polylines_center ` has size :  torch.Size([61, 256, 3])
        
- key : ` dataset_name` has size :  (61,)
        
- key :  kalman_difficulty has size :  torch.Size([61, 3])
        
- key :  trajectory_type has size :  torch.Size([61]) ==> de 0 a 7










## Installation

First start by cloning the repository:
```bash
git clone https://github.com/vita-epfl/unitraj-DLAV.git
cd unitraj-DLAV
```

Then make a virtual environment and install the required packages. 
```bash
python3 -m venv venv
source venv/bin/activate

# Install MetaDrive Simulator
cd ~/  # Go to the folder you want to host these two repos.
git clone https://github.com/metadriverse/metadrive.git
cd metadrive
pip install -e .

# Install ScenarioNet
cd ~/  # Go to the folder you want to host these two repos.
git clone https://github.com/metadriverse/scenarionet.git
cd scenarionet
pip install -e .
```

Finally, install Unitraj and login to wandb via:
```bash
cd unitraj-DLAV # Go to the folder you cloned the repo
pip install -r requirements.txt
pip install -e .
wandb login
```
If you don't have a wandb account, you can create one [here](https://wandb.ai/site). It is a free service for open-source projects and you can use it to log your experiments and compare different models easily.


You can verify the installation of UniTraj via running the training script:
```bash
python train.py method=ptr
```
The incomplete PTR model will be trained on several samples of data available in `motionnet/data_samples`.

## Code Structure
There are three main components in UniTraj: dataset, model and config.
The structure of the code is as follows:
```
motionnet
├── configs
│   ├── config.yaml
│   ├── method
│   │   ├── ptr.yaml
├── datasets
│   ├── base_dataset.py
│   ├── ptr_dataset.py
├── models
│   ├── ptr
│   ├── base_model
├── utils
```
There is a base config, dataset and model class, and each model has its own config, dataset and model class that inherit from the base class.

## Data
You can access the data [here](https://drive.google.com/file/d/1mBpTqM5e_Ct6KWQenPUvNUBJWHn3-KUX/view?usp=sharing). For easier use on SCITAS, we have also provided the dataset in scitas on `/work/vita/datasets/DLAV_unitraj`. We have provided a train and validation set, as well as three testing sets of different difficulty levels: easy, medium and hard.
You will be evaluating your model on the easy test set for the first milestone, and the medium and hard test sets for the second and third milestones, respectively. 
Don't forget to put the path to the real data in the config file.


## Your Task
Your task is to complete the PTR model and train it on the data we have provided. 
The model is a transformer-based model that takes the past trajectory of the vehicle and its surrounding agents, along with the map, and predicts the future trajectory.
![system](https://github.com/vita-epfl/unitraj-DLAV/blob/main/docs/assets/PTR.png?raw=true)
This is the architecture of the encoder part of model (where you need to implement). Supposing we are given the past t time steps for M agents and we have a feature vector of size $d_K$ for each agent at each time step, the encoder part of the model consists of the following steps:
1. Add positional encoding to the input features at the time step dimension for distinguish between different time steps.
2. Perform the temporal attention to capture the dependencies between the trajectories of each agent separately.
3. Perform the spatial attention to capture the dependencies between the different agents at the same time step.
These steps are repeated L times to capture the dependencies between the agents and the time steps.

The model is implemented in `motionnet/models/ptr/ptr_model.py` and the config is in `motionnet/configs/method/ptr.yaml`. 
Take a look at the model and the config to understand the structure of the model and the hyperparameters.

You are asked to complete three parts of the model in `motionnet/models/ptr/ptr_model.py`:
1. The `temporal_attn_fn` function that computes the attention between the past trajectory and the future trajectory.
2. The `spatial_attn_fn` function that computes the attention between different agents at the same time step.
3. The encoder part of the model in the `_forward` function. 

You can find the instructions and some hints in the file itself. 

## Submission
You could follow the steps in the [easy kaggle competition](https://www.kaggle.com/competitions/dlav-vehicle-trajectory-prediction-2024/overview) to submit your results and compare them with the other students in the leaderboard.
Here are the [medium](https://www.kaggle.com/competitions/dlav-vehicle-trajectory-prediction-medium/overview) and [hard](https://www.kaggle.com/competitions/dlav-vehicle-trajectory-prediction-hard/overview) competitions for the second and third milestones, respectively.
We have developed a submission script for your convenience. You can run the following command to generate the submission file:
```bash
python generate_predictions.py method=ptr
```
Before running the above command however, you need to put the path to the checkpoint of your trained model on the config file under `ckpt_path`. You can find the checkpoint of your trained model in the `lightning_logs` directory in the root directory of the project. 
For example, if you have trained your model for 10 epochs, you will find the checkpoint in `lightning_logs/version_0/checkpoints/epoch=10-val/brier_fde=30.93.ckpt`. You need to put the path to this file in the config file.

Additionally, for the `val_data_path` in the config file, you need to put the path to the test data you want to evaluate your model on. For the easy milestone, you can put the path to the easy test data, and for the second and third milestones, you can put the path to the medium and hard test data, respectively.

The script will generate a file called `submission.csv` in the root directory of the project. You can submit this file to the kaggle competition. As this file could be big, we suggest you to compress it before submitting it.
# UnitrajPrediction