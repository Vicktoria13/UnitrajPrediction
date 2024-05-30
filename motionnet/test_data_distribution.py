import pytorch_lightning as pl
import torch
from itertools import chain
torch.set_float32_matmul_precision('medium')
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from models import build_model
from datasets import build_dataset
from utils.utils import set_seed
from pytorch_lightning.callbacks import ModelCheckpoint  # Import ModelCheckpoint
import hydra
from omegaconf import DictConfig, OmegaConf
import os
from utils.visualization import *
import matplotlib.pyplot as plt



#this function launches the training process
#the input is the config file
#the config file is a yaml file that contains all the hyperparameters and the paths to the data
#the config file is loaded using hydra
#the config file is then merged with the method section of the config file
#the model is built using the build_model function

#output : the model is trained using the train_loader and val_loader
#the output is saved in the wandb logger
#pour trouver l'output, il faut aller sur le site de wandb et chercher le projet motionnet

   

def compute_traj_weights(train_set):
    """
    Input : train_set is a list with nb_scenes trajectories (15). 
    train_set[i] is a list with 1 element : a dictionary :
        train_set[i][0] is a dictionary with the following keys :
            - scenario_id: ()
            - obj_trajs: (nb_agents, nb_timesteps, nb_features)
            - track_index_to_predict ()
            ...
            - map_polylines: (nb_polylines, nb_points, 2) ...
            - trajectory_type from 0 to 7

    output : list_weights is a list with 0 to 7 elements.
    class_weights = 1 - (number_of_occurences_of_trajectory_type / total_number_of_occurences)
    class_weights[3] is the weight of the trajectory type 3

    
    """

    list_weights = [0,0,0,0,0,0,0,0]

    total = 0
    for scene in train_set:
        #scene is a list with 1 element : a dictionary
        traj_type = scene[0]['trajectory_type'] #va de 0 a 7
        list_weights[traj_type] += 1
        total += 1

    for i in range(8):
        list_weights[i] = 1 - (list_weights[i] / total)

    return list_weights + np.array([1,1,1,1,1,1,1,1])




class ApplyRandomMasking_first_time_step(object):
    def __init__(self, intervalle_sup_masking = 10):
        
        ## on va appliquer un maskage aleatoire sur les intervalle_sup_masking premiers pas de temps
        self.intervalle_sup_masking = intervalle_sup_masking
        

    def __call__(self, sample_list):
        """
        On va choisir randomly des agents (autre que ego) et on va masquer randomment leurs
        positions pendant les 10 premiers pas de temps

        Pour cela, on touche a l'attribut obj_trajs_mask qui est de taille (nb_agents, nb_timesteps). On va
        aleatoirement mettre False a certains endroits pendant les 10 premiers pas de temps (sauf sur l'ego agent)

        A modifier :
        obj_trajs_mask : pour chaque agent, on masque les 10 premiers pas de temps
        center_gt_trajs_mask
        """

        dict_trajectory = sample_list[0]
        new_dict_trajectory = dict_trajectory.copy()


        new_obj_trajs_mask = dict_trajectory['obj_trajs_mask'].copy()

        nb_agents = new_obj_trajs_mask.shape[0]

        for i in range(nb_agents):
            #si c'est l'ego agent, on ne fait rien
            if i == dict_trajectory['track_index_to_predict']:
                continue

            # sinon, avec une proba mask_prob, on masque les 10 premiers pas de temps
            else:
                mask = np.random.rand(self.intervalle_sup_masking) < 0.5 #50% de chance de masquer un pas de temps
                new_obj_trajs_mask[i, 0:self.intervalle_sup_masking] = mask

        #

        new_dict_trajectory['obj_trajs_mask'] = new_obj_trajs_mask
        return [new_dict_trajectory]


class Apply_noise_on_end_pos_every_agent(object):
    def __init__(self, std_noise = 0.2):
        self.std_noise = std_noise

    def __call__(self, sample_list):
        """ 
        a mofiier :
        obj_trajs pour chaque agent, on ajoute du bruit sur la derniere position
        obj_trajs_pos en consequence
        obj_trajs_last_pos en consequence
        """

        dict_trajectory = sample_list[0]

        new_dict_trajectory = dict_trajectory.copy()

        new_obj_trajs = dict_trajectory['obj_trajs'].copy()
        new_obj_trajs_pos = dict_trajectory['obj_trajs_pos'].copy()
        new_obj_trajs_last_pos = dict_trajectory['obj_trajs_last_pos'].copy()

        noise = np.random.randn(new_obj_trajs.shape[0], 2) * self.std_noise
        new_obj_trajs[:, -1, 0:2] = dict_trajectory['obj_trajs'][:, -1, 0:2] + noise
        new_obj_trajs_pos[:, -1, 0:2] = dict_trajectory['obj_trajs_pos'][:, -1, 0:2] + noise
        new_obj_trajs_last_pos[:, 0:2] = dict_trajectory['obj_trajs_last_pos'][:, 0:2] + noise

        new_dict_trajectory['obj_trajs'] = new_obj_trajs
        new_dict_trajectory['obj_trajs_pos'] = new_obj_trajs_pos
        new_dict_trajectory['obj_trajs_last_pos'] = new_obj_trajs_last_pos

        return [new_dict_trajectory]

@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg):

 
    # permet de fixer la seed pour avoir des resultats reproductibles
    set_seed(cfg.seed)

    # Merge the method section of the config file with the config file

    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)

    # ici, initie le model
    model = build_model(cfg)


    train_set = build_dataset(cfg, transform=[ApplyRandomMasking_first_time_step(),Apply_noise_on_end_pos_every_agent()])
    val_set = build_dataset(cfg,val=True)

    """
    train_set is a list with nb_scenes trajectories (15)
    train_set[i] is a list with 1 element : a dictionary :
        train_set[i][0] is a dictionary with the following keys :
            - scenario_id: ()
            - obj_trajs: (nb_agents, nb_timesteps, nb_features)
            - track_index_to_predict ()
            ...
            - map_polylines: (nb_polylines, nb_points, 2) ...
            - trajectory_type

    trajectory_type : {0, 1 ... 7}
    class TrajectoryType:
            STATIONARY = 0
            STRAIGHT = 1
            STRAIGHT_RIGHT = 2
            STRAIGHT_LEFT = 3
            RIGHT_U_TURN = 4
            RIGHT_TURN = 5
            LEFT_U_TURN = 6
            LEFT_TURN = 7


    """
 

    plt_histo = visualize_histogram_distribution_trajectory_type(val_set)
    plt_histo.savefig('histo_val.png')

    #weights = compute_traj_weights(train_set)
    #print("weights : ",weights)



    train_batch_size = max(cfg.method['train_batch_size'] // len(cfg.devices) // train_set.data_chunk_size,1)
    eval_batch_size = max(cfg.method['eval_batch_size'] // len(cfg.devices) // val_set.data_chunk_size,1)

    call_backs = []

    checkpoint_callback = ModelCheckpoint(
        monitor='val/minADE6',    # Replace with your validation metric
        filename='{epoch}-{val/minADE6:.2f}',
        save_top_k=1,
        mode='min',            # 'min' for loss/error, 'max' for accuracy
    )

    call_backs.append(checkpoint_callback)


    train_loader = DataLoader(
        train_set, batch_size=train_batch_size, num_workers=cfg.load_num_workers, shuffle=True,drop_last=False,
    collate_fn=train_set.collate_fn)

    val_loader = DataLoader(
        val_set, batch_size=eval_batch_size, num_workers=cfg.load_num_workers, shuffle=False, drop_last=False,
    collate_fn=train_set.collate_fn)
    


    trainer = pl.Trainer(
        max_epochs=cfg.method.max_epochs,
        logger= WandbLogger(project="motionnet", name=cfg.exp_name),
        devices=1 if cfg.debug else cfg.devices,
        gradient_clip_val=cfg.method.grad_clip_norm,
        accelerator="cpu" if cfg.debug else "gpu",
        profiler="simple",
        strategy="auto" if cfg.debug else "ddp",
        callbacks=call_backs,
    )

    print("len train_loader : ",len(train_loader)) #len(train_loader) represente le nombre de batchs

 

    
if __name__ == '__main__':
    train()

