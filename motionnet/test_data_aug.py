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


class HorizontalFlip(object):
    """
    Flip the whole scene horizontally
    """
    def __init__(self, proba = 0.5):
        self.proba = proba

    def __call__(self, sample_list):
        """
        sample_list: list of dictionary containing

        - scenario_id: ()
        - obj_trajs: (nb_agents, nb_timesteps, nb_features)
        - track_index_to_predict ()

        ...

        it represents 1 scene !!
        The goal of this function is, from one trajectory, create a new one with a noisy starting position

        Une liste de 1 element : un dictionnaire =================> 1 liste de 1 element : un dictionnaire
               
        Flip horizontally = muiltiply the x coordinate by -1

        A modifier :

        - obj_trajs
        - obj_trajs_pos
        - obj_trajs_last_pos

        - center_gt_trajs
        - center_gt_trajs_src

        - map_polylines
        - map_polylines_center
        - map_polylines_mask

        """

        print("HorizontalFlip")
        dict_trajectory = sample_list[0] #on a une liste de 1 element, on prend le premier element
        new_dict_trajectory = dict_trajectory.copy()

        #flip horizontally
        new_obj_trajs = dict_trajectory['obj_trajs'].copy()
        new_obj_trajs_pos = dict_trajectory['obj_trajs_pos'].copy()
        new_obj_trajs_last_pos = dict_trajectory['obj_trajs_last_pos'].copy()

        new_obj_trajs[:, :, 0] = - dict_trajectory['obj_trajs'][:, :, 0]
        new_obj_trajs_pos[:, :, 0] = - dict_trajectory['obj_trajs_pos'][:, :, 0]
        new_obj_trajs_last_pos[:, 0] = - dict_trajectory['obj_trajs_last_pos'][:, 0]

        new_dict_trajectory['obj_trajs'] = new_obj_trajs
        new_dict_trajectory['obj_trajs_pos'] = new_obj_trajs_pos
        new_dict_trajectory['obj_trajs_last_pos'] = new_obj_trajs_last_pos

        ### MAP POLYLINES + MAP POLYLINES CENTER + MAP POLYLINES MASK
        new_map_polylines = dict_trajectory['map_polylines'].copy()
        new_map_polylines_center = dict_trajectory['map_polylines_center'].copy()

        new_map_polylines[:, :, 0] = - dict_trajectory['map_polylines'][:, :, 0]
        new_map_polylines_center[:, 0] = - dict_trajectory['map_polylines_center'][:, 0]

        new_dict_trajectory['map_polylines'] = new_map_polylines
        new_dict_trajectory['map_polylines_center'] = new_map_polylines_center

        #modifier le type de traj si LEFT_TURN(7) ou RIGHT_TURN(5

        if dict_trajectory['trajectory_type'] == 7:
            new_dict_trajectory['trajectory_type'] = 5
        elif dict_trajectory['trajectory_type'] == 5:
            new_dict_trajectory['trajectory_type'] = 7


        #center_gt_trajs a  modifier
        """
        new_center_gt_trajs = dict_trajectory['center_gt_trajs'].copy()
        new_center_gt_trajs_src = dict_trajectory['center_gt_trajs_src'].copy()

        new_center_gt_trajs[:, :, 0] = - dict_trajectory['center_gt_trajs'][:, :, 0]
        new_center_gt_trajs_src[:, :, 0] = - dict_trajectory['center_gt_trajs_src'][:, :, 0]

        new_dict_trajectory['center_gt_trajs'] = new_center_gt_trajs
        new_dict_trajectory['center_gt_trajs_src'] = new_center_gt_trajs_src
        """



        return [new_dict_trajectory]
    




class Apply_noise_on_start_pos_every_agent(object):
    def __init__(self, std_noise = 0.2):
        self.std_noise = std_noise

    def __call__(self, sample_list):
     
        dict_trajectory = sample_list[0]
        new_dict_trajectory = dict_trajectory.copy()
        new_obj_trajs = dict_trajectory['obj_trajs'].copy()
        noise = np.random.randn(new_obj_trajs.shape[0], 2) * self.std_noise
        new_obj_trajs[:, 0, 0:2] = dict_trajectory['obj_trajs'][:, 0, 0:2] + noise
        new_dict_trajectory['obj_trajs'] = new_obj_trajs
        return [new_dict_trajectory]
    

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

class Apply_noise_on_start_pos(object):
    """
    given a dict = 1 scene, apply noise on the starting position of the ego agent
    """
    
    def __init__(self, std_noise = 0.2):
        self.std_noise = std_noise

    def __call__(self, sample_list):


        """
        Create a noisy starting trajectory
        dict_trajectory: dictionary containing

        - scenario_id: ()
        - obj_trajs: (nb_agents, nb_timesteps, nb_features)
        - track_index_to_predict ()

        ...

        it represents 1 scene !!
        The goal of this function is, from one trajectory, create a new one with a noisy starting position


        Une liste de 1 element : un dictionnaire =================> 1 liste de 1 element : un dictionnaire
               
       """

        dict_trajectory = sample_list[0] #on a une liste de 1 element, on prend le premier element
        ## attention : on ne veut pas modifier le dictionnaire original !!
        #faire un copie profonde du dictionnaire
        new_dict_trajectory = dict_trajectory.copy()
        index_ego = dict_trajectory['track_index_to_predict']
        new_obj_trajs = dict_trajectory['obj_trajs'].copy()

        noise = np.random.randn(2) * self.std_noise

        new_obj_trajs[index_ego, 0, 0:2] = dict_trajectory['obj_trajs'][index_ego, 0, 0:2] + noise

        new_dict_trajectory['obj_trajs'] = new_obj_trajs

        #on doit retourner une liste de 1 element !
        return [new_dict_trajectory]
    


class Lane_Random_Occlusion(object):

    def __init__(self, proba = 0.5):
        self.proba = proba

    def __call__(self, sample_list):
        """
        modifie
         -  map_polylines
         -  map_polylines_mask
        """

        dict_trajectory = sample_list[0]
        new_dict_trajectory = dict_trajectory.copy()

        #on va masquer aleatoirement des segments de la carte
        new_map_polylines = dict_trajectory['map_polylines'].copy()
        new_map_polylines_mask = dict_trajectory['map_polylines_mask'].copy()

        nb_lanes = new_map_polylines.shape[0]
        
        #une lane represente une ligne de la carte. il y en a 256

        for i in range(nb_lanes):
            if np.random.rand() < self.proba:
                new_map_polylines_mask[i] = 0

        new_dict_trajectory['map_polylines'] = new_map_polylines
        new_dict_trajectory['map_polylines_mask'] = new_map_polylines_mask

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


    train_set = build_dataset(cfg, transform=[HorizontalFlip()])
    val_set = build_dataset(cfg,val=True)

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

    
    
    apply_noise = HorizontalFlip()
    sample = train_set[14]
    transformed_sample = apply_noise(sample)


    #pour visualiser l'effet d'une transformation
    plt_original = visualize_scene(sample[0])
    plt_transformed = visualize_scene(transformed_sample[0])
    plt_original.show()
    plt_transformed.show()


    nb_counted = 0
    for i, sample in enumerate(train_set):
        nb_counted += 1
    print("nb_counted = ",nb_counted)

    """
    Ce n'est qu'en iterant sur le train_loader que les transformations sont appliquees !!!!
    """













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



  

    #if the checkpoint path is not None, we validate the model
    if cfg.ckpt_path is not None:
        trainer.validate(model=model, dataloaders=val_loader, ckpt_path=cfg.ckpt_path)


    ######### LAUNCH TRAINING ########
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader,ckpt_path=cfg.ckpt_path)


    
if __name__ == '__main__':
    train()

