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



def apply_noise_on_starting_position(batch):
    """
    cree une copie du batch : on ajoute du bruit sur la position de depart de l'agent ego
    pour chacune des scenes du batch
    """

    #.clone() permet de creer une copie du tensor
    std_noise = 0.1
    input_dict = batch['input_dict']
    new_input_dict = input_dict.copy()
    new_input_dict['obj_trajs'] = input_dict['obj_trajs'].clone()

    nb_scenes = input_dict['obj_trajs'].shape[0]

    for i in range(nb_scenes):
        #on ajoute du bruit sur la position de depart de l'agent ego
        noise = torch.randn(2) * std_noise
        new_input_dict['obj_trajs'][i, 0, 0, 0:2] = input_dict['obj_trajs'][i, 0, 0, 0:2] + noise


    #create a copy of the batch: and replace the starting position of the ego agent with noise
    new_batch = batch.copy()
    new_batch['input_dict'] = new_input_dict
    return new_batch





            

def create_noisy_starting_trajectory(dict_trajectory):
    """
    Create a noisy starting trajectory
    dict_trajectory: dictionary containing

    - scenario_id: ()
    - obj_trajs: (nb_agents, nb_timesteps, nb_features)
    - track_index_to_predict ()

    ...

    it represents 1 scene !!
    The goal of this function is, from one trajectory, create a new one with a noisy starting position
    """

    std_noise = 0.1
    new_dict_trajectory = dict_trajectory.copy()

    index_ego = dict_trajectory['track_index_to_predict']

    #copy the trajectory : it is a numpy array
    new_obj_trajs = dict_trajectory['obj_trajs'].copy()

    #modify the starting position of the ego agent
    noise = np.random.randn(2) * std_noise

    #print("before : ",new_obj_trajs[index_ego, 0, 0:2])
    new_obj_trajs[index_ego, 0, 0:2] = dict_trajectory['obj_trajs'][index_ego, 0, 0:2] + noise
    #print("after : ",new_obj_trajs[index_ego, 0, 0:2])


    return new_dict_trajectory


    


    

@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg):

 
    # permet de fixer la seed pour avoir des resultats reproductibles
    set_seed(cfg.seed)

    # Merge the method section of the config file with the config file

    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)

    # ici, initie le model
    model = build_model(cfg)

    
    train_set = build_dataset(cfg)
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




    ############################ Augmentation of the training set ############################
    #on ajoute du bruit sur la position de depart de l'agent ego
    #on cree une copie du batch : on ajoute du bruit sur la position de depart de l'agent ego
    #pour chacune des scenes du batch

    #.clone() permet de creer une copie du tensor


   #affiche les attributs de train_set est une liste de 1 element. train_set[0] est un dictionnaire
    #ŧrain_set contient 125 listes
    
  

    
    nb_scenes_total_scenes_in_train_set = len(train_set)
    print("il y a {} scenes dans train_set".format(nb_scenes_total_scenes_in_train_set))

    #train_set_bis must have 125*2 scenes
    

    for i in range(len(train_set)):
        og_scene = train_set[i][0]
        noisy_scene = create_noisy_starting_trajectory(og_scene)
        
        plt_og = visualize_scene(og_scene)
        plt_og.show()


        

    

        
    print("apres l'ajout de scenes bruitées, train_set contient {} scenes".format(len(train_set)))
        

    
    #to konw position of ego agent, use path info to get the position of the ego agent
        

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



    #affiche les element de train_loader
    print("Before data augmentation, train_loader contains a total of {} batches".format(len(train_loader)))
    print(type(train_loader))


    

    

    for batch in train_loader:
        new_batch = apply_noise_on_starting_position(batch)
        
        
        plt_bef = visualize_batch(batch)
        plt_aft = visualize_batch(new_batch)

        plt_bef.show()
        plt_aft.show()
       

    

    

    #if the checkpoint path is not None, we validate the model
    if cfg.ckpt_path is not None:
        trainer.validate(model=model, dataloaders=val_loader, ckpt_path=cfg.ckpt_path)


    print(type(train_loader))
    print("After data augmentation, train_loader contains a total of {} batches".format(len(train_loader)))

    ######### LAUNCH TRAINING ########
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader,ckpt_path=cfg.ckpt_path)


    
if __name__ == '__main__':
    train()

