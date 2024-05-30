import numpy as np
import torch
import wandb

import torch.nn as nn
import torch.nn.functional as F
from utils.visualization import visualize_prediction
from motionnet.models.mmTransformer.TF_version.stacked_transformer import STF
from motionnet.models.base_model.base_model import BaseModel
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, ReduceLROnPlateau
from torch import optim
from torch.distributions import MultivariateNormal, Laplace
from scipy import special


import pytorch_lightning as pl




class AutomaticWeightedLoss(torch.nn.Module):
    """automatically weighted multi-task loss
    log(sigma)=a(a是一个可学习的变量)
    torch.exp(-a)=torch.exp(-log(sigma))=torch.exp(log(sigma**-1))=1/sigma
    Params:
        num: int, the number of loss
        x: multi-task loss
    Examples:
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=3):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True) 
        self.params = torch.nn.Parameter(params) 

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += (loss / (self.params[i] ** 2) + torch.log(1 + self.params[i]))
        return loss_sum



class LaneNet(nn.Module):
    def __init__(self, in_channels, hidden_unit, num_subgraph_layers):
        super(LaneNet, self).__init__()
        self.num_subgraph_layers = num_subgraph_layers
        self.layer_seq = nn.Sequential()
        for i in range(num_subgraph_layers):
            self.layer_seq.add_module(
                f'lmlp_{i}', MLP(in_channels, hidden_unit))
            in_channels = hidden_unit*2

    def forward(self, lane):
        '''
            Extract lane_feature from vectorized lane representation

        Args:
            lane: [batch size, max_lane_num, 9, 7] (vectorized representation)

        Returns:
            x_max: [batch size, max_lane_num, 64]
        '''
        x = lane
        for name, layer in self.layer_seq.named_modules():
            if isinstance(layer, MLP):
                # x [bs,max_lane_num,9,dim]
                x = layer(x)
                x_max = torch.max(x, -2)[0]
                x_max = x_max.unsqueeze(2).repeat(1, 1, x.shape[2], 1)
                x = torch.cat([x, x_max], dim=-1)
        x_max = torch.max(x, -2)[0]
        return x_max


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_unit, verbose=False):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_unit),
            nn.LayerNorm(hidden_unit),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


#class mmTrans(nn.Module):
#class mmTrans(pl.LightningModule):
class mmTrans(BaseModel):

    def __init__(self, config):#    def __init__(self, stacked_transformer, config):
        super(mmTrans, self).__init__(config)
        #initialize stacked_transformer
        import importlib
        module_path = f'motionnet.models.mmTransformer.TF_version.stacked_transformer'
        module_name = 'STF'
        target_module = importlib.import_module(module_path)
        stacked_transformer = getattr(target_module, module_name)
        #stacked_transformer = STF

        # stacked transformer class
        self.stacked_transformer = stacked_transformer(config)

        lane_channels = config['lane_channels']
        self.hist_feature_size = config['in_channels']

        self.polyline_vec_shape = 2*config['subgraph_width']
        self.subgraph = LaneNet(
            lane_channels, config['subgraph_width'], config['num_subgraph_layres'])

        self.FUTURE_LEN = config['future_num_frames']
        self.OBS_LEN = config['history_num_frames'] - 1
        self.lane_length = config['lane_length']

        self.criterion = Criterion(self.config)

        self.fisher_information = None
        self.optimal_params = None

    def preprocess_traj(self, traj):
        '''
            Generate the trajectory mask for all agents (including target agent)

            Args:
                traj: [batch, max_agent_num, obs_len, 4]

            Returns:
                social mask: [batch, 1, max_agent_num]

        '''
        # social mask
        social_valid_len = self.traj_valid_len
        social_mask = torch.zeros(
            (self.B, 1, int(self.max_agent_num))).to(traj.device)
        for i in range(self.B):
            social_mask[i, 0, :social_valid_len[i]] = 1

        return social_mask

    def preprocess_lane(self, lane):
        '''
            preprocess lane segments using LaneNet

        Args:
            lane: [batch size, max_lane_num, 10, 5]

        Returns:
            lane_feature: [batch size, max_lane_num, 64 (feature_dim)]
            lane_mask: [batch size, 1, max_lane_num]

        '''

        # transform lane to vector
        lane_v = torch.cat(
            [lane[:, :, :-1, :2],
             lane[:, :, 1:, :2],
             lane[:, :, 1:, 2:]], dim=-1)  # bxnlinex9x7

        # lane mask
        lane_valid_len = self.lane_valid_len
        lane_mask = torch.zeros(
            (self.B, 1, int(self.max_lane_num))).to(lane_v.device)
        for i in range(lane_valid_len.shape[0]):
            lane_mask[i, 0, :lane_valid_len[i]] = 1

        # use vector like structure process lane
        lane_feature = self.subgraph(lane_v)  # [batch size, max_lane_num, 64]

        return lane_feature, lane_mask

    def forward(self, data: dict, batch_idx):
        """
        Args:
            data (Data): 
                HIST: [batch size, max_agent_num, 19, 4]
                POS: [batch size, max_agent_num, 2]
                LANE: [batch size, max_lane_num, 10, 5]
                VALID_LEN: [batch size, 2] (number of valid agents & valid lanes)

        Note:
            max_lane_num/max_agent_num indicates maximum number of agents/lanes after padding in a single batch 
        """

        #Preprocess the data#################################################################################################
    
        inputs = data['input_dict']
        agents_in, agents_mask, roads = inputs['obj_trajs'],inputs['obj_trajs_mask'] ,inputs['map_polylines']

        self.B = inputs['obj_trajs'].size(0)

        self.max_agent_num = inputs['obj_trajs'].size(1)
        self.traj_valid_len = torch.ones((self.B),dtype=torch.int, device=agents_in.device)*(inputs['obj_trajs'].size(1))

        self.max_lane_num = inputs['map_polylines'].size(1)
        self.lane_valid_len =  torch.ones((self.B), dtype=torch.int, device=agents_in.device)*(inputs['map_polylines'].size(1)) #C'est le nombre de valid_lane par BATCH en fait

        #Define pos
        norm_center = torch.zeros((self.B,1,2), device = agents_in.device)
        for i in range(self.B):
            norm_center[i,0,:] = inputs['obj_trajs_last_pos'][i,inputs['track_index_to_predict'][i],:2]
        pos = inputs['obj_trajs_last_pos'][:,:,:2]-norm_center.expand(self.B,inputs['obj_trajs_last_pos'].size(1),2)

        #Define trajs
        vector = torch.arange(agents_in.size(2), dtype=torch.float, device=agents_in.device)
        history = torch.cat([agents_in[...,:2], vector.unsqueeze(0).unsqueeze(0).expand(agents_in.size(0), agents_in.size(1), agents_in.size(2)).unsqueeze(-1),agents_mask.unsqueeze(-1)],dim=-1)
        history = history[:,:,2:,:]
        trajs = history

        lane = torch.cat([inputs['map_polylines'][:,:,:,:4],inputs['map_polylines_mask'][:,:,:].unsqueeze(-1)], dim=3)
        social_mask = self.preprocess_traj(history)
        lane_enc, lane_mask = self.preprocess_lane(lane)

        out = self.stacked_transformer(trajs, pos, self.max_agent_num,social_mask, lane_enc, lane_mask)

        #Puts prediction for the right dimension for get_loss
        prediction = {}
        predicted_traj = torch.cat([out[0], torch.ones((out[0].size(0),out[0].size(1),out[0].size(2),out[0].size(3),3), device = agents_in.device)*0.1],dim=-1)
        prediction['predicted_trajectory'] = torch.zeros((self.B,out[0].size(2),out[0].size(3),5),device=agents_in.device)
        prediction['predicted_probability'] = torch.zeros((self.B,out[0].size(2)),device=agents_in.device)
        for i in range(self.B):
            prediction['predicted_trajectory'][i,:,:,:] = predicted_traj[i,inputs['track_index_to_predict'][i],:,:,:].squeeze(0)#predicted_traj
            prediction['predicted_probability'][i,:] = out[1][i,inputs['track_index_to_predict'][i],:].squeeze(0)

        #####Wandb ############################
        plt = visualize_prediction(data, prediction)
        self.logger.experiment.log({'example': wandb.Image(plt)})
        ######fermer la figure pour éviter d'avoir des figures qui s'accumulent
        plt.close()
        ######################################
        loss = self.get_loss(data, prediction) #Old loss function

        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)


        """
        LOSS FROM PAPER !!
        #on recupere la prediction [batch]
        #print("predicted_trajectory",prediction['predicted_trajectory'].size()) ==> 32, 6, 60, 5 soit 32 batch, 6 prédiction, 60 agents, 5 features

        #on recupere seulement la trajectoire [x y] predite
        pred_traj = prediction['predicted_trajectory'][:,:,:,:2]
    
        #pour avoir fde, on fait la soustraction entre la prediction et le ground truth. Attention aux dimensions
        #on doit avoir fde de taille [batch, 6]

        gt_traj = inputs['center_gt_trajs'][...,:2]

        #pour chaque batch, pour chacune des 6 prédiction, on calcule la distance euclidienne entre la prédiction et le ground truth

        fde = torch.zeros((self.B,6),device=agents_in.device)
        
        for i in range(self.B):
            for j in range(6):
                fde[i,j] = torch.norm(pred_traj[i,j,:,:]-gt_traj[i,:,:],dim=1).sum()

        min_fde_index = torch.argmin(fde,dim=1)

        min_fde_trj = torch.tensor([]).to(agents_in.device)
        C = torch.tensor([]).to(agents_in.device)
        #boucle qui permet de recuperer les trajectoires avec le minimum final displacement error
        for i in range(self.B):
            tmp_trj = pred_traj[i,min_fde_index[i],:,:].unsqueeze(0)
            min_fde_trj = torch.cat([min_fde_trj,tmp_trj],dim=0)
       
            tmp_score = prediction['predicted_probability'][i,min_fde_index[i]].unsqueeze(0)
            C = torch.cat([C,tmp_score],dim=0)
       
       
       
        #recover_pred_trj is the predicted future trajectory of the target vehicle
        index_target = inputs['track_index_to_predict']
        recover_pred_trj = prediction['predicted_trajectory'][:,index_target,:,:2]

        #recover_gt_trj is the real future trajectory of the target vehicle
        recover_gt_trj = inputs['center_gt_trajs'][...,:2]

        #Computes loss function of article, comment this out if want to use get_loss of ptr
        loss = self.calculate_loss(recover_pred_trj, recover_gt_trj, min_fde_trj, gt_traj, C)
        """
        return prediction, loss
    



    def calculate_loss(self,recover_trj, recover_gt_trj, min_fde_trj, gt_trj, C):
        '''
        vanilla training strategy: only using the proposal with the minimum final displacement error
        vanilla training strategy dont need confidence score

        recover_trj:  il s'agit de la prédiction de la trajectoire future du véhicule cible
        recover_gt_trj: il s'agit de la trajectoire future réelle du véhicule cible
        min_fde_trj: il s'agit de la prédiction de la trajectoire future du véhicule cible avec le minimum final displacement error parmi les 6 prédiction
        gt_trj: il s'agit de la trajectoire future réelle du véhicule cible avec le minimum final displacement error parmi les 6 prédiction
        C: il s'agit du score de confiance de la prédiction de la trajectoire future du véhicule cible avec le minimum final displacement error parmi les 6 prédiction
        '''
        # regression loss

        awl = AutomaticWeightedLoss(3)
        device = recover_trj.device
        crit = torch.nn.HuberLoss().to(device)
        regression_loss = crit(min_fde_trj, gt_trj)
        recover_regression_loss = crit(recover_trj, recover_gt_trj)

        P = C 
        classification_loss = -1 * torch.log(P)
        classification_loss = classification_loss.mean() #

        loss_sum = awl(recover_regression_loss, regression_loss, classification_loss)
        return loss_sum


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr= self.config['learning_rate'],eps=0.001,weight_decay=1e-5)
        lr= self.config['learning_rate']
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config['max_epochs'], eta_min=lr * 0.015,verbose=True)

        return [optimizer], [scheduler]
    
    def get_loss(self, batch, prediction):
        """
        batch : dict qui contient les données d'entrée du batch. Ses clés sont : ['batch_size', 'input_dict', 'batch_sample_count']
        prediction : dict qui contient les prédictions du modèle. Ses clés sont : ['predicted_probability', 'predicted_trajectory']
        """

        inputs = batch['input_dict']
        ground_truth = torch.cat([inputs['center_gt_trajs'][...,:2],inputs['center_gt_trajs_mask'].unsqueeze(-1)],dim=-1)[:,:prediction['predicted_trajectory'].size(2),:]
        loss = self.criterion(prediction, ground_truth,inputs['center_gt_final_valid_idx'], inputs['trajectory_type'])

        return loss


class Criterion(nn.Module):
    def __init__(self, config):
        super(Criterion, self).__init__()
        self.config = config


    def forward(self, out, gt,center_gt_final_valid_idx,type_trajectory_GT):

        return self.nll_loss_multimodes(out, gt,center_gt_final_valid_idx, type_trajectory_GT)

    def get_BVG_distributions(self, pred):
        B = pred.size(0)
        T = pred.size(1)
        mu_x = pred[:, :, 0].unsqueeze(2)
        mu_y = pred[:, :, 1].unsqueeze(2)
        sigma_x = pred[:, :, 2]
        sigma_y = pred[:, :, 3]
        rho = pred[:, :, 4]

        cov = torch.zeros((B, T, 2, 2)).to(pred.device)
        cov[:, :, 0, 0] = sigma_x ** 2
        cov[:, :, 1, 1] = sigma_y ** 2
        cov[:, :, 0, 1] = rho * sigma_x * sigma_y
        cov[:, :, 1, 0] = rho * sigma_x * sigma_y

        biv_gauss_dist = MultivariateNormal(loc=torch.cat((mu_x, mu_y), dim=-1), covariance_matrix=cov)
        return biv_gauss_dist

    def get_Laplace_dist(self, pred):
        return Laplace(pred[:, :, :2], pred[:, :, 2:4])

    def nll_pytorch_dist(self, pred, data, mask, rtn_loss=True):
        biv_gauss_dist = self.get_Laplace_dist(pred)
        data_reshaped = data[:, :, :2]
        if rtn_loss:
            return ((-biv_gauss_dist.log_prob(data_reshaped)).sum(-1) * mask).sum(1)  # Laplace
        else:
            return ((-biv_gauss_dist.log_prob(data_reshaped)).sum(dim=2) * mask).sum(1)  # Laplace

    def nll_loss_multimodes(self, output, data, center_gt_final_valid_idx,type_trajectory_GT):
        """NLL loss multimodes for training. MFP Loss function
        Args:
          pred: [K, T, B, 5]
          data: [B, T, 5]
          modes_pred: [B, K], prior prob over modes
          noise is optional
        """
        modes_pred = output['predicted_probability']
        pred = output['predicted_trajectory'].permute(1, 2, 0, 3)
        mask = data[..., -1]

        entropy_weight = self.config['entropy_weight']
        kl_weight = self.config['kl_weight']
        use_FDEADE_aux_loss = self.config['use_FDEADE_aux_loss']

        modes = len(pred)
        nSteps, batch_sz, dim = pred[0].shape

        # compute posterior probability based on predicted prior and likelihood of predicted trajectory.
        log_lik = np.zeros((batch_sz, modes))
        with torch.no_grad():
            for kk in range(modes):
                nll = self.nll_pytorch_dist(pred[kk].transpose(0, 1), data, mask, rtn_loss=False)
                log_lik[:, kk] = -nll.cpu().numpy()

        priors = modes_pred.detach().cpu().numpy()
        log_posterior_unnorm = log_lik + np.log(priors)
        log_posterior = log_posterior_unnorm - special.logsumexp(log_posterior_unnorm, axis=-1).reshape((batch_sz, -1))
        post_pr = np.exp(log_posterior)
        post_pr = torch.tensor(post_pr).float().to(data.device)


        w = [1.976, 1.496 , 1.936, 1.976 , 2. , 1.84  ,1.992 ,1.784]

        weights = torch.zeros_like(type_trajectory_GT)
        weights = weights.float()


        for i in range(type_trajectory_GT.shape[0]):
            type_traj = type_trajectory_GT[i] #tensor(1, device='cuda:0')
            #le convertir en int
            index = int(type_traj)
            weights[i] = w[index] #pour chacune des 64 trajectory, on regarde la valeur            
        weights = weights.to(data.device)

        # Compute loss.
        loss = 0.0
        for kk in range(modes):
            nll_k = self.nll_pytorch_dist(pred[kk].transpose(0, 1), data, mask, rtn_loss=True) * post_pr[:, kk]
            loss += nll_k.mean()

        # Adding entropy loss term to ensure that individual predictions do not try to cover multiple modes.
        entropy_vals = []
        for kk in range(modes):
            entropy_vals.append(self.get_BVG_distributions(pred[kk]).entropy())
        entropy_vals = torch.stack(entropy_vals).permute(2, 0, 1)
        entropy_loss = torch.mean((entropy_vals).sum(2).max(1)[0])
        loss += entropy_weight * entropy_loss

        # KL divergence between the prior and the posterior distributions.
        kl_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')  # type: ignore
        kl_loss = kl_weight * kl_loss_fn(torch.log(modes_pred), post_pr)

        # compute ADE/FDE loss - L2 norms with between best predictions and GT.
        if use_FDEADE_aux_loss:
            adefde_loss = self.l2_loss_fde(pred, data, mask)
        else:
            adefde_loss = torch.tensor(0.0).to(data.device)

        # post_entropy
        final_loss = loss + kl_loss + adefde_loss

        return final_loss

    def l2_loss_fde(self, pred, data, mask):

        fde_loss = (torch.norm((pred[:, -1, :, :2].transpose(0, 1) - data[:, -1, :2].unsqueeze(1)), 2, dim=-1) * mask[:,
                                                                                                                 -1:])
        ade_loss = (torch.norm((pred[:, :, :, :2].transpose(1, 2) - data[:, :, :2].unsqueeze(0)), 2,
                               dim=-1) * mask.unsqueeze(0)).mean(dim=2).transpose(0, 1)
        loss, min_inds = (fde_loss + ade_loss).min(dim=1)
        return 100.0 * loss.mean()