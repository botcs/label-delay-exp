
import omegaconf
import torch
import torch.nn as nn
from copy import deepcopy
import torch
import torch.nn as nn
import torch.jit
import torch.nn.functional as F

from solo.methods.tent import TENT
import math
def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state
@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 1
    x = x/ temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x
def update_model_probs(current_model_probs, new_probs):
    if current_model_probs is None:
        if new_probs.size(0) == 0:
            return None
        else:
            with torch.no_grad():
                return new_probs.mean(0)
    else:
        if new_probs.size(0) == 0:
            with torch.no_grad():
                return current_model_probs
        else:
            with torch.no_grad():
                return 0.9 * current_model_probs + (1 - 0.9) * new_probs.mean(0)
class EATA(TENT):
    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)
        self.method = cfg.method

    
        e_margin=cfg.method_kwargs.e_margin_scale * math.log(cfg.data.num_classes)
        # d_margin=0.05
        d_margin=cfg.method_kwargs.d_margin
        self.num_samples_update_1 = 0  # number of samples after First filtering, exclude unreliable samples
        self.num_samples_update_2 = 0  # number of samples after Second filtering, exclude both unreliable and redundant samples
        self.e_margin = e_margin # hyper-parameter E_0 (Eqn. 3)
        self.d_margin = d_margin # hyper-parameter \epsilon for consine simlarity thresholding (Eqn. 5)

        self.current_model_probs = None # the moving average of probability vector (Eqn. 4)

        # self.fishers = self.compute_fishers(self.backbone, self.tta_parameters(), cfg, fisher_loader=None) # fisher regularizer items for anti-forgetting, need to be calculated pre model adaptation (Eqn. 9)
        # self.fisher_alpha = 2000.0 # trade-off \beta for two losses (Eqn. 8) 
        self.fisher_alpha = float(cfg.method_kwargs.fisher_alpha)

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        # self.model_state, self.optimizer_state = \
            # copy_model_and_optimizer(self.backbone, self.optimizer)
        
    def reset_model_probs(self, probs):
        self.current_model_probs = probs
        
    def tta_procedure(self,outs: dict, **kwargs):
        if self.method == 'eata':

            self.fishers = self.compute_fishers(self.backbone, self.tta_parameters,  fisher_loader=None)
        elif self.method == 'eat':
            self.fishers = None
        num_counts_2, num_counts_1, updated_probs = self.forward_and_adapt_eata(outs, self.fishers, self.e_margin, self.current_model_probs, fisher_alpha=self.fisher_alpha, num_samples_update=self.num_samples_update_2, d_margin=self.d_margin)
        self.num_samples_update_2 += num_counts_2
        self.num_samples_update_1 += num_counts_1
        self.reset_model_probs(updated_probs)


    
    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt_eata(self, outs, fishers, e_margin, current_model_probs, fisher_alpha=50.0, d_margin=0.05, scale_factor=2, num_samples_update=0):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        Return: 
        1. model outputs; 
        2. the number of reliable and non-redundant samples; 
        3. the number of reliable samples;
        4. the moving average  probability vector over all previous samples
        """

        # forward
        outputs = outs["logits"][0]
        # adapt
        entropys = softmax_entropy(outputs)
        
        # filter unreliable samples
        filter_ids_1 = torch.where(entropys < e_margin)
        ids1 = filter_ids_1
        if len(outputs[filter_ids_1]) == 0:
            outs['unsupervised_loss'] = 0 * entropys.mean(0)
            return 0,0,update_model_probs(current_model_probs, outputs[filter_ids_1].softmax(1))

        ids2 = torch.where(ids1[0]>-0.1)
        entropys = entropys[filter_ids_1] 
        # filter redundant samples
        # print (current_model_probs)
        if current_model_probs is not None: 
            cosine_similarities = F.cosine_similarity(current_model_probs.unsqueeze(dim=0), outputs[filter_ids_1].softmax(1), dim=1)
            # print(cosine_similarities, d_margin)
            filter_ids_2 = torch.where(torch.abs(cosine_similarities) < d_margin)
            entropys = entropys[filter_ids_2]
            ids2 = filter_ids_2
            updated_probs = update_model_probs(current_model_probs, outputs[filter_ids_1][filter_ids_2].softmax(1))
        else:
            updated_probs = update_model_probs(current_model_probs, outputs[filter_ids_1].softmax(1))
        coeff = 1 / (torch.exp(entropys.clone().detach() - e_margin))
        # implementation version 1, compute loss, all samples backward (some unselected are masked)
        entropys = entropys.mul(coeff) # reweight entropy losses for diff. samples
        loss = entropys.mean(0)
        """
        # implementation version 2, compute loss, forward all batch, forward and backward selected samples again.
        # if x[ids1][ids2].size(0) != 0:
        #     loss = softmax_entropy(model(x[ids1][ids2])).mul(coeff).mean(0) # reweight entropy losses for diff. samples
        """
        if fishers is not None:
            ewc_loss = 0
            for name, param in self.backbone.named_parameters():
                if name in fishers:
                    ewc_loss += fisher_alpha * (fishers[name][0] * (param - fishers[name][1])**2).sum()
            loss += ewc_loss
        if outputs[ids1][ids2].size(0) != 0:
            outs['unsupervised_loss'] = self.scale_loss * loss
        else:
            outs['unsupervised_loss'] = 0 * loss
        return  entropys.size(0), filter_ids_1[0].size(0), updated_probs


    def compute_fishers(self, model, params, fisher_loader=None):

        
        fishers = {}
        if fisher_loader is None:
            learnable_names = [dic['name']for dic in params]
            for name, param in model.named_parameters():
                    if param.grad is not None and name in learnable_names:
                        fisher = param.grad.data.clone().detach() ** 2 
                        fishers.update({name: [fisher, param.data.clone().detach()]})

        
        else:

            ewc_optimizer = torch.optim.SGD(params, 0.001)

            # fisher_loader = ImageNet_val_subset_data(data_dir=args.imagenet_path, 
                                                    # batch_size=args.batch_size, shuffle=args.shuffle, subset_size=args.fisher_size)
            train_loss_fn = nn.CrossEntropyLoss().cuda()
            
            # print("Computing fishers ...")

            for iter_, (images, targets) in enumerate(fisher_loader, start=1):      
                if self.device is not None:
                    images = images.cuda(self.device, non_blocking=True)
                if torch.cuda.is_available():
                    targets = targets.cuda(self.device, non_blocking=True)
                outputs = model(images)
                _, targets = outputs.max(1)
                loss = train_loss_fn(outputs, targets)
                loss.backward()
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if iter_ > 1:
                            fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                        else:
                            fisher = param.grad.data.clone().detach() ** 2
                        if iter_ == len(fisher_loader):
                            fisher = fisher / iter_
                        fishers.update({name: [fisher, param.data.clone().detach()]})
                ewc_optimizer.zero_grad()

            
            
            del ewc_optimizer
        # print("Computing fisher matrices finished")

        return fishers

