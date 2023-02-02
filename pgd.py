# Author:       Zihao Dong
# Description:  Modified PGD Attacks for
#      1. Maximizing Matching Loss

import torch
import torch.nn as nn

from attack import Attack
from utils import match_loss

class MatchingLossPGD(Attack):
    '''
    Linf PGD Attack Designed to create adversarial perturbations
    that Maximizes Matching Loss
    '''
    def __init__(self, model, args):
        super().__init__("MatchingLossPGD", model)
        self.args = args
        self.max_ch0 = 1
        self.min_ch0 = 0
        self.scale_ch0 = 1
        self.max_ch1 = 1
        self.min_ch1 = 0
        self.scale_ch1 = 1
        self.max_ch2 = 1
        self.min_ch2 = 0
        self.scale_ch2 = 1
        if args.dataset == "CIFAR10":
            self.max_ch0, self.min_ch0 = (1-0.4914)/0.2023, -0.4914/0.2023
            self.scale_ch0 = self.max_ch0 - self.min_ch0
            self.max_ch1, self.min_ch1 = (1-0.4822)/0.1994, -0.4822/0.1994
            self.scale_ch1 = self.max_ch1 - self.min_ch1
            self.max_ch2, self.min_ch2 = (1-0.4465)/0.2010, -0.4465/0.2010
            self.scale_ch2 = self.max_ch2 - self.min_ch2
        print(f"Diff Range: [{self.args.eps*self.scale_ch0, self.args.eps*self.scale_ch1, self.args.eps*self.scale_ch2}]")
            
    def forward(self, oriImages, oriLabels, synImages, synLabels):

        oriImages = oriImages.clone().detach().to(self.device)
        oriLabels = oriLabels.clone().detach().to(self.device)
        synImages = synImages.clone().detach().to(self.device)
        synLabels = synLabels.clone().detach().to(self.device)

        modelParam = list(self.model.parameters())

        loss = nn.CrossEntropyLoss().to(self.device)

        advSynImages = synImages.clone().detach()

        for _ in range(self.args.steps):

            oriImages.requires_grad = True
            advSynImages.requires_grad = True

            # Get Outputs
            oriOutputs = self.model(oriImages)
            advSynOutputs = self.model(advSynImages)

            # Calculate Matching Loss
            oriLoss = loss(oriOutputs, oriLabels)
            oriGrad = torch.autograd.grad(oriLoss, modelParam)
            oriGrad = list((_.detach().clone() for _ in oriGrad))
            synLoss = loss(advSynOutputs, synLabels)
            synGrad = torch.autograd.grad(synLoss, modelParam, create_graph=True)
            matchingLoss = match_loss(synGrad, oriGrad, self.args)

            # Compute Grad with Adv Images
            synUpdatedGrad = torch.autograd.grad(matchingLoss, advSynImages,
                retain_graph=False, create_graph=False, allow_unused=True)[0]
            
            # Update Images
            with torch.no_grad():
                advSynImages[:, 0] = advSynImages[:, 0].detach() + self.args.alpha*self.scale_ch0 * (synUpdatedGrad.sign())[:, 0]
                advSynImages[:, 1] = advSynImages[:, 1].detach() + self.args.alpha*self.scale_ch1 * (synUpdatedGrad.sign())[:, 1]
                advSynImages[:, 2] = advSynImages[:, 2].detach() + self.args.alpha*self.scale_ch2 * (synUpdatedGrad.sign())[:, 2]
                synDelta_ch0 = torch.clamp((advSynImages - synImages)[:, 0], min=-self.args.eps*self.scale_ch0, max=self.args.eps*self.scale_ch0)
                synDelta_ch1 = torch.clamp((advSynImages - synImages)[:, 1], min=-self.args.eps*self.scale_ch1, max=self.args.eps*self.scale_ch1)
                synDelta_ch2 = torch.clamp((advSynImages - synImages)[:, 2], min=-self.args.eps*self.scale_ch2, max=self.args.eps*self.scale_ch2)
                advSynImages[:, 0] = torch.clamp(synImages[:, 0] + synDelta_ch0, min=self.min_ch0, max=self.max_ch0).detach()
                advSynImages[:, 1] = torch.clamp(synImages[:, 1] + synDelta_ch1, min=self.min_ch1, max=self.max_ch1).detach()
                advSynImages[:, 2] = torch.clamp(synImages[:, 2] + synDelta_ch2, min=self.min_ch2, max=self.max_ch2).detach()

        return oriImages, advSynImages