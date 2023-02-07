import argparse
import numpy as np
import time

import torch
import torch.nn as nn
from torchvision.utils import save_image
from utils import get_dataset, get_network

from tqdm import tqdm

from pgd import PGD

classes = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def main():

    # Test model accuracy under PGD attack
    
    parser = argparse.ArgumentParser(description="parameter for attack on models")
    parser.add_argument('--steps', type=int, default=10, help='number of steps for PGD attack')
    parser.add_argument('--eps', type=float, default=8, help='perturbation bound for PGD attach, default 8(/255)')
    parser.add_argument('--alpha', type=float, default=2, help='learning rate or Alpha for PGD attack, default 2(/255)')
    parser.add_argument('--scale', type=float, default=1, help='scale in pgd attack')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--checkpoint', type=str, help="Path to model checkpoint")
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--batch_size', type=int, default=1, help="batch size in pgd attack")
    parser.add_argument('--output_path', type=str, default='result/modelRobustness.txt', help="File to output model robustness test result")

    args = parser.parse_args()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.eps /= 255
    args.alpha /= 255

    # Initialize DataLoader
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)
    dataLoader = torch.utils.data.DataLoader(dst_test, batch_size=args.batch_size, shuffle=True)

    # Initialize Network from Checkpoint File
    model = get_network(args.model, channel, num_classes, im_size).to(args.device) # get a random model
    model.load_state_dict(torch.load(args.checkpoint))

    # Initialize Adversarial Agent
    agent = PGD(model, eps=args.eps, alpha=args.alpha, steps=args.steps, scale=args.scale)

    # Initialize Counters
    numExamples = 0
    numSuccessBeforeAttack = 0
    numSuccessAfterAttack = 0

    # Attack main loop
    for data, label in tqdm(dataLoader):
        data = data.to(args.device)

        # Get Original Output
        originalOutput = model(data)
        originalPrediction = torch.argmax(originalOutput, axis=-1)

        numSuccessBeforeAttack += np.sum(np.equal(originalPrediction, label.data.numpy()))
        numExamples += label.shape[0]

        # Generate Adversarial Examples
        adversarialData = agent(data, label)

        # Get Adversarial Output
        adversarialOutput = model(adversarialData)
        adversarialPrediction = torch.argmax(adversarialOutput, axis=-1)
        numSuccessAfterAttack += np.sum(np.equal(adversarialPrediction, label.data.numpy()))
    
    with open(args.output_path, 'a') as f:
        f.write('====================%s======================\n' % (time.asctime(time.localtime(time.time()))))
        f.write('Experiment with %s checkpoint %s on %s:\n' % (args.model, args.checkpoint, args.dataset))
        f.write('PGD: batch size %d, eps = %d/255, alpha = %d/255, scale = %d/255, with %d steps\n'
            % (args.batch_size, args.eps, args.alpha, args.scale, args.steps))
        f.write('\tTotal Number of Examples Tested: %d\n' % (numExamples))
        f.write('\tSuccess Rate Before Attack: %.2f\n' % (numSuccessBeforeAttack/numExamples))
        f.write('\tSuccess Rate After Attack: %.2f\n' % (numSuccessAfterAttack/numExamples))