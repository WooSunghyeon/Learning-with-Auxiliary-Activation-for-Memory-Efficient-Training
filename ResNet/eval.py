import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import argparse
from resnet import ResNet, ResNet_ARA, ResNet_ARA_GCP, ResNet_ABA_ASA_GCP
from utils import progress_bar

#%%
parser = argparse.ArgumentParser(description='Auxiliary Activation Learning')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--batch-size', type=int, 
                    default=512, help='input batch size for training (default: 512)')
parser.add_argument('--test-batch-size', type=int, 
                    default=512, help='input batch size for teset (default: 512)')
parser.add_argument('--epochs', type=int, default=90,  
                    help='number of epochs to train (default: 90)')
parser.add_argument('--dataset', type= str ,choices = ['mnist', 'cifar10', 'cifar100', 'svhn', 'tiny-imagenet', 'imagenet'], 
                    default='cifar10', help='choose dataset (default: cifar10)')
parser.add_argument('--model', type= str ,choices = ['resnet18' ,'resnet34','resnet50', 'resnet101', 'resnet152', 'shnet'], 
                    default='resnet50', help='choose architecture (default: resnet50)')
parser.add_argument('--learning-rule', type=str, 
                    choices = ['bp', 'ara', 'aba', 'asa'],
                    default='bp',  help='learning rule to use (default: bp)')
parser.add_argument('--optimizer', type=str, choices = ['sgd', 'adam'], default = 'sgd'
                    , help = 'choose optimizer (default : sgd)')
parser.add_argument('--scheduler', type=str, 
                    choices = ['step', 'cos'],
                    default='step',  help='scheduler to use (default: step)')
parser.add_argument('--device', nargs='+', type= int,  help='device_num')
parser.add_argument('--ARA-stride', type = int, nargs='+', help='The number of stride for ARA')
parser.add_argument('--checkpoint', action = 'store_true', default = False,
                    help = 'use checkpoint')
parser.add_argument('--gcp', action = 'store_true', default = False,
                    help = 'gradient checkpointing')
parser.add_argument('--get-li', action = 'store_true', default = False,
                    help = 'calculate learning_indicator')
parser.add_argument('--model_path', type=str, help='The path to the saved model file')

def main():
    args = parser.parse_args()
    device = args.device
    if type(device) != int:
        device = args.device[0]
    
    # Data
    # All data licensed under CC-BY-SA.
    dataset = args.dataset
    if dataset == 'mnist':
        testloader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),batch_size=args.test_batch_size, shuffle=False)
    
    elif dataset == 'svhn':
        transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        testloader = torch.utils.data.DataLoader(torchvision.datasets.SVHN('./data', split = 'test',download=True, transform=transform),
                                                  batch_size=args.test_batch_size,
                                                  shuffle=False)
        
    elif dataset == 'cifar10':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)
                
    elif dataset == 'cifar100':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.test_batch_size, shuffle=False, num_workers=8)
        
    elif dataset == 'tiny-imagenet':
        tiny_mean = [0.48024578664982126, 0.44807218089384643, 0.3975477478649648]
        tiny_std = [0.2769864069088257, 0.26906448510256, 0.282081906210584]
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(tiny_mean, tiny_std)])
        num_workers = 16
        testset = torchvision.datasets.ImageFolder('/nfs/home/wshey/tiny-imagenet/val', transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False,
                                                 num_workers=num_workers)
        
    elif dataset == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform_test = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        num_workers = 4 * len(args.device)
        testset = torchvision.datasets.ImageFolder('/scratchpad/datasets/ImageNet/val', transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False,
                                                 num_workers=num_workers)
        
    # Set the model               
    net = ResNet(args.dataset, args.model)
    save_file_name = args.dataset + '_' + args.model + '_' + args.learning_rule
    if args.learning_rule == 'ara':
        net = ResNet_ARA(args.dataset, args.model, args.ARA_stride)
        save_file_name += '_' + str(args.ARA_stride)
        if args.gcp:
            net = ResNet_ARA_GCP(args.dataset, args.model, args.ARA_stride, args.get_li)
            save_file_name += '_GCP' 
    elif args.learning_rule == 'aba' or args.learning_rule == 'asa':
        net = ResNet_ABA_ASA_GCP(args.dataset, args.model, args.learning_rule, args.ARA_stride, args.get_li)
        save_file_name += '_' + str(args.ARA_stride)
    
    if type(args.device) != int:
        net = nn.DataParallel(net, device_ids=args.device) 
        net = net.to(device)
    else:
        net = net.to(device)
    
    PATH = args.model_path
    checkpoint = torch.load(PATH)
    state_dict = checkpoint['net']
    #net.load_state_dict({k.replace('module.',''):v for k,v in state_dict.items()}, strict = True)
    net.load_state_dict(state_dict, strict=True) 
    
    
    # Evaluation
    criterion = nn.CrossEntropyLoss()
    def test():
        net.eval()
        test_loss = correct = total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs, li = net(inputs)
                loss = criterion(outputs, targets) 
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        return 100.*correct/total, test_loss
        
    test_acc, test_loss = test()
    print('test accuracy : ' + str(round(test_acc, 2)))
if __name__ == '__main__':
    main()      
