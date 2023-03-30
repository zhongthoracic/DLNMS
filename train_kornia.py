import datetime
import os
import time
from torch.autograd import Variable
import torch
from torch.optim import lr_scheduler
import torch.utils.data
from torch import nn
import torchvision
import datasets
import presets
import utils
import torch.nn as nn
import numpy as np
import Focalloss
import torch.optim
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import datasets_auc as datasets_kornia
import random
from libauc.losses import AUCMLoss
from libauc.optimizers import PESG
try:
    from apex import amp
except ImportError:
    amp = None

def calcauc(labels,probs):
    N = 0
    P = 0
    neg_prob = []
    pos_prob = []
    for _,i in enumerate(labels):
        if(i==1):
            P +=1
            pos_prob.append(probs[_])
        else:
            N+=1
            neg_prob.append(probs[_])
    number = 0
    for pos in pos_prob:
        for neg in neg_prob:
            if(pos>neg):
                number+=1
            elif(pos==neg):
                number +=0.5
    return number/(N*P)

def mixup_data(x1, y, alpha, device, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x1.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).to(device)
    else:
        index = torch.randperm(batch_size)

    mixed_x1 = lam * x1 + (1 - lam) * x1[index, :]
    #mixed_x2 = lam * x2 + (1 - lam) * x2[index, :] 
    y_a, y_b = y, y[index]
    return mixed_x1, y_a, y_b, lam

def mixup_data2(x1, x2, y, alpha, device, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x1.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).to(device)
    else:
        index = torch.randperm(batch_size)

    mixed_x1 = lam * x1 + (1 - lam) * x1[index, :]
    mixed_x2 = lam * x2 + (1 - lam) * x2[index, :] 
    y_a, y_b = y, y[index]
    return mixed_x1, mixed_x2, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)



def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq, writer, apex=False, mixup = False, train_mode = 'pet-ct'):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value}'))
    i=0
    header = 'Epoch: [{}]'.format(epoch)
    for index, image, target in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        if mixup:
            if 'pet-ct' in train_mode:
                imaget = image
                image, image2 = imaget[:,0],imaget[:,1]
                image, image2, target,target2, lam = mixup_data2(image, image2,target, 0.2, device)
                if train_mode =='pet-ct':
                    output, output_aux1, output_aux2 = model(image, image2)
                    loss_main = mixup_criterion(criterion,output,target,target2,lam)
                    loss_aux1 = mixup_criterion(criterion,output_aux1,target,target2,lam)
                    loss_aux2 = mixup_criterion(criterion,output_aux2,target,target2,lam)
                    loss = loss_main + (loss_aux1+loss_aux2)*0.1
                else:
                    output = model(image,image2)
                    loss = mixup_criterion(criterion,output,target,target2,lam)
            else:
                image, targets_a, targets_b, lam = mixup_data(image, target,
                                                       0.2, device)
                output = model(image)
                loss = mixup_criterion(criterion, output, targets_a, targets_b, lam)
        else:
            if 'pet-ct' in train_mode:
                imaget = image
                image, image2 = imaget[:,0],imaget[:,1]
                if train_mode == 'pet-ct':
                    output, output_aux1, output_aux2 = model(image, image2)
                    if args.opt == 'libauc':
                        output = torch.sigmoid(output)
                        output = torch.sigmoid(output_aux1)
                        output = torch.sigmoid(output_aux2)
                    loss_main = criterion(output, target)
                    loss_aux1 = criterion(output_aux1, target)
                    loss_aux2 = criterion(output_aux2, target)
                    loss = loss_main + (loss_aux1+loss_aux2)*0.1
                else:
                    output = model(image,image2)
                    if args.opt == 'libauc':
                        output = torch.sigmoid(output)
                    loss = criterion(output, target)
            else:
                output = model(image)
                loss = criterion(output, target)
        torch.nn.utils.clip_grad_value_(model.parameters(),35)
        optimizer.zero_grad()
        if apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 2))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))
        writer.add_scalar("train_loss",loss,epoch*len(data_loader)+i)
        i+=1

def evaluate(model, criterion, data_loader, device,writer, print_freq=100, epoch=0, train_mode = 'pet-ct',eval_mode='val'):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    outx = []
    with torch.no_grad():
        for index, image, target in metric_logger.log_every(data_loader, print_freq, header):
            t = target.numpy()
            image = image.to(device,non_blocking=True)
            target = target.to(device,non_blocking=True)
            if 'pet-ct' in train_mode:
                imaget = image
                image, image2 = imaget[:,0],imaget[:,1]
                output,output_aux1,output_aux2 = model(image,image2)
                outputx = torch.nn.functional.softmax(output,dim=1)
                outputx1 = torch.softmax(output_aux1,dim=1)
                outputx2 = torch.softmax(output_aux2,dim=1)
                outputx = outputx.detach().cpu().numpy().ravel()
                outputx1 = outputx1.detach().cpu().numpy().ravel()
                outputx2 = outputx2.detach().cpu().numpy().ravel()
                #outputx = (0.8*outputx + 0.1*outputx1 + 0.1*outputx2)
                loss = criterion(output, target)
            else:
                output = model(image)
                loss = criterion(output, target)
                outputx = torch.nn.functional.softmax(output,dim=1)
                outputx = outputx.detach().cpu().numpy().ravel()
            outx.append(np.concatenate((outputx,t),0))
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 2))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    outx = np.array(outx)
    #np.savetxt('pt1.txt',outx,'%.2f')
    
    labels = outx[:,2]
    probs = outx[:,1]
    #auc = calcauc(labels,probs)
    
    writer.add_pr_curve(eval_mode+"_pr",labels,probs,epoch)
    fpr, tpr, thresholds = metrics.roc_curve(labels, probs, pos_label=1)
    f = plt.figure()
    plt.title('roc')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.plot(fpr,tpr)
    writer.add_figure(eval_mode+'_roc',f,epoch)
    auc = metrics.auc(fpr, tpr)
    print(auc)
    writer.add_scalar(eval_mode+"_auc",auc,epoch)
    print(' * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5))
    return auc


def _get_cache_path(filepath):
    import hashlib
    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def load_data(traindir, valdir, args,):
    # Data loading code
    print("Loading data")
    resize_size, crop_size = (342, 299) if args.model == 'inception_v3' else (224, 112)

    print("Loading training data")
    st = time.time()
    cache_path = _get_cache_path(traindir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print("Loading dataset_train from {}".format(cache_path))
        dataset, _ = torch.load(cache_path)
    else:
        auto_augment_policy = getattr(args, "auto_augment", None)
        random_erase_prob = getattr(args, "random_erase", 0.0)

        df = pd.read_csv('/home/gu1h/178data/pet-ct/pet-ct/internalN2.csv')
        #pet_ct_id = np.array(df_pet_ct['ID'])
        #N1orN2 = np.array(df_pet_ct[args.label])

        T = np.array(df[df[args.label]==1].index)
        F = np.array(df[df[args.label]==0].index)
        T_label = np.ones_like(T)
        F_label = np.zeros_like(F)
        kf_T = list(StratifiedKFold(n_splits=args.split_nums,shuffle=True,random_state=10).split(T,T_label))
        kf_F = list(StratifiedKFold(n_splits=args.split_nums,shuffle=True,random_state=10).split(F,F_label))
        Ttrain_index = kf_T[args.dataset_num][0]
        Tval_index = kf_T[args.dataset_num][1]
        Ftrain_index = kf_F[args.dataset_num][0]
        Fval_index = kf_F[args.dataset_num][1]
        tt = T[Ttrain_index]
        tf = F[Ftrain_index]
        vt = T[Tval_index]
        vf = F[Fval_index]
        w = 12
        if(args.label == "N1"):
            w = 10
        t_npy = np.concatenate(w*[tt]+[tf])
        np.random.shuffle(t_npy)
        train_df = df.iloc[t_npy].reset_index(drop=True)
        val_df = df.iloc[np.concatenate([vt]+[vf])].reset_index(drop=True)
        

        dataset = datasets_kornia.PetCT(train_df,'train',args.label,presets.ClassificationPresetTrain(
                crop_size= crop_size,
                resize_size=resize_size,
                auto_augment_policy=auto_augment_policy,
                random_erase_prob=random_erase_prob,
            ),args.mx, args.device)
        if args.cache_dataset:
            print("Saving dataset_train to {}".format(cache_path))
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset, traindir), cache_path)
    print("Took", time.time() - st)

    print("Loading validation data")
    cache_path = _get_cache_path(valdir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print("Loading dataset_test from {}".format(cache_path))
        dataset_test, _ = torch.load(cache_path)
    else:
        dataset_test = datasets_kornia.PetCT(val_df,'val',args.label,presets.ClassificationPresetEval(
                crop_size= crop_size,
                resize_size=resize_size,
            ),args.mx,args.device)
        
        if args.cache_dataset:
            print("Saving dataset_test to {}".format(cache_path))
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_test, valdir), cache_path)
    test_df = pd.read_csv('/home/gu1h/178data/pet-ct/pet-ct/externalN2.csv')
    dataset_test_final = datasets_kornia.PetCT(test_df,'test',args.label,presets.ClassificationPresetEval(
                crop_size= crop_size,
                resize_size=resize_size,
            ),args.mx,args.device)
    #,mean_pet=(0.90450126,0.90450126,0.90450126),std_pet=(1.0321479,1.0321479,1.0321479),mean_ct=(-421.07526,-421.07526,-421.07526),std_ct=(460.58328,460.58328,460.58328)
    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test,shuffle = False)
        test_sampler_final = torch.utils.data.distributed.DistributedSampler(dataset_test_final,shuffle = False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)
        test_sampler_final = torch.utils.data.SequentialSampler(dataset_test_final)

    return dataset, dataset_test,dataset_test_final, train_sampler, test_sampler,test_sampler_final

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
def main(args):
    if args.apex and amp is None:
        raise RuntimeError("Failed to import apex. Please install apex from https://www.github.com/nvidia/apex "
                           "to enable mixed-precision training.")
    args.output_dir = os.path.join("workspace",args.output_dir,args.label,args.mx,str(args.dataset_num))
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)
    #setup_seed(10)
    device = torch.device(args.device)

    torch.backends.cudnn.benchmark = True

    train_dir = os.path.join(args.data_path, "train_1")
    val_dir = os.path.join(args.data_path, "test_1")
    dataset, dataset_test,dataset_test_final, train_sampler, test_sampler,test_sampler_final = load_data(train_dir, val_dir, args)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers)
    
    data_loader_test_final = torch.utils.data.DataLoader(
        dataset_test_final, batch_size=1,
        sampler=test_sampler_final, num_workers=args.workers)
    class create_head(nn.Module):
        def __init__(self, in_features=2048,out_features=2):
            super().__init__()
            self.ap = nn.AdaptiveAvgPool2d(1)
            self.mp = nn.AdaptiveMaxPool2d(1)
            self.flat = nn.Flatten()
            self.bn1 = nn.BatchNorm1d(in_features*2)
            self.dp = nn.Dropout(0.25)
            self.fc1 = nn.Linear(in_features*2,512)
            self.relu = nn.ReLU(True)
            self.bn2 = nn.BatchNorm1d(512)
            self.dp2 = nn.Dropout(0.25)
            self.fc2 = nn.Linear(512,out_features)
            self.sigmoid = nn.Sigmoid()
        def forward(self,x):
            x = torch.cat([self.mp(x),self.ap(x)],1)
            x = self.flat(x)
            x = self.bn1(x)
            if self.training:
                x = self.dp(x)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.bn2(x)
            if self.training:
                x = self.dp2(x)
            x = self.fc2(x)
            return x


    class NxNet(nn.Module):
        def __init__(self):
            super().__init__()
            model = torchvision.models.__dict__[args.model](pretrained=True)
            in_features = model.fc.in_features
            self.encode =  nn.Sequential(*list(model.children())[:-2])
            self.decode = create_head(in_features,2)
        def forward(self,x):
            x = self.encode(x)
            x = self.decode(x)
            return x
    class head(nn.Module):
        def __init__(self,in_features=2048, out_features=2):
            super().__init__()
            self.fc1 = nn.Linear(in_features,in_features//4)
            self.relu = nn.ReLU(inplace=True)
            self.bn2 = nn.BatchNorm1d(in_features//4)
            #self.dropout2 = nn.Dropout(0.25,False)
            self.fc2 = nn.Linear(in_features//4,out_features)
        def forward(self,x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.bn2(x)
            #x = self.dropout2(x)
            x = self.fc2(x)
            return x
    class multimodel(nn.Module):
        def __init__(self):
            super().__init__()
            model = torchvision.models.__dict__[args.model](pretrained=True)
            
            model2 = torchvision.models.__dict__[args.model](pretrained=True)
            in_features = model.fc.in_features
            self.encode_ct =  nn.Sequential(*list(model.children())[:-2])
            self.encode_pet = nn.Sequential(*list(model2.children())[:-2])
            self.head = create_head(in_features*2)
            self.head_aux1 = create_head(in_features)
            self.head_aux2 = create_head(in_features)
        def forward(self, x_ct, x_pt):
            x_ct = self.encode_ct(x_ct)
            x_pt = self.encode_pet(x_pt)
            x = torch.cat((x_ct,x_pt),1)
            #x = torch.flatten(x,1)
            x = self.head(x)
            
            aux1 = self.head_aux1(x_ct)
            aux2 = self.head_aux2(x_pt)
            return x, aux1, aux2

    class multimodel2(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torchvision.models.__dict__[args.model](pretrained=True)
            in_features = self.model.fc.in_features
            self.model.fc = head(in_features,2)
            self.model.conv1 = nn.Conv2d(6,64,kernel_size=7,stride=2,padding=3,bias=False)
            #self.encode =  nn.Sequential(*list(model.children())[:-1])
            #self.base = nn.Sequential(nn.Conv2d(6,3,3,1,1,bias=False),nn.BatchNorm2d(3),nn.ReLU())
            #self.head = head(in_features)
        def forward(self, x_ct, x_pt):
            x = torch.cat((x_ct,x_pt),1)
            #x = self.base(x)
            x = self.model(x)         
            return x
    

    print("Creating model")
    # model = torchvision.models.__dict__[args.model](pretrained=True)
    # in_features = model.fc.in_features
    # model.fc = head(in_features,2)
    
    model = NxNet()
    if args.mx == 'pet-ct2':
        model = multimodel2()
    elif args.mx == 'pet-ct':
        model = multimodel()
    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    weight_tensor = torch.cuda.FloatTensor([1,1.1])
    criterion2 = nn.CrossEntropyLoss(weight_tensor)
    
    #criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,1.1]))
    #criterion = AUCMLoss()
    criterion = Focalloss.focal_loss(alpha=[1,args.ratio])
    opt_name = args.opt.lower()
    if opt_name == 'sgd':
        #import torch.optim
        #optimizer = torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=0.05)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        
        # optimizer2 = torch.optim.Adam(
        #     model.parameters(), lr=1e-4,  weight_decay=args.weight_decay)
    elif opt_name == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=args.momentum,
                                        weight_decay=args.weight_decay, eps=0.0316, alpha=0.9)
    elif opt_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.05)
        #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,150,1e-5)
    elif opt_name == 'adamw_amsgrad':
        optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.05,amsgrad=True)
    elif opt_name == 'libauc':
        criterion = AUCMLoss()
        optimizer = PESG(model,criterion.a,criterion.b,criterion.alpha,0.1,1,args.lr,500,1,1e-4)
        
        #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,150,1e-5)
    else:
        raise RuntimeError("Invalid optimizer {}. Only SGD and RMSprop are supported.".format(args.opt))
    if args.lr_sch == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_sch == 'cos':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs)
    elif args.lr_sch == 'cosr':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,args.epochs//2,1,1e-4)
    elif args.lr_sch == "multisteplr":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
    else:
        raise RuntimeError("Invalid optimizer {}. Only step cos cosr are supported.".format(args.opt))
    if args.apex:
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.apex_opt_level
                                          )
    warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, total_iters=5
            )
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, lr_scheduler], milestones=[5]
        )
    #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,10,5e-5)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
    writer = SummaryWriter(args.output_dir+'_'+opt_name+'_'+args.lr_sch+'_'+args.model+'_'+str(args.dataset_num))
    print(args.output_dir+'_'+opt_name+'_'+args.lr_sch+'_'+args.model)
    writer.add_hparams({'loss_ratio':args.ratio,'optimizer':opt_name,'lr_scheduler':args.lr_sch,'lr':args.lr}, {'mixup':args.mixup})
    if args.test_only:
        evaluate(model, criterion2, data_loader_test_final, device,writer,100,0,args.mx,'test')
        return

    print("Start training")
    start_time = time.time()
    maxauc = 0
    maxtestauc = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args.print_freq, writer,args.apex, args.mixup, args.mx)
        lr_scheduler.step()
        auc = evaluate(model, criterion2, data_loader_test, device,writer,100,epoch,args.mx,'val')
        eval_auc = evaluate(model, criterion2, data_loader_test_final, device,writer,100,epoch,args.mx,'test')
        checkpoint = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'auc':auc,
                    'eval_auc':auc}
        utils.save_on_master(
                    checkpoint,
                    os.path.join(args.output_dir, 'model_last.pth'))
        if auc>maxauc:
            if args.output_dir:
                utils.save_on_master(
                    checkpoint,
                    os.path.join(args.output_dir, 'checkpoint_eval.pth'))
                maxauc = auc
        if eval_auc>maxtestauc:
            if args.output_dir:
                utils.save_on_master(
                    checkpoint,
                    os.path.join(args.output_dir, 'checkpoint_test.pth'))
                maxtestauc = eval_auc
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('--split_nums',default=5,type=int,help='K fold split')
    parser.add_argument('--dataset_num', default=0, type=int)
    parser.add_argument('--data-path', default='pet-ct', help='dataset')
    parser.add_argument('--mx', default='pet-ct', help='dataset')
    parser.add_argument('--label', default='N2', help='dataset label')
    parser.add_argument('--mixup', action='store_true',help='Use mixup for training')
    parser.add_argument('--model', default='resnet18', help='model')
    parser.add_argument('--device', default=0,type = int, help='device')
    parser.add_argument('-b', '--batch-size', default=64, type=int)
    parser.add_argument('--ratio', default=1,type=float,help='loss ratio')
    parser.add_argument('--epochs', default=220, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--opt', default='sgd', type=str, help='optimizer')
    parser.add_argument('--lr_sch', default='step',type=str,help ='lr scheduler')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument("--lr-steps",
                        default=[16, 22],
                        nargs="+",
                        type=int,
                        help="decrease lr every step-size epochs (multisteplr scheduler only)",)
    parser.add_argument('--lr-step-size', default=30, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=30, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )
    parser.add_argument('--auto-augment', default=None, help='auto augment policy (default: None)')
    parser.add_argument('--random-erase', default=0.0, type=float, help='random erasing probability (default: 0.0)')

    # Mixed precision training parameters
    parser.add_argument('--apex', action='store_true',
                        help='Use apex for mixed precision training')
    parser.add_argument('--apex-opt-level', default='O1', type=str,
                        help='For apex mixed precision training'
                             'O0 for FP32 training, O1 for mixed precision training.'
                             'For further detail, see https://github.com/NVIDIA/apex/tree/master/examples/imagenet'
                        )

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
