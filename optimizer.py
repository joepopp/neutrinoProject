import h5py
import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import datetime
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

import ray
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch

#=======================================================================================================================================================
# Dataset class
# 
#=======================================================================================================================================================
class EventDataset(Dataset):
    def __init__(self, hdf5_file, features, time_features, train=True, transform=None):
        self.hdf5_file = hdf5_file
        self.features = features
            # ['NPulses','First Time','Last Time','Avg Time','STD Time','Max Time','Total Q','Max Q']
        self.time_features = time_features
            # ['First Time','Max Time','Last Time']
        self.labels = ['DirX','DirY','DirZ']
        self.transform = transform
        self.train = train
        
    def __len__(self):
        # counts the number of events (groups) in inputted hdf5 file
        with h5py.File(self.hdf5_file, 'r') as file:
            return len(file.keys())        
        
    def __getitem__(self, idx):
        with h5py.File(self.hdf5_file, 'r') as file:
            
            # Group
            # selects group name (one event)
            group = file[list(file.keys())[idx]]
            
            # Features
            # create numpy array of all feature values in selected event
            dataF = group.get('features')
            allDataFeatures = np.array(dataF, dtype=np.float64)
                
            # create python list of all feature names (same for all events)
            attrsF = list(dataF.attrs.items())[0][1].split(', ')
                
            # Time features
            # create numpy array of time features
            allDataFeaturesT = np.transpose(allDataFeatures)
            for i, feature in enumerate(self.time_features):
                if i == 0:
                    # intitialize array with first feature's data
                    eventTimeFeatures = np.expand_dims(allDataFeaturesT[attrsF.index(feature)], axis=0)
                else:
                    # populate with remainging features' data
                    eventTimeFeatures = np.append(eventTimeFeatures,np.expand_dims(allDataFeaturesT[attrsF.index(feature)], axis=0), axis=0)    
            
            # Requested features
            # create numpy array of requested features
            for i, feature in enumerate(self.features):
                if i == 0:
                    # intitialize array with first feature's data
                    reqDataFeatures = np.expand_dims(allDataFeaturesT[attrsF.index(feature)], axis=0)
                else:
                    # populate with remainging features' data
                    reqDataFeatures = np.append(reqDataFeatures,np.expand_dims(allDataFeaturesT[attrsF.index(feature)], axis=0), axis=0)
            
            # Labels
            # create numpy array of all label values in selected event
            dataL = group.get('labels')
            allDataLabels = np.array(dataL, dtype=np.float64)

            # create python list of all label names (same for all events)
            attrsL = list(dataL.attrs.items())[0][1].split(', ')

            # create numpy array of only requested labels
            reqDataLabels = np.zeros(len(self.labels))
            for i, label in enumerate(self.labels):
                reqDataLabels[i] = allDataLabels[attrsL.index(label)]
            
        # Tensors
        dataFeatureTensor = torch.from_numpy(reqDataFeatures)
        dataLabelsTensor = torch.from_numpy(reqDataLabels)
        #change time features to a 6000X3 tensor 
        eventTimeTensor = torch.transpose(torch.from_numpy(eventTimeFeatures).reshape(len(self.time_features),-1),0,1).unsqueeze(0)
        
        # Transforms
        if self.transform:
            dataFeatureTensor = self.transform(dataFeatureTensor)
            
        return eventTimeTensor.float(), dataFeatureTensor.float(), dataLabelsTensor.float()
    
#=======================================================================================================================================================
# Normalize tranform function
# 
#=======================================================================================================================================================
class MeanNormalize(object):
    def __init__(self, means, stds):
        self.means = means
        self.stds = stds
        
    def __call__(self, sample):
        dataFeatureTensor = sample
        for i, feature in enumerate(dataFeatureTensor):
            dataFeatureTensor[i] = (feature-self.means[i])/(self.stds[i]+(10**-8))
        
        return dataFeatureTensor

#=======================================================================================================================================================
# Time convolution model
# 
#=======================================================================================================================================================
class Time_Convolution(nn.Module):
    def __init__(self, numFeatures, activation):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=(1,numFeatures))
        
        #initialize activation
        self.activation = activation
        
    @staticmethod
    def activation_func(act_str):
        if act_str=="tanh":
            return eval("torch." + act_str)
        elif act_str=="hardtanh" or act_str=="softshrink" or act_str=="tanhshrink" or act_str=="leaky_relu" or act_str=="elu":   
            return eval("F." + act_str)
        
    def forward(self, x):
        batchSize = x.size()[0]
        x = self.activation_func(self.activation)(self.conv(x))
        x = x.view(batchSize,1,60,10,10)
        return x

#=======================================================================================================================================================
# Standard model
# varaiable size
#=======================================================================================================================================================
class Network(nn.Module):
    def __init__(self, numFeatures, activation, kernel, cDepth=3, cWidth=[32,32,4], lDepth=3, lWidth=[150,75], convolve_time=True):
        super().__init__()
        
        #intitialize variable convolutional layers
        self.conv = nn.ModuleList([nn.Conv3d(numFeatures+int(convolve_time), cWidth[0], kernel_size=kernel, padding=kernel//2)])
        for c in range(cDepth-1):
            self.conv.append(nn.Conv3d(cWidth[c], cWidth[c+1], kernel_size=kernel, padding=kernel//2))
        
        #intitialize variable linear layers
        size = [750,60,60,7,7]
        start = cWidth[-1]*size[cDepth-2]
        self.lin = nn.ModuleList([nn.Linear(start, lWidth[0])])
        for l in range(lDepth-2):
            self.lin.append(nn.Linear(lWidth[l], lWidth[l+1]))
        self.lin.append(nn.Linear(lWidth[-1], 3))
        
        #initialize activation
        self.activation = activation
        
    @staticmethod
    def activation_func(act_str):
        if act_str=="tanh":
            return eval("torch." + act_str)
        elif act_str=="hardtanh" or act_str=="softshrink" or act_str=="tanhshrink" or act_str=="leaky_relu" or act_str=="elu":   
            return eval("F." + act_str)
        
    def forward(self, x):
        batchSize = x.size()[0]
        for num, layer in enumerate(self.conv):
            if (num % 2 == 0):
                x = F.avg_pool3d(self.activation_func(self.activation)(layer(x)),2)
            else:
                x = self.activation_func(self.activation)(layer(x))
        x = x.view(batchSize, -1)
        for num, layer in enumerate(self.lin):
            if num < len(self.lin)-1:
                x = self.activation_func(self.activation)(layer(x))
            else:
                x = torch.tanh(layer(x))
        return x
    
#=======================================================================================================================================================
# Loss function
# 
#=======================================================================================================================================================
class CartesionAngularDistanceLoss(nn.Module):
    def __init__(self):
        super(CartesionAngularDistanceLoss, self).__init__()

    def forward(self, outputs, labels):

        # calculate RA and Dec for outputted result
        theta = torch.acos(outputs[:,2]/torch.sqrt(outputs[:,2]**2+outputs[:,1]**2+outputs[:,0]**2)) # must normalize
        ze = torch.tensor(np.pi) - theta
        out_dec = ze - torch.tensor(np.pi/2.0)
        phi = torch.atan2(outputs[:,1], outputs[:,0])
        out_ra = az = (torch.tensor(np.pi) + phi) 

        # calculate RA and Dec for expected result
        theta = torch.acos(labels[:,2]) # already normalized
        ze = torch.tensor(np.pi) - theta
        lab_dec = ze - torch.tensor(np.pi/2.0)
        phi = torch.atan2(labels[:,1], labels[:,0])
        lab_ra = az = (torch.tensor(np.pi) + phi) 

        # calculate the angle of the great circle distance
        c1 = torch.cos(out_dec)
        c2 = torch.cos(lab_dec)
        s1 = torch.sin(out_dec)
        s2 = torch.sin(lab_dec)
        sd = torch.sin(lab_ra-out_ra)
        cd = torch.cos(lab_ra-out_ra)
        loss = torch.atan2(torch.hypot(c2*sd,c1*s2-s1*c2*cd),s1*s2+c1*c2*cd)
        loss = loss.mean()
        
        return loss
    
#=======================================================================================================================================================
# Get data function
# Means and stds calculated previously
# MUST CHANGE DEPENDING ON LOCATION OF DATA
#=======================================================================================================================================================
def get_data(inFeatures, timeFeatures, data_split):
    # create list of randomly sorted HDF5 data files
    files = []
    arr = np.arange(100)
    np.random.seed(42)
    np.random.shuffle(arr)
    for num in arr:
        if (num < 10):
            files.append("/data/user/mcampana/joe_data/PSTv4.2/21002/IC86.2016_NuMu.021002.00000" + str(num) + ".hdf5")
        else:
            files.append("/data/user/mcampana/joe_data/PSTv4.2/21002/IC86.2016_NuMu.021002.0000" + str(num) + ".hdf5")

    # create datasets
    train_datasets = []
    val_datasets = []
    test_datasets = []
    all_means = {'NPulses':0.013346409381564136, 
             'First Time':60.60686515452193, 
             'Last Time':62.153358528008916, 
             'Avg Time':61.018561108972264, 
             'STD Time':0.5179789017030961, 
             'Max Time':60.93466567950771, 
             'Total Q':0.02563008514713864, 
             'Max Q':0.007355732345611767}
    all_stds = {'NPulses':0.5256794225575779, 
            'First Time':857.6918329436753, 
            'Last Time':883.5513861477272, 
            'Avg Time':863.5959767295625, 
            'STD Time':26.445724850740365, 
            'Max Time':862.5870011952006, 
            'Total Q':7.164528063537033,  
            'Max Q':0.41598216422705014}
    means = []
    stds = []
    for feature in inFeatures:
        means.append(all_means[feature])
        stds.append(all_stds[feature])
        
    for i, file in enumerate(files):
        if (i < data_split[0]):
            train_data = EventDataset(file, inFeatures, timeFeatures, train=True, transform=MeanNormalize(means,stds))
            train_datasets.append(train_data)
        elif (i < (data_split[0]+data_split[1])):
            val_data = EventDataset(file, inFeatures, timeFeatures, train=False, transform=MeanNormalize(means,stds))
            val_datasets.append(val_data)
        elif (i < (data_split[0]+data_split[1]+data_split[2])):
            test_data = EventDataset(file, inFeatures, timeFeatures, train=False, transform=MeanNormalize(means,stds))
            test_datasets.append(test_data)
    
    return ConcatDataset(train_datasets), ConcatDataset(val_datasets), ConcatDataset(test_datasets)
    
#=======================================================================================================================================================
# Training function
# 
#=======================================================================================================================================================
def training_loop(config):
    
    inFeatures = config["inFeatures"].split("-")
    timeFeatures = config["timeFeatures"].split("-")
    data_split = [config["train%"],config["validate%"],1]
    
    #initialize models
    #convolutional layers architecture
    all_cWidth = [config["cWidth1"],config["cWidth2"],config["cWidth3"],config["cWidth4"],config["cWidth5"],config["cWidth6"]]
    cWidth = []
    for layer in range(config["cDepth"]):
        cWidth.append(all_cWidth[layer])
    #linear layers architecture
    all_lWidth = [config["lWidth2"],config["lWidth3"],config["lWidth4"],config["lWidth5"],config["lWidth6"]]
    lWidth = []
    for layer in range(config["lDepth"]-1):
        lWidth.append(all_lWidth[layer])
    #full model
    model = Network(len(inFeatures), config["activation"], config["kernel"], config["cDepth"], cWidth, config["lDepth"], lWidth)
    #time feature
    time_model = Time_Convolution(len(timeFeatures), config["timeActivation"])
    
    #select device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model) # not functional (outdated function)
    model.to(device)
    
    #initialize loss
    loss_fn = CartesionAngularDistanceLoss()
    
    #initialuze optimizer
    params = list(model.parameters()) + list(time_model.parameters())
    optimizer = torch.optim.Adam(params, lr=config["lr"])
    
    #load existing checkpoint
    if train.get_checkpoint():
        loaded_checkpoint = train.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, time_model_state, optimizer_state = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
            )
            model.load_state_dict(model_state)
            time_model.load_state_dict(time_model_state)
            optimizer.load_state_dict(optimizer_state)
    
    #load data
    trainset, valset, _ = get_data(inFeatures, timeFeatures, data_split)
    
    train_loader = DataLoader(trainset, batch_size=config["batch_size"], shuffle=True, num_workers=8)
    val_loader = DataLoader(valset, batch_size=config["batch_size"], shuffle=True, num_workers=8)
    
    avg_loss_train = []
    avg_loss_val = []
    best_loss = 1000
    try:
        for epoch in range(10):
            loss_train = 0.0
            for time_tensors, events, locations in train_loader:
                # calcuate time dependent feature
                time_features = time_model(time_tensors.float())
                
                # add time dependent feature to input
                events = torch.cat((events,time_features),1)
                
                #move data to proper device
                events = events.to(device)
                locations = locations.to(device)
                
                # determine a loss for all samples in batch
                outputs = model(events.float())
                loss = loss_fn(outputs, locations)
                if torch.isnan(loss):
                    raise ValueError("Loss must be a number to continue training")

                # back propigation
                optimizer.zero_grad()
                loss.backward()
                params = list(model.parameters()) + list(time_model.parameters())
                nn.utils.clip_grad_norm_(params, 4.0)
                optimizer.step()

                # to determine average loss of epic
                loss_train += loss.item()

            # append loss value for each epoch
            avg_loss_train.append(loss_train/len(train_loader))

            loss_val = 0.0        
            for time_tensors, events, locations in val_loader:
                with torch.no_grad():
                    #calcuate time dependent feature
                    time_features = time_model(time_tensors.float())

                    #add time dependent feature to input
                    events = torch.cat((events,time_features),1)
                    
                    #move data to proper device
                    events = events.to(device)
                    locations = locations.to(device)
                    
                    #determine a loss for all samples in batch
                    outputs = model(events.float())
                    loss = loss_fn(outputs, locations)
                    if torch.isnan(loss):
                        raise ValueError("Loss must be a number to continue training")

                    #to determine average loss of epic
                    loss_val += loss.item()

            #append loss value for each epoch
            target_loss = (loss_val/len(val_loader))
            avg_loss_val.append(target_loss)

            #save checkpoints
            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
                torch.save(
                    (model.state_dict(), time_model.state_dict(), optimizer.state_dict()), path
                )
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
                train.report(
                    {"loss": target_loss, "angle difference": target_loss*180/np.pi},
                    checkpoint=checkpoint,
                )
    
    except KeyboardInterrupt:
        print("Interupted")
              
    except ValueError as exc:
        print("Interupted: Nan")
        
        if not avg_loss_val:
        
            target_loss = np.pi

            #save checkpoints
            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
                torch.save(
                    (model.state_dict(), time_model.state_dict(), optimizer.state_dict()), path
                )
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
                train.report(
                    {"loss": target_loss, "angle difference": target_loss*180/np.pi},
                    checkpoint=checkpoint,
                )

#=======================================================================================================================================================
# Test function
# used by tune
#=======================================================================================================================================================
def test(best_result):
    
    inFeatures = best_result.config["inFeatures"].split("-")
    timeFeatures = best_result.config["timeFeatures"].split("-")
    data_split = [1,1,10]
    
    #initialize models
    #convolutional layers architecture
    all_cWidth = [best_result.config["cWidth1"],best_result.config["cWidth2"],best_result.config["cWidth3"],
                  best_result.config["cWidth4"],best_result.config["cWidth5"],best_result.config["cWidth6"]]
    cWidth = []
    for layer in range(best_result.config["cDepth"]):
        cWidth.append(all_cWidth[layer])
    #linear layers architecture
    all_lWidth = [best_result.config["lWidth2"],best_result.config["lWidth3"],
                  best_result.config["lWidth4"],best_result.config["lWidth5"],best_result.config["lWidth6"]]
    lWidth = []
    for layer in range(best_result.config["lDepth"]-1):
        lWidth.append(all_lWidth[layer])
    #full model
    best_trained_model = Network(len(inFeatures), best_result.config["activation"], best_result.config["kernel"], 
                    best_result.config["cDepth"], cWidth, best_result.config["lDepth"], lWidth)
    #time feature
    best_trained_time_model = Time_Convolution(len(timeFeatures), best_result.config["timeActivation"])
    
    #determine device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    best_trained_model.to(device)

    #load checkpoint models
    checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "checkpoint.pt")
    model_state, time_model_state, optimizer_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)
    best_trained_time_model.load_state_dict(time_model_state)
    
    #initialize loss
    loss_fn = CartesionAngularDistanceLoss()
    
    #load data        
    _, _, testset = get_data(inFeatures, timeFeatures, data_split)
    test_loader = DataLoader(testset, batch_size=10, shuffle=True, num_workers=2)
    
    #create list of angles
    with torch.no_grad():
        angles = []
        for time_tensors, events, locations in test_loader:
            #calcuate time dependent feature
            time_features = best_trained_time_model(time_tensors.float())
            #add time dependent feature to input
            events = torch.cat((events,time_features),1)
            #move data to proper device
            events = events.to(device)
            locations = locations.to(device)
            #determine a loss for all samples in batch
            outputs = best_trained_model(events.float())
            loss = loss_fn(outputs, locations).unsqueeze(dim=0)
            #create list of losses
            angles.append(loss)
        angles = torch.cat(angles)
        angles = angles*180/np.pi

    # calculate batchwise mean of list
    print("Best trial test set average angle: {}".format(angles.mean()))
    
#=======================================================================================================================================================
#-------------------------------------------------------------------------------------------------------------------------------------------------------
# Main
# change tunable paramters here
# MUST CHANGE OUTPUT FILES
#-------------------------------------------------------------------------------------------------------------------------------------------------------
#=======================================================================================================================================================
def main(num_samples, max_num_epochs, gpus_per_trial):
    config = {
        "inFeatures": tune.choice(["Avg Time-STD Time",
                                  "First Time-Avg Time-STD Time",
                                  "Last Time-Avg Time-STD Time",
                                  "Avg Time-STD Time-Max Time",
                                  "First Time-Last Time-Avg Time-STD Time",
                                  "First Time-Last Time-Avg Time-STD Time-Max Time",
                                  "Avg Time-STD Time-Total Q-Max Q",
                                  "NPulses-Avg Time-STD Time",
                                  "NPulses-Avg Time-STD Time-Total Q-Max Q",
                                  "NPulses-Max Time-Total Q-Max Q",
                                  "NPulses-First Time-Total Q-Max Q",
                                   "NPulses-First Time-Last Time-Max Time-Total Q-Max Q",
                                  "NPulses-First Time-Last Time-Avg Time-STD Time-Max Time-Total Q-Max Q",
                                  "NPulses-Total Q-Max Q",
                                  "Total Q-Max Q"]),
        "timeFeatures": tune.choice(["First Time-Max Time-Last Time","First Time-Last Time"]),
        "train%": tune.quniform(1, 10, 1),
        "validate%": tune.quniform(1, 10, 1),
        "activation": tune.choice(["tanh","hardtanh","softshrink","tanhshrink","leaky_relu","elu"]),
        "timeActivation": tune.choice(["tanh","hardtanh","softshrink","tanhshrink","leaky_relu","elu"]),
        "kernel": tune.choice([3,5,7,9]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([10, 25, 50, 75, 100, 150]),
        
        "cDepth": tune.choice([2, 3, 4, 5, 6]),
        "cWidth1": tune.sample_from(lambda _: 2**np.random.randint(2, 6)),
        "cWidth2": tune.sample_from(lambda _: 2**np.random.randint(2, 6)),
        "cWidth3": tune.sample_from(lambda _: 2**np.random.randint(2, 6)),
        "cWidth4": tune.sample_from(lambda _: 2**np.random.randint(2, 6)),
        "cWidth5": tune.sample_from(lambda _: 2**np.random.randint(2, 6)),
        "cWidth6": tune.sample_from(lambda _: 2**np.random.randint(2, 6)),
        "lDepth": tune.choice([2, 3, 4, 5, 6]),
        "lWidth2": tune.sample_from(lambda _: 2**np.random.randint(3, 7)),
        "lWidth3": tune.sample_from(lambda _: 2**np.random.randint(3, 7)),
        "lWidth4": tune.sample_from(lambda _: 2**np.random.randint(3, 7)),
        "lWidth5": tune.sample_from(lambda _: 2**np.random.randint(3, 7)),
        "lWidth6": tune.sample_from(lambda _: 2**np.random.randint(3, 7))
    }

    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    
    algo = OptunaSearch()
    algo = ConcurrencyLimiter(algo, max_concurrent=5)
    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(training_loop),
            resources={"cpu": 1, "gpu": gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            search_alg=algo,
            num_samples=num_samples,
        ),
        param_space=config,
        run_config=train.RunConfig(storage_path="/data/user/jpopp/optimizing_out/training_files")
    )
    results = tuner.fit()
    
    best_result = results.get_best_result("loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))
    print("Best trial final validation angle: {}".format(
        best_result.metrics["angle difference"]))

    test(best_result)
    
#=======================================================================================================================================================
# Input of parameters
# change parameters here
#=======================================================================================================================================================

if __name__ == "__main__":
    
    main(num_samples=400, max_num_epochs=10, gpus_per_trial=1)
