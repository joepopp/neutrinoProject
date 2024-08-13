import h5py
import os
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

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

#=======================================================================================================================================================
# Dataset class
# data is processed from files to usable formats here
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
# data is mean normalized by this custom function
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
# computes time dependent feature to be inputted into the standard model
#=======================================================================================================================================================
class Time_Convolution(nn.Module):
    def __init__(self, numFeatures):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=(1,numFeatures))
        
    def forward(self, x):
        batchSize = x.size()[0]
        x = torch.tanh(self.conv(x))
        x = x.view(batchSize,1,60,10,10)
        return x

#=======================================================================================================================================================
# Standard model
#
#=======================================================================================================================================================
class Network(nn.Module):
    def __init__(self, numFeatures, convolve_time=True):
        super().__init__()
        self.conv1 = nn.Conv3d(numFeatures+int(convolve_time), 32, kernel_size=7, padding=3)
        self.conv2 = nn.Conv3d(32, 32, kernel_size=7, padding=3)
        self.conv3 = nn.Conv3d(32, 8, kernel_size=7, padding=3)
        self.lin1 = nn.Linear(480, 64)
        self.lin2 = nn.Linear(64, 16)
        self.lin3 = nn.Linear(16, 3)
        
    def forward(self, x):
        batchSize = x.size()[0]
        x = F.avg_pool3d(F.elu(self.conv1(x)),2)
        x = F.elu(self.conv2(x))
        x = F.avg_pool3d(F.elu(self.conv3(x)),2)
        x = x.view(batchSize, -1)
        x = F.elu(self.lin1(x))
        x = F.elu(self.lin2(x))
        x = F.hardtanh(self.lin3(x))
        return x
    
#=======================================================================================================================================================
# Loss function
# calulates the angular distance between expected and predicted locations in the sky
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
# extracts data from files and creates requires dataloaders and datasets, normalizes using previously calulated means and stds
# MUST CHANGE DEPENDING ON LOCATION OF DATA
#=======================================================================================================================================================
def get_data(inFeatures, timeFeatures, trainPercent, valPercent, testPercent, batchSize):
    # create list of randomly sorted HDF5 data files
    files = []
    arr = np.arange(100)
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
        if (i < trainPercent):
            train_data = EventDataset(file, inFeatures, timeFeatures, train=True, transform=MeanNormalize(means,stds))
            train_datasets.append(train_data)
        elif (i < (valPercent+trainPercent)):
            val_data = EventDataset(file, inFeatures, timeFeatures, train=False, transform=MeanNormalize(means,stds))
            val_datasets.append(val_data)
        elif (i < (testPercent+valPercent+trainPercent)):
            test_data = EventDataset(file, inFeatures, timeFeatures, train=False, transform=MeanNormalize(means,stds))
            test_datasets.append(test_data)

    # create dataloaers
    test_dataset = ConcatDataset(test_datasets)
    train_loader = DataLoader(ConcatDataset(train_datasets), batch_size=batchSize, shuffle=False, num_workers=8, sampler=DistributedSampler(ConcatDataset(train_datasets)))
    val_loader = DataLoader(ConcatDataset(val_datasets), batch_size=batchSize, shuffle=False, num_workers=8, sampler=DistributedSampler(ConcatDataset(val_datasets)))
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=2)
    
    return train_loader, val_loader, test_loader, test_dataset
    
#=======================================================================================================================================================
# Training function
# trains both time convolution model and standard model, updates a loss graph throughout training to track progress
# MUST CHANGE OUTPUT FILES
#=======================================================================================================================================================
def training_loop(n_epochs, model, time_model, optimizer, loss_fn, train_loader, val_loader, device):
    avg_loss_train = []
    avg_loss_val = []
    best_loss = 1000
    if device==0:
        start = datetime.datetime.now()
        #print(f'Start time: {start}')
    try:
        for epoch in range(n_epochs):
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
                params = list(model.module.parameters()) + list(time_model.parameters())
                nn.utils.clip_grad_norm_(params, 2.5)
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

                    #to determine average loss of epic
                    loss_val += loss.item()

            #append loss value for each epoch
            target_loss = (loss_val/len(val_loader))
            avg_loss_val.append(target_loss)

            #save best parameters
            if target_loss < best_loss:
                torch.save({'epoch': epoch+1, 
                            'model_state_dict': model.module.state_dict(),
                            'time_model_state_dict': time_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(), 
                            'loss': (target_loss)}, 
                            '/data/user/jpopp/training_out/best_time_model_test.pt')
                best_loss = target_loss

            if (((epoch+1) < 10) or ((epoch+1) % 10 == 0) or ((epoch+1) == n_epochs))  and device == 0:
                print(f'Epoch {epoch+1}, Training angle: {avg_loss_train[epoch]}, Validating angle: {avg_loss_val[epoch]}')
                
            if (epoch+1) == 10  and device == 0:
                difference = datetime.datetime.now() - start
                print(f'Expected finish time: {(difference*(n_epochs/10)) + start}')
                print(f'Duration: {difference*(n_epochs/10)}')
                
            plt.clf()    
            plt.plot(range(len(avg_loss_train)), avg_loss_train, label = 'Train')
            plt.plot(range(len(avg_loss_val)), avg_loss_val, label = 'Validate')
            plt.plot([], [], ' ', label=f'Loss: {best_loss}')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend()
            plt.savefig('/data/user/jpopp/training_out/loss_graph_test.png')
    
    except KeyboardInterrupt:
        if (len(avg_loss_train) % 10 != 0 and len(avg_loss_train) > 10  and device == 0):
            print(f'Epoch {len(avg_loss_train)}, Training angle: {avg_loss_train[-1]}, Verifying angle: {avg_loss_val[-1]}')
              
    except ValueError as exc:
        if (len(avg_loss_train) % 10 != 0 and len(avg_loss_train) > 10  and device == 0):
            print(f'Epoch {len(avg_loss_train)}, Training angle: {avg_loss_train[-1]}, Verifying angle: {avg_loss_val[-1]}')
        print(exc)
                
    finally:
        # plot loss over entire training peroid
        plt.clf()
        plt.plot(range(len(avg_loss_train)), avg_loss_train, label = 'Train')
        plt.plot(range(len(avg_loss_val)), avg_loss_val, label = 'Validate')
        plt.plot([], [], ' ', label=f'Final Loss: {best_loss}')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig('/data/user/jpopp/training_out/loss_graph_test.png')

        if device==0:
            end = datetime.datetime.now()
            print(f'End time: {end}')
            print(f'Total training time: {end-start}')
        
#=======================================================================================================================================================
# Test function
# runs all test data through some model and outputs results in text and a histogram
# MUST CHANGE OUTPUT FILES
#=======================================================================================================================================================
def test(model, time_model, parameters, train_loader, val_loader, test_loader, loss_fn):
    test_model = torch.load(parameters)
    print("============================================================================================================")
    print(f"Model created after epoch {test_model['epoch']} with an average error of {test_model['loss']*180/np.pi}°")
    print("============================================================================================================")
    
    # replace model with test one
    model.load_state_dict(test_model['model_state_dict'])
    model.eval()
    time_model.load_state_dict(test_model['time_model_state_dict'])
    time_model.eval()
    
    # find average angle in each data loader
    for name, loader in [("Train", train_loader), ("Validate", val_loader), ("Test", test_loader)]:
        with torch.no_grad():
            angles = []
            for time_tensors, events, locations in loader:
                time_features = time_model(time_tensors.float())
                events = torch.cat((events,time_features),1)
                outputs = model(events.float())
                loss = loss_fn(outputs, locations).unsqueeze(dim=0)
                angles.append(loss)
            angles = torch.cat(angles)
            angles = angles*180/np.pi
        
        # calculate batchwise mean and std
        median = angles.median()
        mean = angles.mean()
        std = angles.std() 
        
        print(f"{name} || Median angle difference: {median}°, Standard deviation: {std}°")
        
        # create a histogram of test data
        if name == "Test":
            bins=100
            angles.numpy()
            plt.clf()
            plt.hist(angles, bins=np.logspace(np.log10(0.001),np.log10(180), bins), histtype='step')
            plt.semilogx()
            plt.axvline(x=mean, color='red', label = mean.item())
            plt.axvline(x=median, color='orange', label = median.item())
            plt.title('With Time Convolution')
            plt.xlabel('Anglular Error (°)')
            plt.ylabel('Rate')
            plt.legend(loc='upper left')
            plt.savefig('/data/user/jpopp/training_out/angles_histogram_test.png')

#=======================================================================================================================================================
# Check function
# runs a random event through some model and outputs the results
#=======================================================================================================================================================
def check(model, time_model, dataset, loss_fn):
    
    # choose random sample in dataset to put through model
    with torch.no_grad():
        test_num = np.random.randint(len(dataset)-1)
        time_tensor, event, location = dataset[test_num]
        time_feature = time_model(time_tensor.float())
        event = torch.cat((event,time_feature.squeeze(0)),0)
        output = model(torch.unsqueeze(event.float(),0))
        loss = loss_fn(output, location.unsqueeze(dim=0))
        
    # display difference
    print(f'Expected XYZ: {location[0]}, {location[1]}, {location[2]}')
    print(f'Predicted XYZ: {output[:,0].item()}, {output[:,1].item()}, {output[:,2].item()}')
    print(f'Angle difference: {loss*180/np.pi}°')
    
#=======================================================================================================================================================
#-------------------------------------------------------------------------------------------------------------------------------------------------------
# Main
# MUST CHANGE OUTPUT FILES
#-------------------------------------------------------------------------------------------------------------------------------------------------------
#=======================================================================================================================================================
def main(rank, world_size, epochs, inFeatures, timeFeatures, dataSplit, lr, batchSize):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size) 
    
    # initialize models, data, optimizer, and loss function
    data = get_data(inFeatures, timeFeatures, dataSplit[0], dataSplit[1], dataSplit[2], batchSize)
    model = Network(len(inFeatures))
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
    time_model = Time_Convolution(len(timeFeatures))
    params = list(model.parameters()) + list(time_model.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    loss_fn = CartesionAngularDistanceLoss()

    # train model
    training_loop(epochs, model, time_model, optimizer, loss_fn, data[0], data[1], rank)
    destroy_process_group()

    if rank == 0:
        # create model for testing
        model = Network(len(inFeatures))
        # find result averages
        test(model, time_model, "/data/user/jpopp/training_out/best_time_model_test.pt", data[0], data[1], data[2], loss_fn)
        print("============================================================================================================")
        # calculate outputs for three events
        check(model, time_model, data[3], loss_fn)
        print("---------------------------------------------------------------------------------------------")
        check(model, time_model, data[3], loss_fn)
        print("---------------------------------------------------------------------------------------------")
        check(model, time_model, data[3], loss_fn)

#=======================================================================================================================================================
# Input of hyperparameters
# CHANGE PARAMETERS HERE
#=======================================================================================================================================================
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    
    # initialize inputs
    epochs = 1000
    inFeatures = ['First Time','Last Time','Avg Time','STD Time']
    timeFeatures = ['First Time','Max Time','Last Time'] # in chronological order
    dataSplit = (15,3,10) # (training, validating, testing)
    lr = 0.0013
    batchSize = 100
    
    # run training
    mp.spawn(main, args=(world_size, epochs, inFeatures, timeFeatures, dataSplit, lr, batchSize),nprocs=world_size)
    

    
