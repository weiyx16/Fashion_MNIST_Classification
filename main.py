'''

Image Classification with finetuning or feature extraction on pretrained resnet-50
10 kind of leaves provided from assistant teachers
Pytorch 1.1.0 & python 3.6

Author: @weiyx16.github.io
weiyx16@mails.tsinghua.edu.cn

adapted from https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

'''
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image
from tqdm import tqdm
from datetime import date
from MyDataset import CustomTensorDataset

print("PyTorch Version: ",torch.__version__)

# hyperparameter
# Top level data directory. Here we assume the format of the directory conforms to the ImageFolder structure

train_data_dir = r"./data/train.npy"
train_gt_dir = r"./data/train.csv"
validation_data_dir = r"./data/validation.npy"
validation_gt_dir = r"./data/validation.csv"
test_data_dir = r"./data/test.npy"

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "resnet"

# Number of classes in the dataset
num_classes = 10

# Batch size for training (change depending on how much memory you have)
batch_size = 12

# Number of epochs to train
num_epochs = 10

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = False

def train_model(model, dataloaders, criterion, optimizer, num_epochs=15, is_inception=False):
    since = time.time()

    # validation accuracy
    val_acc_history = []
    # for save the best accurate model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in tqdm(range(num_epochs), ncols=70):
        print('\n [*] Epoch {}/{}'.format(epoch, num_epochs - 1))

        # Each epoch has a training and validation phase
        # In fact the input has two dataloader(one for train and one for test)
        for phase in ['train', 'val']:
            print(' [**] Begin {} ...'.format(phase))
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            # Another way: dataIter = iter(dataloaders[phase]) then next(dataIter)
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                # labels = torch.tensor(labels, dtype=torch.long, device=device)
                labels = labels.squeeze().long().to(device)
                # labels = labels.long()
                # labels = Variable(torch.FloatTensor(inputs.size[0]).uniform_(0, 10).long())

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                # notice the validation set will run this with block but do not set gradients trainable
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        # criterion define the loss function
                        # calculate the loss also on the validation set
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    # along the batch axis
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize parameters only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

    time_elapsed = time.time() - since
    print(' Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print(' Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def set_parameter_requires_grad(model, feature_extracting):
    """
    When feature extract with pretrained model, we needn't retrain the parameters before FC
    But different when fine tune
    """
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract=True, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    # Other wise we will need to define the structure by ourselves with forward function using module and sequential to organize
    
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet52
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features # (fc): Linear(in_features=2048, out_features=1000, bias=True)
        model_ft.fc = nn.Linear(num_ftrs, num_classes) # replace fc with 2048 to num_class
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

def inference(model, dataloader):
    since = time.time()

    test_result = None
    for inputs, _ in dataloader:
        inputs = inputs.to(device)

        # forward
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            if test_result is not None:
                test_result = torch.cat((test_result, preds))
            else:
                test_result = preds

    time_elapsed = time.time() - since
    print(' Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return test_result

if __name__ == "__main__":
    # Step1 Model:
    # Initialize the model for this run
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

    # Print the model we just instantiated
    # print(model_ft)

    # Step2 Dataset:
    # Data augmentation and normalization function for training
    # Also rgb2gray
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(input_size, interpolation=Image.NEAREST),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(input_size, interpolation=Image.NEAREST),
            transforms.CenterCrop(input_size),
            transforms.ToTensor()
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(input_size, interpolation=Image.NEAREST),
            transforms.ToTensor()
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    train_data = np.load(train_data_dir)
    train_gt = np.genfromtxt(train_gt_dir, delimiter=',')
    train_gt = train_gt[1:,]
    val_data = np.load(validation_data_dir)
    val_gt = np.genfromtxt(validation_gt_dir, delimiter=',')
    val_gt = val_gt[1:,]
    test_data = np.load(test_data_dir)

    print(" >> Initializing Datasets and Dataloaders")
    
    train_tensor_x = torch.stack([torch.Tensor(i) for i in train_data])
    train_tensor_x = train_tensor_x.reshape((-1, 1, 28, 28))
    # train_tensor_x = train_data.reshape((-1, 1, 28, 28))
    # train_tensor_x = torch.tensor(train_tensor_x)

    train_tensor_y = torch.stack([torch.Tensor(np.asarray(i[1])) for i in train_gt])
    train_tensor_y = train_tensor_y.reshape(train_tensor_y.shape[0], 1)

    val_tensor_x = torch.stack([torch.Tensor(i) for i in val_data])
    val_tensor_x = val_tensor_x.reshape((-1, 1, 28, 28))

    val_tensor_y = torch.stack([torch.Tensor(np.asarray(i[1])) for i in val_gt])
    val_tensor_y = val_tensor_y.reshape(val_tensor_y.shape[0], 1)

    test_tensor_x = torch.stack([torch.Tensor(i) for i in test_data])
    test_tensor_x = test_tensor_x.reshape((-1, 1, 28, 28))

    tensor_x={'train':train_tensor_x, 'val':val_tensor_x, 'test':test_tensor_x}
    tensor_y={'train':train_tensor_y, 'val':val_tensor_y, 'test':None}

    # Create valing and validation datasets
    image_datasets_dict = {phase: CustomTensorDataset(tensors=(tensor_x[phase], tensor_y[phase]), 
                                                        transform=data_transforms[phase], 
                                                        is_training=False if phase=='test' else True, 
                                                        clone_to_three=True) for phase in ['train', 'val', 'test']}
    # Create training and validation dataloaders
    dataloaders_dict = {phase: DataLoader(image_datasets_dict[phase], 
                                            batch_size=batch_size, 
                                            shuffle=False if phase=='test' else True, 
                                            num_workers=4) for phase in ['train', 'val', 'test']}

    # Step3 Transfer to GPU
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Step4 Optimizer
    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                # print("\t",name)
    else:
        params_to_update = model_ft.parameters()
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                pass
                # print("\t",name)
    
    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(params_to_update, lr=1e-3) #optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    # Step5 Loss and train
    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()
    print(' >> Model Created And Begin Training')
    # Train and evaluate
    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))
    if not os.path.exists("./model"):
        os.mkdir("./model")
    torch.save(model_ft.state_dict(), './model/{}-{}.pkl' .format(model_name, date.today())) #model = model_object.load_state_dict(torch.load('params.pkl'))
    
    # show training result
    plt.figure(1)
    plt.title("Validation Accuracy vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Validation Accuracy")
    plt.plot(range(1,num_epochs+1),hist)
    plt.ylim((0,1.))
    plt.xticks(np.arange(1, num_epochs+1, 1.0))
    #plt.show()

    # model_ft.load_state_dict(torch.load('./model/resnet-2019-11-22.pkl'))
    # model_ft = model_ft.to(device)
    # run test
    model_ft.eval()
    test_result = inference(model_ft, dataloaders_dict['test'])
    test_result = test_result.cpu().detach().numpy()
    # test_result = test_result.reshape((test_result.shape[0],1))
    csv_file = np.zeros((test_result.shape[0],2), dtype=np.int32)
    csv_file[:,0] = np.arange(test_result.shape[0])
    csv_file[:,1] = test_result
    with open("./data/test.csv", "wb") as f:
        f.write(b'image_id,label\n')
        np.savetxt(f, csv_file.astype(int), fmt='%i', delimiter=",")
    # np.savetxt('./data/test.csv', csv_file, delimiter=',', header='image_id,label')