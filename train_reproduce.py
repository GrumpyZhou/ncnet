import os
import numpy as np
import torch
import torch.optim as optim

from lib.dataloader import DataLoader # modified dataloader
from lib.model import ImMatchNet
from lib.im_pair_dataset import ImagePairDataset
from lib.normalization import NormalizeImageDict
from lib.torch_util import save_checkpoint, str_to_bool
from lib.torch_util import BatchTensorToVars, str_to_bool

from collections import namedtuple

# Seed and CUDA
use_cuda = torch.cuda.is_available()
torch.manual_seed(1)
if use_cuda:
    torch.cuda.manual_seed(1)
np.random.seed(1)

print('ImMatchNet training script')

Args = namedtuple('Args', ['checkpoint', 'image_size', 'num_epochs', 
                           'dataset_image_path', 'dataset_csv_path', 
                           'batch_size', 'lr', 
                           'result_model_fn', 'result_model_dir',
                           'ncons_channels', 'ncons_kernel_sizes',
                           'fe_finetune_params'])
args = Args(checkpoint = '', 
            image_size = 400, 
            num_epochs = 10,
            dataset_image_path = 'datasets/ivd/', 
            dataset_csv_path = 'datasets/ivd/image_pairs/',
            batch_size = 16, 
            lr = 0.0005,
            result_model_fn = 'repro',
            result_model_dir = 'trained_models/ncn_ivd/',
            ncons_channels = [16, 1],
            ncons_kernel_sizes = [3, 3],
            fe_finetune_params = 0 # if 0 only nc-net is trained
           )

# Create model
print('Creating CNN model...')
model = ImMatchNet(use_cuda=use_cuda,
                   checkpoint=args.checkpoint,
                   ncons_kernel_sizes=args.ncons_kernel_sizes,
                   ncons_channels=args.ncons_channels)

# Set which parts of the model to train
if args.fe_finetune_params>0:
    for i in range(args.fe_finetune_params):
        for p in model.FeatureExtraction.model[-1][-(i+1)].parameters(): 
            p.requires_grad=True

print('Trainable parameters:')
for i,p in enumerate(filter(lambda p: p.requires_grad, model.parameters())): 
    print(str(i+1)+": "+str(p.shape))
    
# Optimizer
print('using Adam optimizer')
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
cnn_image_size=(args.image_size,args.image_size)

Dataset = ImagePairDataset
train_csv = 'train_pairs.csv'
test_csv = 'val_pairs.csv'
normalization_tnf = NormalizeImageDict(['source_image','target_image'])
batch_preprocessing_fn = BatchTensorToVars(use_cuda=use_cuda)   

# Dataset and dataloader
dataset = Dataset(transform=normalization_tnf,
	              dataset_image_path=args.dataset_image_path,
	              dataset_csv_path=args.dataset_csv_path,
                  dataset_csv_file = train_csv,
                  output_size=cnn_image_size)

dataloader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, 
                        num_workers=0)

dataset_test = Dataset(transform=normalization_tnf,
	                   dataset_image_path=args.dataset_image_path,
	                   dataset_csv_path=args.dataset_csv_path,
                       dataset_csv_file=test_csv,
                       output_size=cnn_image_size)

dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size,
                        shuffle=True, num_workers=4)
    
# Define checkpoint name
checkpoint_name = os.path.join(args.result_model_dir, args.result_model_fn + '.pth.tar')
print('Checkpoint name: '+checkpoint_name)    
    
# Train
best_test_loss = float("inf")

def weak_loss(model,batch,normalization='softmax',alpha=30):
    if normalization is None:
        normalize = lambda x: x
    elif normalization=='softmax':     
        normalize = lambda x: torch.nn.functional.softmax(x,1)
    elif normalization=='l1':
        normalize = lambda x: x/(torch.sum(x,dim=1,keepdim=True)+0.0001)

    b = batch['source_image'].size(0)
    # positive
    #corr4d = model({'source_image':batch['source_image'], 'target_image':batch['target_image']})
    corr4d = model(batch)
    
    batch_size = corr4d.size(0)
    feature_size = corr4d.size(2)
    nc_B_Avec=corr4d.view(batch_size,feature_size*feature_size,feature_size,feature_size) # [batch_idx,k_A,i_B,j_B]
    nc_A_Bvec=corr4d.view(batch_size,feature_size,feature_size,feature_size*feature_size).permute(0,3,1,2) # 
    
    nc_B_Avec = normalize(nc_B_Avec)
    nc_A_Bvec = normalize(nc_A_Bvec)
    
    # compute matching scores    
    scores_B,_= torch.max(nc_B_Avec,dim=1)
    scores_A,_= torch.max(nc_A_Bvec,dim=1)
    score_pos = torch.mean(scores_A+scores_B)/2

    # negative
    batch['source_image']=batch['source_image'][np.roll(np.arange(b),-1),:] # roll
    corr4d = model(batch)
    #corr4d = model({'source_image':batch['source_image'], 'target_image':batch['negative_image']})
    
    batch_size = corr4d.size(0)
    feature_size = corr4d.size(2)
    nc_B_Avec=corr4d.view(batch_size,feature_size*feature_size,feature_size,feature_size) # [batch_idx,k_A,i_B,j_B]
    nc_A_Bvec=corr4d.view(batch_size,feature_size,feature_size,feature_size*feature_size).permute(0,3,1,2) # 
    
    nc_B_Avec = normalize(nc_B_Avec)
    nc_A_Bvec = normalize(nc_A_Bvec)
        
    # compute matching scores    
    scores_B,_= torch.max(nc_B_Avec,dim=1)
    scores_A,_= torch.max(nc_A_Bvec,dim=1)
    score_neg = torch.mean(scores_A+scores_B)/2  
    
    # loss
    loss = score_neg - score_pos  
    return loss

loss_fn = lambda model,batch: weak_loss(model,batch,normalization='softmax')

# define epoch function
def process_epoch(mode,epoch,model,loss_fn,optimizer,dataloader,batch_preprocessing_fn,use_cuda=True,log_interval=50):
    epoch_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        if mode=='train':            
            optimizer.zero_grad()
        tnf_batch = batch_preprocessing_fn(batch)
        loss = loss_fn(model,tnf_batch)
        loss_np = loss.data.cpu().numpy()
        epoch_loss += loss_np
        if mode=='train':
            loss.backward()
            optimizer.step()
        else:
            loss=None
        if batch_idx % log_interval == 0:
            print(mode.capitalize()+' Epoch: {} [{}/{} ({:.0f}%)]\t\tLoss: {:.6f}'.format(
                epoch, batch_idx , len(dataloader),
                100. * batch_idx / len(dataloader), loss_np))
    epoch_loss /= len(dataloader)
    print(mode.capitalize()+' set: Average loss: {:.4f}'.format(epoch_loss))
    return epoch_loss

train_loss = np.zeros(args.num_epochs)
test_loss = np.zeros(args.num_epochs)

print('Starting training...')

model.FeatureExtraction.eval()

for epoch in range(1, args.num_epochs+1):
    train_loss[epoch-1] = process_epoch('train',epoch,model,loss_fn,optimizer,dataloader,batch_preprocessing_fn,log_interval=1)
    test_loss[epoch-1] = process_epoch('test',epoch,model,loss_fn,optimizer,dataloader_test,batch_preprocessing_fn,log_interval=1)
      
    # remember best loss
    is_best = test_loss[epoch-1] < best_test_loss
    best_test_loss = min(test_loss[epoch-1], best_test_loss)
    save_checkpoint({
        'epoch': epoch,
        'args': args,
        'state_dict': model.state_dict(),
        'best_test_loss': best_test_loss,
        'optimizer' : optimizer.state_dict(),
        'train_loss': train_loss,
        'test_loss': test_loss,
    }, is_best,checkpoint_name)

print('Done!')