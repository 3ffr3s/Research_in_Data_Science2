import sys
sys.path.append('/workspace/sy/sungyun/mdpi_IDS')

from Utils import PklsFolder, make_cls_idx, split_train_val_test
from Layers import Flow_CLF
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sys


if __name__ == '__main__':
    
    #parameters
    batch_size = 32
    num_epoch = 100

    num_layers = 2 # Byte_EncoderLayer 개수
    d_model = 40
    num_heads = 4
    d_k = 10
    d_v = 10
    d_hid = (d_model * 2) #PositionwiseFeedForward hidden dim
    add_attn_dim = (d_model * 2) # PacketEncoder attn_dim
    pck_len = 350
    num_classes = 2
    dropout = 0.1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = 'cuda:7'
    
    flow_dataset = PklsFolder('../../../pkls')  # ISCX 2012 Dataset 
    train_set_idx = np.load('train_set_idx_ISCX2012.npy')
    train_random_sampler = torch.utils.data.SubsetRandomSampler(train_set_idx)
    train_dataloader = DataLoader(flow_dataset, batch_size = batch_size, shuffle = False,  sampler = train_random_sampler)
    model = Flow_CLF(num_layers, d_model,num_heads, d_k, d_v, d_hid, add_attn_dim, pck_len, device, num_classes = num_classes, dropout = dropout).to(device)
    #model = torch.nn.DataParallel(model, device_ids = [7,0,1,2,3,4,5,6], output_device = 7)
    model = torch.nn.DataParallel(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    best= {'loss' : sys.float_info.max}
    #attn_model_save_path = './model'
    attn_model_save_path = './test_model'
    loss_history = []

    model.train()

    for epoch in range(num_epoch):
        epoch_loss = 0

        for batch_id, (train_x, train_y) in enumerate(tqdm(train_dataloader)):  

            optimizer.zero_grad()
            preds = model(train_x.type(torch.long).to(device))
            loss = criterion(preds, train_y.type(torch.long).to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if (batch_id+1) % 100 == 0:
                print(f' iter : {batch_id+1} avg_loss : {epoch_loss/(batch_id+1)}')
                print(f' iter : {batch_id+1} accuracy : {((preds.argmax(dim=1) == train_y.type(torch.long).to(device)).sum().item())/batch_size}')
        loss_history.append(epoch_loss)
        if epoch_loss < best['loss']:
            best['state'] = model.state_dict()
            best['loss'] = epoch_loss
            best['epoch'] = epoch

        print(f'epoch : {epoch}, total train loss : {epoch_loss}') 
        with open(f'{attn_model_save_path}/model_epoch_{epoch}.pt','wb') as f:
            torch.save({
                'state' : model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'epoch' : epoch,
                'loss_history' : loss_history,
                'best_loss' : best['loss']
            },f)

    with open(f'{attn_model_save_path}/best_model.pt', 'wb') as f:
        torch.save(
        {
            'state' : best['state'],
            'best_epoch' : best['epoch'],
            'loss_history' : loss_history
        }, f)
