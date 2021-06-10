import numpy as np
import dgl
import model
import torch
import torch.optim as optim
import torch.nn as nn
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import pdb


class error():
    def __init__(self):
        self.costs = dgl.cost_functional()
        self.Q = torch.tensor([[1, 0], [0, 1]],dtype=torch.float)
        self.R = torch.tensor([[1]], dtype = torch.float)

    def value_iteration(self,trajectory, control, old_control, new_control, value_function, on_optimum):
        traj = torch.squeeze(trajectory, 0)

        old_controls = old_control(traj).detach()
        new_controls= new_control(traj).detach()
        diff = torch.squeeze(old_controls - control, 0)
        oc = torch.squeeze(old_controls, 0)

        points_a = torch.matmul(traj, torch.matmul(self.Q, traj.transpose(1,2))) + torch.matmul(oc, torch.matmul(self.R, oc.transpose(1,2)))
        points_b = torch.matmul(new_controls, torch.matmul(self.R, diff))
        points_together = 2*points_b - points_a

        control_loss =0.05* torch.mean(points_together)
        
        if on_optimum == True:
            overall_loss =  torch.squeeze(torch.squeeze(value_function(trajectory[0][0]))) - 0.5*torch.square(trajectory[0][0][0][0])- torch.square(trajectory[0][0][0][1]) #TODO make sure this is correct
        elif on_optimum == False:
            overall_loss =  torch.squeeze(torch.squeeze(value_function(trajectory[0][0]))) - torch.squeeze(torch.squeeze(value_function(trajectory[0][-1]).detach())) + control_loss 

        overall_loss = torch.square(overall_loss)# + torch.square(value_function(torch.tensor([[0,0]], dtype = torch.float)))
        
        return overall_loss

    def policy_improvement(self,trajectory, control, old_control, new_control, value_function, op_factor = 0.5):
        traj = torch.squeeze(trajectory, 0)
        
        old_controls = old_control(traj).detach()
        new_controls= new_control(traj)
        diff = torch.squeeze(old_controls - control, 0)
        oc= torch.squeeze(old_controls, 0)

        points_a = torch.matmul(traj, torch.matmul(self.Q, traj.transpose(1,2))) + torch.matmul(oc, torch.matmul(self.R, oc.transpose(1,2)))
        points_b = torch.matmul(new_controls, torch.matmul(self.R, diff.transpose(1,2)))
        points_together = 2*points_b - points_a
        control_loss =0.05* torch.mean(points_together)#TODO this is anyways always positive?
        #this is either on optimum or on the given value_function
        #overall_loss =  (value_function(trajectory[0][0]).detach() - value_function(trajectory[0][-1]).detach() + control_loss).squeeze()
        #overall_loss = 0.5* traj[0][0][0]**2 + traj[0][0][1]**2 - 0.5* traj[-1][0][0]**2 - traj[-1][0][1]**2  + control_loss
        overall_loss = (1-op_factor)*(value_function(trajectory[0][0]).detach() - value_function(trajectory[0][-1]).detach()).squeeze() + (op_factor)*(0.5* traj[0][0][0]**2 + traj[0][0][1]**2 - 0.5* traj[-1][0][0]**2 - traj[-1][0][1]**2)  + control_loss

        #print((value_function(trajectory[0][0]).detach() - value_function(trajectory[0][-1]).detach()-0.5* traj[0][0][0]**2 - traj[0][0][1]**2 + 0.5* traj[-1][0][0]**2 + traj[-1][0][1]**2))

        '''
            compare_diff = torch.squeeze(-torch.unsqueeze(traj[:,:,1]*traj[:,:,0], 2)- torch.squeeze(control, 1), 0)
            compare_b = torch.matmul(new_controls, torch.matmul(self.R, compare_diff))
            compare_a = torch.matmul(traj, torch.matmul(self.Q, traj.transpose(1,2))) + torch.matmul(-torch.unsqueeze(traj[:,:,1]*traj[:,:,0], 2), torch.matmul(self.R, -torch.unsqueeze(traj[:,:,1]*traj[:,:,0], 2).transpose(1,2)))
            compare_loss = 0.1*torch.mean(2*compare_b -compare_a)
            pdb.set_trace()
            compare = 0.5* traj[0][0][0]**2 + traj[0][0][1]**2 - 0.5* traj[-1][0][0]**2 - traj[-1][0][1]**2 + compare_loss
        '''
        #new_control.zero_grad()
        #overall_loss = 0.05*torch.mean(torch.abs(new_control(traj) + torch.unsqueeze(traj[:,:,0]*traj[:,:,1],1)))
        #overall_loss = torch.mean(torch.abs(new_control(traj) + torch.unsqueeze(traj[:,:,1]*traj[:,:,0],1)))
        #overall_loss =torch.mean(torch.abs(new_control(traj)+torch.ones(6).unsqueeze(1).unsqueeze(1))) 
        
        overall_loss = torch.abs(overall_loss)
        return overall_loss

    def policy_warmup(self,trajectory, control, old_control, new_control, value_function, op_factor = 0.5):
        traj = torch.squeeze(trajectory, 0)
        
        new_control.zero_grad()
        overall_loss = 0.05*torch.mean(torch.abs(new_control(traj) + torch.unsqueeze(traj[:,:,0]*traj[:,:,1],1)))
        return overall_loss




def optimal_value_function(traj):
    #TODO disturbations seem to bring huge instability
    return (0.5* traj[0,0]**2 + traj[0,1]**2)*1

if __name__ == '__main__':
    Writer = SummaryWriter()
    error = error()
    dataset = dgl.Dataset()
    trainset = dataset.create_dataset_different_control_and_starts(amount_startpoints=200)
    train_loader = DataLoader(dataset = trainset, batch_size = 1, shuffle =True)

    
    testset = dataset.create_dataset_different_control_and_starts(amount_startpoints=10)
    test_loader = DataLoader(dataset = testset, batch_size = 1, shuffle =True)
    
    dataset_stretch_factor = len(train_loader)/len(test_loader)

    ''' 
    ##########
    #plotting of the trajectories
    ##########
    for i,piece in enumerate(trainset):
        for step in range(piece[0].size()[0]):
            Writer.add_scalars('trajectories', {'first coord'+str(i): piece[0][step][0][0], 'second coord'+str(i): piece[0][step][0][1]}, step)
        #Writer.add_histogram('trajectories', 
    '''

    print("##################################")
    old_control = model.actor(stabilizing = False, control_dim = 1, space_dim = 2)
    new_control = model.actor(stabilizing = False, control_dim = 1, space_dim = 2)
    value_function = model.critic(positive = True, space_dim = 2)
    costs = dgl.cost_functional()

    control_optimizer = optim.SGD(new_control.parameters(), lr=0.005) #i am unsure about this
    value_optimizer = optim.SGD(value_function.parameters(), lr=0.05)
    #control_optimizer = optim.Adam(new_control.parameters(), lr=0.05)
    #value_optimizer = optim.Adam(value_function.parameters(), lr=0.02)

    lmbda = lambda epoch :1# 0.996
    value_scheduler = optim.lr_scheduler.MultiplicativeLR(value_optimizer, lr_lambda = lmbda)

    #Warmup
    for epoch in range(3):
        print("warmup: ", epoch)
        old_control.train()
        value_function.train()
        new_control.train()
        for j,(x, u) in enumerate(train_loader):
            control_optimizer.zero_grad()
            value_optimizer.zero_grad()
            policy_error = error.policy_warmup(x, u, old_control, new_control, value_function, op_factor = 1)
            #assert policy_error !=  0
            policy_error.backward()
            control_optimizer.step()
 
    #Training
    for epoch in range(10):
        print("epoch: ", epoch)
        old_control.train()
        value_function.train()
        new_control.train()
        for j,(x, u) in enumerate(train_loader):
            #Value iteration
            if False:#epoch < 10:
                control_optimizer.zero_grad()
                value_optimizer.zero_grad()

                value_error= error.value_iteration(x, u, old_control, new_control, value_function, on_optimum = True)
                assert value_error !=  0
                value_error.backward()
                value_optimizer.step()
            
            if False:#epoch >= 40:
                control_optimizer.zero_grad()
                value_optimizer.zero_grad()

                value_error= error.value_iteration(x, u, old_control, new_control, value_function, on_optimum = False)
                assert value_error !=  0
                value_error.backward()
                value_optimizer.step()

            #policy improvement
            if epoch < 4 or epoch >= 6:
                control_optimizer.zero_grad()
                value_optimizer.zero_grad()

                policy_error = error.policy_improvement(x, u, old_control, new_control, value_function, op_factor = 1)
                #assert policy_error !=  0
                policy_error.backward()
                control_optimizer.step()

                if (j + len(train_loader)*epoch) %100 == 0:
                    print(policy_error)
                    #Writer.add_scalars('errors', {'train_policy_error': policy_error,'train_value_error':value_error},j + len(train_loader)*epoch)
                    Writer.add_scalars('errors', {'train_policy_error': policy_error,'train_value_error':0},j + len(train_loader)*epoch)
                    old_control = deepcopy(new_control)
            
            if epoch >= 4 and epoch < 6:
                control_optimizer.zero_grad()
                value_optimizer.zero_grad()

                policy_error = error.policy_improvement(x, u, old_control, new_control, value_function, op_factor = 1)
                #assert policy_error !=  0
                policy_error.backward()
                control_optimizer.step()

                if (j + len(train_loader)*epoch) %100 == 0:
                    print(policy_error, 'policy error')
                    #Writer.add_scalars('errors', {'train_policy_error': policy_error,'train_value_error':value_error}, j + len(train_loader)*epoch)
                    Writer.add_scalars('errors', {'train_policy_error': policy_error,'train_value_error':0}, j + len(train_loader)*epoch)
                    #print(optimal_value_function(x[0][0])-value_function(x[0][0]) + value_function(torch.tensor([[0,0]], dtype = torch.float)), 'difference a')
                    #print(optimal_value_function(x[0][-1])-value_function(x[0][-1])+ value_function(torch.tensor([[0,0]], dtype = torch.float)), 'difference b')
                    old_control = deepcopy(new_control)


        '''TESTING'''
        old_control.eval()
        value_function.eval()
        new_control.eval()
        for j, (x, u) in enumerate(test_loader):
            value_error= error.value_iteration(x, u, old_control, new_control, value_function, on_optimum = True)
            
            policy_error = error.policy_improvement(x, u, old_control, new_control, value_function, op_factor = 1)
            Writer.add_scalars('errors', {'test policy error': policy_error,'test value_error':value_error},(j + len(test_loader)*epoch)*dataset_stretch_factor )
 
        value_scheduler.step()
    
    int_const = value_function(torch.tensor([[0,0]], dtype= torch.float)).detach()
    print(int_const, ' - this is the integration constant')
    op_val_img = np.zeros((3,100,100))
    op_con_img = np.zeros((3,100,100))
    con_img = np.zeros((3,100,100))
    val_img = np.zeros((3,100,100))
    for x_1 in range(100):
        for x_2 in range(100):
            #print("tuple: ", x_1/1000, x_2/1000)
            op_con_img[0][x_1][x_2] = - x_1/1000 * x_2/1000 
            op_val_img[0][x_1][x_2] = 0.5*(x_1/1000) **2 + (x_2/1000 )**2
            val_img[0][x_1][x_2] = value_function(torch.tensor([[x_1/1000 , x_2/1000 ]], dtype = torch.float)).detach() - int_const
            #print(x_1/1000, x_2/1000)
            con_img[0][x_1][x_2] = old_control(torch.tensor([[x_1/1000 , x_2/1000]], dtype = torch.float)).detach()
    #rescaling the images:

    val_max = max(np.amax(val_img), np.amax(op_val_img))
    val_min = min(np.amin(val_img), np.amin(op_val_img))
    con_max = max(np.amax(con_img), np.amax(op_con_img))
    con_min = min(np.amin(con_img), np.amin(op_con_img))
    
    print(np.amin(op_con_img))
    print('con max, con min', con_max, con_min)

    print("control at zero:",new_control(torch.tensor([[0,0]], dtype= torch.float)))
    print("control at 1,1:",new_control(torch.tensor([[0.1,0.1]], dtype= torch.float)))

    

    print(con_img[0], 'con img')
    print(op_con_img[0], 'op con img')
    print('-----------------------')

    val_img = (val_img-val_min)/(val_max - val_min)
    con_img = (con_img-con_min)/(con_max - con_min)
    op_val_img = (op_val_img-val_min)/(val_max-val_min)
    op_con_img = (op_con_img-con_min)/(con_max-con_min)
    
    Writer.add_image('optimal value', op_val_img , 0)
    Writer.add_image('optimal control function', op_con_img , 0)
    Writer.add_image('value_function', val_img , 0)
    Writer.add_image('control_function', con_img , 0)
    Writer.close()
    print(con_img[0], 'con img')
    print(op_con_img[0], 'op con img')
    
    print('-----------------------')
    print(val_img[0], 'val img')
    print(op_val_img[0], 'op val img')
    
    print('done')
    
