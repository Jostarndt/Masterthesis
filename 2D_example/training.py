import numpy as np
import dgl
import model
import torch
import torch.optim as optim
import torch.nn as nn
#import torch.linalg as la
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import pdb



batchsize = 128



class error():
    def __init__(self):
        self.costs = dgl.cost_functional()
        self.Q = torch.tensor([[1, 0], [0, 1]],dtype=torch.float)
        self.R = torch.tensor([[1]], dtype = torch.float)

    def both_iterations_direct_solution(self,trajectory, control, old_control, value_function, theta_u, theta_v):
        #new_control: model that delivers a vector of monomials
        #old_control: exactly the same as new_control -> not needed at all! TODO remove
        #value_function: also list of monomials - in the paper ist theta_v
        #TODO assert dimensionality of theta and monomial basis
        control_monomials = old_control(trajectory)#TODO should have same shape as control! -> this is going to be difficult?
        rho_q = torch.matmul(trajectory, torch.matmul(self.Q, trajectory.transpose(-1,-2))).mean((1,2,3)) #MEAN

        #print(control.size())
        #print(torch.matmul(value_function(trajectory[:,0]), theta_v))
        rho_delta_phi = value_function(trajectory[:,0]) - value_function(trajectory[:,-1])#(torch.matmul(value_function(trajectory[:,0]) - value_function(trajectory[:,-1]), theta_v)) 
        
        control_approx = torch.matmul(control_monomials, theta_u)
        rho_psi = torch.mul(torch.matmul(control_approx, self.R), control_approx).mean(1)
        
        rho_u_psi= torch.matmul(control_approx.unsqueeze(3) - control , control_monomials).mean(1)#torch.matmul(control*control_monomials, theta_u)
        
        pi = rho_q + rho_psi.squeeze()

        #print(rho_delta_phi * theta_v+ rho_u_psi * theta_u) # should be same as cat(rho, rho) * cat(theta, theta)

        #torch.matmul(torch.cat((rho_delta_phi, rho_u_psi), 2), torch.cat((theta_v, theta_u)))
        z = torch.cat((rho_delta_phi, rho_u_psi), 2).squeeze()
        
        #the equation system is now:     z*torch.cat((theta_v, theta_u)) = pi
        
        #TODO
        #pdb.set_trace()
        grad_step = torch.matmul( torch.matmul(z.transpose(0,1),  z), torch.cat((theta_v, theta_u))) - torch.matmul(z.transpose(0,1), pi) #TODO: sure that pi is correct?

        #or do it by hand:
        #theta =torch.matmul(torch.matmul( torch.inverse(torch.matmul(z.transpose(0,1), z)), z.transpose(0,1)),pi) #not invertable?!
        #theta = (z * z )^(-1) *z * pi
        
        #theta = theta - alpha * grad
        theta = torch.cat((theta_v, theta_u))  -  2 * 0.05 * grad_step #TODO check if correct.
        theta_v, theta_u = torch.split(theta, [3,4], dim = 0)#TODO implement torch.size()[] instead of hard coding
        
        residual = torch.abs(torch.matmul(z, theta) - pi).sum()
        return residual, theta_v, theta_u
    
def present_results(value_function, new_control, stepname= "after_training"):
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
            con_img[0][x_1][x_2] = new_control(torch.tensor([[x_1/1000 , x_2/1000]], dtype = torch.float)).detach()
    
    #rescaling the images:
    val_max = max(np.amax(val_img), np.amax(op_val_img))
    val_min = min(np.amin(val_img), np.amin(op_val_img))
    con_max = max(np.amax(con_img), np.amax(op_con_img))
    con_min = min(np.amin(con_img), np.amin(op_con_img))
    
    print(con_img[0], 'con img')
    print(op_con_img[0], 'op con img')
    print('-----------------------')

    val_img = (val_img-val_min)/(val_max - val_min)
    op_val_img = (op_val_img-val_min)/(val_max-val_min)

    con_img = (con_img-con_min)/(con_max - con_min)
    op_con_img = (op_con_img-con_min)/(con_max-con_min)
    
    Writer.add_image('optimal value/'+stepname, op_val_img , 0)
    Writer.add_image('optimal control function/'+stepname, op_con_img , 0)
    Writer.add_image('value_function/'+stepname, val_img , 0)
    Writer.add_image('control_function/'+stepname, con_img , 0)
    Writer.close()
    print(con_img[0], 'con img')
    print(op_con_img[0], 'op con img')
    
    print('-----------------------')
    print(val_img[0], 'val img')
    print(op_val_img[0], 'op val img')


if __name__ == '__main__':
    Writer = SummaryWriter()
    error = error()
    dataset = dgl.Dataset()
    trainset = dataset.create_dataset_different_control_and_starts(amount_startpoints=100)
    train_loader = DataLoader(dataset = trainset, batch_size = batchsize, shuffle =True)

    
    testset = dataset.create_dataset_different_control_and_starts(amount_startpoints=10)
    test_loader = DataLoader(dataset = testset, batch_size = 2, shuffle =True)#2 to bring this into batch format
    
    dataset_stretch_factor =  len(train_loader)/len(test_loader)


    print("##################################")
    
    control_function = model.polynomial_linear_actor()
    value_function = model.polynomial_linear_critic()
    
    theta_u = torch.ones(4)
    theta_v = torch.ones(3)
    
    theta_u =torch.tensor([0, 0, -1, 0], dtype = torch.float)
    theta_v =torch.tensor([0.5, 1, 0], dtype = torch.float)

    costs = dgl.cost_functional()


    #present results!

    #Training and Testing
    for epoch in range(40):
        print("epoch: ", epoch)
        print('++++++++++++++++++++Train++++++++++++')
        for j,(x, u) in enumerate(train_loader):

            residual, theta_v, theta_u = error.both_iterations_direct_solution(trajectory= x, control=u, old_control = control_function, value_function = value_function , theta_u= theta_u, theta_v= theta_v)
            print(residual, theta_v, theta_u)

        '''TESTING'''
        print('Test')
        for j, (x, u) in enumerate(test_loader):
            pass
            #residual, _ , _ = error.both_iterations_direct_solution(trajectory= x, control=u, old_control = control_function, value_function = value_function , theta_u= theta_u, theta_v= theta_v)
            #print(residual, theta_v, theta_u)
    
    #Presenting results
    present_results(value_function, new_control)
    print('parameters of value function', value_function.parameters())
    for name, param in value_function.named_parameters():
        print(name, param.data)
        
    print('done')


