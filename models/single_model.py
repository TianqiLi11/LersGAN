import numpy as np
import torch
import torch.fft
from torch import nn
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
import random
from . import networks
import sys
import math
import torch.nn.functional as F
import torch.fft as fft
from PIL import Image
import numpy as np


class SingleModel(BaseModel):
    def name(self):
        return 'SingleGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        self.opt = opt
        self.input_A = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B = self.Tensor(nb, opt.output_nc, size, size)
        self.input_img = self.Tensor(nb, opt.input_nc, size, size)
        self.input_A_gray = self.Tensor(nb, 1, size, size)
        self.cutoff_ratio = 0.2
        self.high_pass_mask = self.create_frequency_mask(320, 320, 0.2)

        if opt.vgg > 0:
            self.vgg_loss = networks.PerceptualLoss(opt)
            if self.opt.IN_vgg:
                self.vgg_patch_loss = networks.PerceptualLoss(opt)
                self.vgg_patch_loss.cuda()
            self.vgg_loss.cuda()
            self.vgg = networks.load_vgg16("./model", self.gpu_ids)
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif opt.fcn > 0:
            self.fcn_loss = networks.SemanticLoss(opt)
            self.fcn_loss.cuda()
            self.fcn = networks.load_fcn("./model")
            self.fcn.eval()
            for param in self.fcn.parameters():
                param.requires_grad = False
        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)

        skip = True if opt.skip > 0 else False
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids, skip=skip, opt=opt)
        # self.netG_B = networks.define_G(opt.output_nc, opt.input_nc,
        #                                 opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids, skip=False, opt=opt)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids, False)
            if self.opt.patchD:
                self.netD_P = networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_patchD, opt.norm, use_sigmoid, self.gpu_ids, True)
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            # self.load_network(self.netG_B, 'G_B', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                if self.opt.patchD:
                    self.load_network(self.netD_P, 'D_P', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            # self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            if opt.use_wgan:
                self.criterionGAN = networks.DiscLossWGANGP()
            else:
                self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            if opt.use_mse:
                self.criterionCycle = torch.nn.MSELoss()
            else:
                self.criterionCycle = torch.nn.L1Loss()
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG_A.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            if self.opt.patchD:
                self.optimizer_D_P = torch.optim.Adam(self.netD_P.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A)
        # networks.print_network(self.netG_B)
        if self.isTrain:
            networks.print_network(self.netD_A)
            if self.opt.patchD:
                networks.print_network(self.netD_P)
            # networks.print_network(self.netD_B)
        if opt.isTrain:
            self.netG_A.train()
            # self.netG_B.train()
        else:
            self.netG_A.eval()
            # self.netG_B.eval()
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        input_img = input['input_img']
        input_A_gray = input['A_gray']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_A_gray.resize_(input_A_gray.size()).copy_(input_A_gray)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.input_img.resize_(input_img.size()).copy_(input_img)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    


    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.real_A_gray = Variable(self.input_A_gray, volatile=True)
        if self.opt.noise > 0:
            self.noise = Variable(torch.cuda.FloatTensor(self.real_A.size()).normal_(mean=0, std=self.opt.noise/255.))
            self.real_A = self.real_A + self.noise
        if self.opt.input_linear:
            self.real_A = (self.real_A - torch.min(self.real_A))/(torch.max(self.real_A) - torch.min(self.real_A))
        # print(np.transpose(self.real_A.data[0].cpu().float().numpy(),(1,2,0))[:2][:2][:])
        if self.opt.skip == 1:
            self.fake_B, self.latent_real_A = self.netG_A.forward(self.real_A, self.real_A_gray)
        else:
            self.fake_B = self.netG_A.forward(self.real_A, self.real_A_gray)
        # self.rec_A = self.netG_B.forward(self.fake_B)

        self.real_B = Variable(self.input_B, volatile=True)


    def predict(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.real_A_gray = Variable(self.input_A_gray, volatile=True)
        if self.opt.noise > 0:
            self.noise = Variable(torch.cuda.FloatTensor(self.real_A.size()).normal_(mean=0, std=self.opt.noise/255.))
            self.real_A = self.real_A + self.noise
        if self.opt.input_linear:
            self.real_A = (self.real_A - torch.min(self.real_A))/(torch.max(self.real_A) - torch.min(self.real_A))
        # print(np.transpose(self.real_A.data[0].cpu().float().numpy(),(1,2,0))[:2][:2][:])
        if self.opt.skip == 1:
            self.fake_B, self.latent_real_A = self.netG_A.forward(self.real_A, self.real_A_gray)
        else:
            self.fake_B = self.netG_A.forward(self.real_A, self.real_A_gray)
        # self.rec_A = self.netG_B.forward(self.fake_B)

        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        A_gray = util.atten2im(self.real_A_gray.data)
        # rec_A = util.tensor2im(self.rec_A.data)
        # if self.opt.skip == 1:
        #     latent_real_A = util.tensor2im(self.latent_real_A.data)
        #     latent_show = util.latent2im(self.latent_real_A.data)
        #     max_image = util.max2im(self.fake_B.data, self.latent_real_A.data)
        #     return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('latent_real_A', latent_real_A),
        #                     ('latent_show', latent_show), ('max_image', max_image), ('A_gray', A_gray)])
        # else:
        #     return OrderedDict([('real_A', real_A), ('fake_B', fake_B)])
        # return OrderedDict([('fake_B', fake_B)])
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B)])

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake, use_ragan):
        # Real
        pred_real = netD.forward(real)
        pred_fake = netD.forward(fake.detach())
        if self.opt.use_wgan:
            loss_D_real = pred_real.mean()
            loss_D_fake = pred_fake.mean()
            loss_D = loss_D_fake - loss_D_real + self.criterionGAN.calc_gradient_penalty(netD, 
                                                real.data, fake.data)
        elif self.opt.use_ragan and use_ragan:
            loss_D = (self.criterionGAN(pred_real - torch.mean(pred_fake), True) +
                                      self.criterionGAN(pred_fake - torch.mean(pred_real), False)) / 2
        else:
            loss_D_real = self.criterionGAN(pred_real, True)
            loss_D_fake = self.criterionGAN(pred_fake, False)
            loss_D = (loss_D_real + loss_D_fake) * 0.5
        # loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        fake_B = self.fake_B
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B, True)
        self.loss_D_A.backward()
    
    def backward_D_P(self):
        if self.opt.hybrid_loss:
            loss_D_P = self.backward_D_basic(self.netD_P, self.real_patch, self.fake_patch, False)
            if self.opt.patchD_3 > 0:
                for i in range(self.opt.patchD_3):
                    loss_D_P += self.backward_D_basic(self.netD_P, self.real_patch_1[i], self.fake_patch_1[i], False)
                self.loss_D_P = loss_D_P/float(self.opt.patchD_3 + 1)
            else:
                self.loss_D_P = loss_D_P
        else:
            loss_D_P = self.backward_D_basic(self.netD_P, self.real_patch, self.fake_patch, True)
            if self.opt.patchD_3 > 0:
                for i in range(self.opt.patchD_3):
                    loss_D_P += self.backward_D_basic(self.netD_P, self.real_patch_1[i], self.fake_patch_1[i], True)
                self.loss_D_P = loss_D_P/float(self.opt.patchD_3 + 1)
            else:
                self.loss_D_P = loss_D_P
        if self.opt.D_P_times2:
            self.loss_D_P = self.loss_D_P*2
            
            
            
        if self.opt.patchD:
            loss_D_medium = self.backward_D_basic(self.netD_P, self.real_patch_medium, self.fake_patch_medium, False)
            if self.opt.patchD_3 > 0:
                for i in range(self.opt.patchD_3):
                    loss_D_medium += self.backward_D_basic(self.netD_P, self.input_patch_medium, self.fake_patch_1[i], False)
                loss_D_medium = loss_D_medium / float(self.opt.patchD_3 + 1)
            # 合并
            # self.loss_D_P += loss_D_medium
            # print('medium')
            
        self.loss_D_P.backward()


        
        
    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        self.real_A_gray = Variable(self.input_A_gray)
        self.real_img = Variable(self.input_img)
        if self.opt.noise > 0:
            self.noise = Variable(torch.cuda.FloatTensor(self.real_A.size()).normal_(mean=0, std=self.opt.noise/255.))
            self.real_A = self.real_A + self.noise
        if self.opt.input_linear:
            self.real_A = (self.real_A - torch.min(self.real_A))/(torch.max(self.real_A) - torch.min(self.real_A))
        if self.opt.skip == 1:
            self.fake_B, self.latent_real_A = self.netG_A.forward(self.real_img, self.real_A_gray)
        else:
            self.fake_B = self.netG_A.forward(self.real_img, self.real_A_gray)
        if self.opt.patchD:
            w = self.real_A.size(3)
            h = self.real_A.size(2)
            w_offset = random.randint(0, max(0, w - self.opt.patchSize - 1))
            h_offset = random.randint(0, max(0, h - self.opt.patchSize - 1))

            self.fake_patch = self.fake_B[:,:, h_offset:h_offset + self.opt.patchSize,
                   w_offset:w_offset + self.opt.patchSize]
            self.real_patch = self.real_B[:,:, h_offset:h_offset + self.opt.patchSize,
                   w_offset:w_offset + self.opt.patchSize]
            self.input_patch = self.real_A[:,:, h_offset:h_offset + self.opt.patchSize,
                   w_offset:w_offset + self.opt.patchSize]
            
        if self.opt.patchD:  # 使用 medium_patchD
            self.opt.new_patchSize = self.opt.patchSize * 4
            medium_w_offset = random.randint(0, max(0, w - self.opt.new_patchSize - 1))
            medium_h_offset = random.randint(0, max(0, h - self.opt.new_patchSize - 1))

            self.fake_patch_medium = self.fake_B[:, :, medium_h_offset:medium_h_offset + self.opt.new_patchSize,
                                                medium_w_offset:medium_w_offset + self.opt.new_patchSize]
            self.real_patch_medium = self.real_B[:, :, medium_h_offset:medium_h_offset + self.opt.new_patchSize,
                                                medium_w_offset:medium_w_offset + self.opt.new_patchSize]
            self.input_patch_medium = self.real_A[:, :, medium_h_offset:medium_h_offset + self.opt.new_patchSize,
                                                medium_w_offset:medium_w_offset + self.opt.new_patchSize]
            
        if self.opt.patchD_3 > 0:
            self.fake_patch_1 = []
            self.real_patch_1 = []
            self.input_patch_1 = []
            w = self.real_A.size(3)
            h = self.real_A.size(2)
            for i in range(self.opt.patchD_3):
                w_offset_1 = random.randint(0, max(0, w - self.opt.patchSize - 1))
                h_offset_1 = random.randint(0, max(0, h - self.opt.patchSize - 1))
                self.fake_patch_1.append(self.fake_B[:,:, h_offset_1:h_offset_1 + self.opt.patchSize,
                    w_offset_1:w_offset_1 + self.opt.patchSize])
                self.real_patch_1.append(self.real_B[:,:, h_offset_1:h_offset_1 + self.opt.patchSize,
                    w_offset_1:w_offset_1 + self.opt.patchSize])
                self.input_patch_1.append(self.real_A[:,:, h_offset_1:h_offset_1 + self.opt.patchSize,
                    w_offset_1:w_offset_1 + self.opt.patchSize])

            # w_offset_2 = random.randint(0, max(0, w - self.opt.patchSize - 1))
            # h_offset_2 = random.randint(0, max(0, h - self.opt.patchSize - 1))
            # self.fake_patch_2 = self.fake_B[:,:, h_offset_2:h_offset_2 + self.opt.patchSize,
            #        w_offset_2:w_offset_2 + self.opt.patchSize]
            # self.real_patch_2 = self.real_B[:,:, h_offset_2:h_offset_2 + self.opt.patchSize,
            #        w_offset_2:w_offset_2 + self.opt.patchSize]
            # self.input_patch_2 = self.real_A[:,:, h_offset_2:h_offset_2 + self.opt.patchSize,
            #        w_offset_2:w_offset_2 + self.opt.patchSize]

    def backward_G(self, epoch):
        pred_fake = self.netD_A.forward(self.fake_B)
        if self.opt.use_wgan:
            self.loss_G_A = -pred_fake.mean()
        elif self.opt.use_ragan:
            pred_real = self.netD_A.forward(self.real_B)

            self.loss_G_A = (self.criterionGAN(pred_real - torch.mean(pred_fake), False) +
                                      self.criterionGAN(pred_fake - torch.mean(pred_real), True)) / 2
            
        else:
            self.loss_G_A = self.criterionGAN(pred_fake, True)
        
        loss_G_A = 0
        if self.opt.patchD:
            pred_fake_patch = self.netD_P.forward(self.fake_patch)
            if self.opt.hybrid_loss:
                loss_G_A += self.criterionGAN(pred_fake_patch, True)
            else:
                pred_real_patch = self.netD_P.forward(self.real_patch)
                
                loss_G_A += (self.criterionGAN(pred_real_patch - torch.mean(pred_fake_patch), False) +
                                      self.criterionGAN(pred_fake_patch - torch.mean(pred_real_patch), True)) / 2
        if self.opt.patchD_3 > 0:   
            for i in range(self.opt.patchD_3):
                pred_fake_patch_1 = self.netD_P.forward(self.fake_patch_1[i])
                if self.opt.hybrid_loss:
                    loss_G_A += self.criterionGAN(pred_fake_patch_1, True)
                else:
                    pred_real_patch_1 = self.netD_P.forward(self.real_patch_1[i])
                    
                    loss_G_A += (self.criterionGAN(pred_real_patch_1 - torch.mean(pred_fake_patch_1), False) +
                                        self.criterionGAN(pred_fake_patch_1 - torch.mean(pred_real_patch_1), True)) / 2
                    
            if not self.opt.D_P_times2:
                self.loss_G_A += loss_G_A/float(self.opt.patchD_3 + 1)
            else:
                self.loss_G_A += loss_G_A/float(self.opt.patchD_3 + 1)*2
        else:
            if not self.opt.D_P_times2:
                self.loss_G_A += loss_G_A
            else:
                self.loss_G_A += loss_G_A*2
                
        if epoch < 0:
            vgg_w = 0
        else:
            vgg_w = 1
        if self.opt.vgg > 0:
            self.loss_vgg_b = self.vgg_loss.compute_vgg_loss(self.vgg, 
                    self.fake_B, self.real_A) * self.opt.vgg if self.opt.vgg > 0 else 0
            if self.opt.patch_vgg:
                if not self.opt.IN_vgg:
                    loss_vgg_patch = self.vgg_loss.compute_vgg_loss(self.vgg, 
                    self.fake_patch, self.input_patch) * self.opt.vgg
                else:
                    loss_vgg_patch = self.vgg_patch_loss.compute_vgg_loss(self.vgg, 
                    self.fake_patch, self.input_patch) * self.opt.vgg
                if self.opt.patchD_3 > 0:
                    for i in range(self.opt.patchD_3):
                        if not self.opt.IN_vgg:
                            loss_vgg_patch += self.vgg_loss.compute_vgg_loss(self.vgg, 
                                self.fake_patch_1[i], self.input_patch_1[i]) * self.opt.vgg
                        else:
                            loss_vgg_patch += self.vgg_patch_loss.compute_vgg_loss(self.vgg, 
                                self.fake_patch_1[i], self.input_patch_1[i]) * self.opt.vgg
                    self.loss_vgg_b += loss_vgg_patch/float(self.opt.patchD_3 + 1)
                else:
                    self.loss_vgg_b += loss_vgg_patch
            self.loss_G = self.loss_G_A + self.loss_vgg_b*vgg_w
        elif self.opt.fcn > 0:
            self.loss_fcn_b = self.fcn_loss.compute_fcn_loss(self.fcn, 
                    self.fake_B, self.real_A) * self.opt.fcn if self.opt.fcn > 0 else 0
            if self.opt.patchD:
                loss_fcn_patch = self.fcn_loss.compute_vgg_loss(self.fcn, 
                    self.fake_patch, self.input_patch) * self.opt.fcn
                if self.opt.patchD_3 > 0:
                    for i in range(self.opt.patchD_3):
                        loss_fcn_patch += self.fcn_loss.compute_vgg_loss(self.fcn, 
                            self.fake_patch_1[i], self.input_patch_1[i]) * self.opt.fcn
                    self.loss_fcn_b += loss_fcn_patch/float(self.opt.patchD_3 + 1)
                else:
                    self.loss_fcn_b += loss_fcn_patch
            self.loss_G = self.loss_G_A + self.loss_fcn_b*vgg_w

        def calculate_gradients(tensor, axis):
            return torch.abs(tensor - torch.roll(tensor, shifts=1, dims=axis))

        def channel_wise_gradients(tensor):
            # 计算每个通道的梯度
            grad_x = calculate_gradients(tensor, axis=1)  # 水平
            grad_y = calculate_gradients(tensor, axis=0)  # 垂直
            num_channels = tensor.size(0)
            grads = {}
            for i in range(num_channels):
                grads[f'Channel_{i}'] = (grad_x[i, :, :], grad_y[i, :, :])
            return grads
        
        
        
        def discrete_cosine_similarity(x, y):
            # 计算像素级别的点积
            return torch.sum(x * y, dim=(1, 2))

        def color_consistency_loss(P, O, lambda_val=1.0):
            assert P.size() == O.size(), "The shape of P and O must be the same."
            B, C, H, W = P.size()
            # 计算离散余弦相似度损失
            similarity = discrete_cosine_similarity(P, O)
            similarity_loss = -torch.mean(similarity) 
            # 计算亮度损失
            P_avg = P.mean(dim=[1, 2, 3], keepdim=True) 
            O_avg = O.mean(dim=[1, 2, 3], keepdim=True)
            brightness_loss = torch.mean(torch.abs(P_avg - O_avg))
            
            total_loss = similarity_loss + lambda_val * brightness_loss
            return abs(total_loss)
        
        def block_color_consistency_loss(P, O, block_size=32, lambda_val=1.0):
            assert P.size() == O.size(), "The shape of P and O must be the same."
            B, C, H, W = P.size()
            num_blocks_h = H // block_size
            num_blocks_w = W // block_size
            # 计算亮度损失
            P_avg = P.mean(dim=[1, 2, 3], keepdim=True)
            O_avg = O.mean(dim=[1, 2, 3], keepdim=True)
            block_brightness_loss = torch.mean(torch.abs(P_avg - O_avg))
            # 初始化块相似度损失
            block_similarity_loss = 0
            # 遍历每个32x32像素块并计算相似度
            for i in range(num_blocks_h):
                for j in range(num_blocks_w):
                    start_h, end_h = i * block_size, (i + 1) * block_size
                    start_w, end_w = j * block_size, (j + 1) * block_size
                    P_block = P[:, :, start_h:end_h, start_w:end_w]
                    O_block = O[:, :, start_h:end_h, start_w:end_w]
                    block_similarity = discrete_cosine_similarity(P_block, O_block)
                    # 累加所有块的相似度损失
                    block_similarity_loss += -torch.mean(block_similarity)
            # 将块相似度损失平均到每个块
            block_similarity_loss /= (num_blocks_h * num_blocks_w)
            block_brightness_loss /= (num_blocks_h * num_blocks_w)
            block_total_loss = block_similarity_loss + lambda_val * block_brightness_loss
            return abs(block_total_loss)

        block_total_loss = 0.001*block_color_consistency_loss(self.fake_B, self.real_A, block_size=32, lambda_val=1.0)
        color_consistency_loss = 0.0001*color_consistency_loss(self.fake_B, self.real_A, 0.6)
        fin_color_consistency_loss = 0.5*(block_total_loss + color_consistency_loss)

        
        def high_frequency_loss(img1, img2, cutoff_ratio=0.2):
            # 将图片转换到频率域
            fft_img1 = fft.fft2(img1, dim=(-2, -1))
            fft_img2 = fft.fft2(img2, dim=(-2, -1))
            # 应用频率掩模来提取高频分量
            high_freq_img1 = fft_img1 * high_pass_mask
            high_freq_img2 = fft_img2 * high_pass_mask
            # 计算差值的绝对值作为损失
            loss = int(torch.sum(torch.abs(high_freq_img1 - high_freq_img2)))
            return loss

        high_pass_mask = self.high_pass_mask
        high_frequency_loss = 0.00000001*high_frequency_loss(self.fake_B, self.real_A, 0.2)
        


        def calculate_entropy(image_tensor):
            '''图像信息熵'''
            image_tensor = image_tensor.detach().cpu()
            image_numpy = image_tensor.numpy()
            if image_numpy.shape[0] == 1 and image_numpy.shape[1] == 1:
                image_numpy = image_numpy[0, 0]
            if image_numpy.ndim == 4: 
                image_numpy = image_numpy[0, 0]
            image_numpy = image_numpy.astype(np.uint8)
            img = Image.fromarray(image_numpy)
            if img.mode != 'L':
                img = img.convert('L')
            # 计算每个像素值的出现次数
            occurrences, _ = np.histogram(img, bins=range(256), density=True)
            # 防止除以零
            if occurrences.sum() == 0:
                return 0.0
            # 计算概率分布，并添加一个小的正数 epsilon 避免对数为负无穷
            epsilon = 1e-10
            probabilities = (occurrences + epsilon) / (occurrences.sum() + epsilon)
            # 计算信息熵
            entropy = -np.sum(probabilities * np.log2(probabilities))
            return entropy

        entropy_difference_loss = 0.1*abs(calculate_entropy(self.fake_B) - calculate_entropy(self.real_A))

        
        class ExposureLoss(torch.nn.Module):
            def __init__(self, num_bins=256, lambda_uniform=1.0, lambda_peak=1.0):
                super(ExposureLoss, self).__init__()
                self.num_bins = num_bins
                self.lambda_uniform = lambda_uniform
                self.lambda_peak = lambda_peak

            def forward(self, fake_B):
                # 非负
                fake_B = F.relu(fake_B)
                fake_B_flattened = fake_B.view(-1)
                # 计算直方图
                histogram = torch.histc(fake_B_flattened, bins=self.num_bins, min=0, max=1)
                # 为histogram添加一个小的epsilon值避免除以0和log(0)
                epsilon = 1e-6
                histogram = torch.clamp(histogram, min=epsilon)
                # 均匀分布的期望直方图
                uniform_histogram = torch.ones_like(histogram) / self.num_bins
                # 均匀性损失
                uniform_loss = F.kl_div(histogram.log(), uniform_histogram, reduction='mean')
                # 峰值损失
                peak_loss = torch.max(histogram) - histogram.mean()
                total_loss = self.lambda_uniform * uniform_loss + self.lambda_peak * peak_loss
                return total_loss

        exposure_loss = ExposureLoss()
        exposure_loss_value = 0.0000002*exposure_loss(self.fake_B)
        
        # print('self.loss_G:', self.loss_G)
        # print('color_consistency_loss:', color_consistency_loss)
        # print('block_total_loss:', block_total_loss)
        # print('fin_color_consistency_loss:',fin_color_consistency_loss)
        # print('high_frequency_loss', high_frequency_loss)
        # print('entropy_difference_loss:', entropy_difference_loss)
        # print('exposure_loss:', exposure_loss_value)
        
        self.loss_G += fin_color_consistency_loss + high_frequency_loss + entropy_difference_loss + exposure_loss_value
        
        # print('new——self.loss_G:', self.loss_G)
        
        self.loss_G.backward()


    def create_frequency_mask(self, height, width, cutoff_ratio):
        # 创建频率掩模，并确保它在正确的设备上
        mask = torch.zeros((height, width), dtype=torch.float32, device=torch.device('cuda:0'))
        mid_height, mid_width = height // 2, width // 2
        for i in range(height):
            for j in range(width):
                distance = ((i - mid_height)**2 + (j - mid_width)**2)**0.5
                mask[i, j] = 1 if distance > (cutoff_ratio * min(mid_height, mid_width)) else 0
        return mask
    
    
    def optimize_parameters(self, epoch):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G(epoch)
        self.optimizer_G.step()
        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        if not self.opt.patchD:
            self.optimizer_D_A.step()
        else:
            # self.forward()
            self.optimizer_D_P.zero_grad()
            self.backward_D_P()
            self.optimizer_D_A.step()
            self.optimizer_D_P.step()


    def get_current_errors(self, epoch):
        D_A = self.loss_D_A.item()
        D_P = self.loss_D_P.item() if self.opt.patchD else 0
        G_A = self.loss_G_A.item()
        if self.opt.vgg > 0:
            vgg = self.loss_vgg_b.item()/self.opt.vgg if self.opt.vgg > 0 else 0
            return OrderedDict([('D_A', D_A), ('G_A', G_A), ("vgg", vgg), ("D_P", D_P)])
        elif self.opt.fcn > 0:
            fcn = self.loss_fcn_b.item()/self.opt.fcn if self.opt.fcn > 0 else 0
            return OrderedDict([('D_A', D_A), ('G_A', G_A), ("fcn", fcn), ("D_P", D_P)])
        

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        if self.opt.skip > 0:
            latent_real_A = util.tensor2im(self.latent_real_A.data)
            latent_show = util.latent2im(self.latent_real_A.data)
            if self.opt.patchD:
                fake_patch = util.tensor2im(self.fake_patch.data)
                real_patch = util.tensor2im(self.real_patch.data)
                if self.opt.patch_vgg:
                    input_patch = util.tensor2im(self.input_patch.data)
                    if not self.opt.self_attention:
                        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('latent_real_A', latent_real_A),
                                ('latent_show', latent_show), ('real_B', real_B), ('real_patch', real_patch),
                                ('fake_patch', fake_patch), ('input_patch', input_patch)])
                    else:
                        self_attention = util.atten2im(self.real_A_gray.data)
                        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('latent_real_A', latent_real_A),
                                ('latent_show', latent_show), ('real_B', real_B), ('real_patch', real_patch),
                                ('fake_patch', fake_patch), ('input_patch', input_patch), ('self_attention', self_attention)])
                else:
                    if not self.opt.self_attention:
                        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('latent_real_A', latent_real_A),
                                ('latent_show', latent_show), ('real_B', real_B), ('real_patch', real_patch),
                                ('fake_patch', fake_patch)])
                    else:
                        self_attention = util.atten2im(self.real_A_gray.data)
                        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('latent_real_A', latent_real_A),
                                ('latent_show', latent_show), ('real_B', real_B), ('real_patch', real_patch),
                                ('fake_patch', fake_patch), ('self_attention', self_attention)])
            else:
                if not self.opt.self_attention:
                    return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('latent_real_A', latent_real_A),
                                ('latent_show', latent_show), ('real_B', real_B)])
                else:
                    self_attention = util.atten2im(self.real_A_gray.data)
                    return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B),
                                    ('latent_real_A', latent_real_A), ('latent_show', latent_show),
                                    ('self_attention', self_attention)])
        else:
            if not self.opt.self_attention:
                return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])
            else:
                self_attention = util.atten2im(self.real_A_gray.data)
                return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B),
                                    ('self_attention', self_attention)])

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        if self.opt.patchD:
            self.save_network(self.netD_P, 'D_P', label, self.gpu_ids)
        # self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        # self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)

    def update_learning_rate(self):
        
        if self.opt.new_lr:
            lr = self.old_lr/2
        else:
            lrd = self.opt.lr / self.opt.niter_decay
            lr = self.old_lr - lrd
        for param_group in self.optimizer_D_A.param_groups:
            param_group['lr'] = lr
        if self.opt.patchD:
            for param_group in self.optimizer_D_P.param_groups:
                param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
