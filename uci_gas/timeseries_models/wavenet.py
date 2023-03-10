import torch
from torch import nn
import torch.nn.functional as F
from .wavenet_layer_utils import ResidualBlock,CausalConv1d
from utils import *
from .wavenet_layer_utils import SELayer1d
class WaveNet(nn.Module):
	def __init__(self,in_depth=96,
					dilation_channels=32,
					res_channels=32,
					skip_channels=256,
					end_channels=64,
					kernel_size=2,
					bias=False,
					dilation_depth=7,n_blocks=4):
		super(WaveNet,self).__init__()
		self.n_blocks = n_blocks
		self.dilation_depth = dilation_depth

		self.pre_conv = nn.Conv1d(in_depth,res_channels,kernel_size,bias=bias)
		self.dilations = []
		self.resblocks = nn.ModuleList()
		init_dilation=1
		receptive_field = 2
		for i in range(n_blocks):
			addition_scope = kernel_size-1
			new_dilation = 1
			for i in range(dilation_depth):
				self.dilations.append((new_dilation,init_dilation))
				self.resblocks.append(ResidualBlock(dilation_channels,res_channels,
														skip_channels,kernel_size,bias))
				receptive_field+=addition_scope
				addition_scope*=2
				init_dilation = new_dilation
				new_dilation*=2


		self.post = nn.Sequential(#SELayer1d(skip_channels),
									nn.ELU(),
									nn.Conv1d(skip_channels,skip_channels,1,bias=True),
									nn.BatchNorm1d(skip_channels),
									#SELayer1d(skip_channels),                                
									nn.ELU(),
									nn.Conv1d(skip_channels,end_channels,1,bias=True))
		self.receptive_field = receptive_field
	def forward(self,inputs):
		# print('WaveNet inputs:', inputs.shape)
		x = self.pre_conv(inputs)
		# print('WaveNet x:', x.shape)
		#print x.size()
		skip = 0

		for i in range(self.n_blocks*self.dilation_depth):
			(dilation,init_dilation) = self.dilations[i]
			# print('WaveNet dilation,init_dilation : ', dilation,init_dilation)
			x,s = self.resblocks[i](x,dilation,init_dilation)
			# print('WaveNet x,s : ', x.shape, s.shape)
			try:
				skip = skip[:,:,-s.size(2):]
				# print('WaveNet skip1 : ', skip.shape)
			except:
				skip = 0
				# print('WaveNet skip1 : ', skip)
			#if not isinstance(skip,int):
				#print 'skip',skip.size(),'s',s.size()
			skip = skip+s
			# print('WaveNet skip2 : ', skip.shape)
		outputs = self.post(skip)

		return outputs