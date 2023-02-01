import torch.nn.functional as F
from torch import nn
import torch
from uci_gas.timeseries_models.tcn import TemporalConvNet
from uci_gas.timeseries_models.dilated_conv import DilatedConvEncoder
from uci_gas.timeseries_models.wavenet import WaveNet
import numpy as np
from matplotlib import pyplot as plt
import utils as utils

class Gas_Model(nn.Module):
    def __init__(self, config):
        super(Gas_Model, self).__init__()

        self.mode = config['method']
        self.class_num = config['class_num']
        self.time_length = config['data']['length']
        
        self.init_gas(config)
        
        self.loss_func = config['train']['loss']
        if self.loss_func == 'arcface' or self.loss_func == 'cosface' or self.loss_func == 'sphereface':
            self.adms_loss = utils.AngularPenaltySMLoss(self.in_features, self.class_num, loss_type=self.loss_func)
        elif self.loss_func == 'CenterLoss':
            self.criterion_cent = utils.CenterLoss(num_classes=self.class_num, feat_dim=self.in_features, use_gpu=True)
            
    def forward(self, inputs, embed=False, out_flag=False):
        """Inputs have to have dimension (N, C_in, L_in)"""
        return self.forward_smr(inputs, embed, out_flag)

    def features(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        return self.forward_smr(inputs, True, False)

    def init_gas(self, config):
        freq_len = self.time_length
        feature_num = config['data']['feature_num']

        self.parallel = config['parallel']
        
        if self.mode=='one_way' or self.mode=='bi_dir' or self.mode=='sp_bi_dir':
            tcn_nhid = config['model']['tcn']['nhid']
            tcn_levels = config['model']['tcn']['levels']
            tcn_kernel_size = config['model']['tcn']['ksize']
            tcn_dropout = config['model']['tcn']['dropout']
            num_channels = [tcn_nhid] * tcn_levels
            input_size = freq_len
            
            if self.mode=='sp_bi_dir':
                self.tcn1 = TemporalConvNet(input_size, num_channels, kernel_size=tcn_kernel_size, dropout=tcn_dropout)
                self.tcn2 = TemporalConvNet(input_size, num_channels, kernel_size=tcn_kernel_size, dropout=tcn_dropout)
            else:
                self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=tcn_kernel_size, dropout=tcn_dropout)
            
            if self.mode=='one_way':
                inputs = num_channels[-1]
            else:
                inputs = num_channels[-1]*2
            
            self.in_features = inputs
        elif self.mode=='h_dialated_conv' or self.mode=='dialated_conv':
            output_dims = config['model']['dialated_conv']['output_dims']
            hidden_dims = config['model']['dialated_conv']['hidden_dims']
            depth = config['model']['dialated_conv']['depth']
            input_dims = freq_len
            if self.mode=='h_dialated_conv':
                int_hidden_dims = input_dims
                self.input_fc = nn.Linear(input_dims, int_hidden_dims)
            self.dialated_conv = DilatedConvEncoder(int_hidden_dims, [hidden_dims] * depth + [output_dims], kernel_size=3)
            # print(self.tcn)
            self.repr_dropout = nn.Dropout(p=0.1)
            self.in_features = output_dims
        elif self.mode=='wavenet':
            self.wavenet = WaveNet(in_depth = self.time_length,
                                    dilation_channels = 32,
                                    res_channels = 32,
                                    skip_channels = 256,
                                    end_channels = 128,
                                    dilation_depth = 4,
                                    n_blocks = 5)
            self.post = nn.Sequential(nn.Dropout(p = 0.2),
                                    nn.Linear(128, 256),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.3))
            self.in_features = 256
        elif self.mode=='rnn' or self.mode=='brnn' or self.mode=='rnn_att' or self.mode=='brnn_att':
            input_dims = feature_num
            n_layers = config['model']['rnn']['n_layers']
            hidden_dim = config['model']['rnn']['hidden_dim']
            bidirectional = False
            if self.mode=='brnn' or self.mode=='brnn_att':
                bidirectional = True
            self.rnn = nn.LSTM(self.time_length, hidden_dim, num_layers = n_layers, bidirectional = bidirectional, batch_first=True)

            if bidirectional:
                hidden_dim = 2 * hidden_dim
            self.lnorm = nn.LayerNorm([input_dims, hidden_dim])
            self.in_features = hidden_dim
        elif self.mode=='crnn':
            n_layers = config['model']['crnn']['n_layers']
            hidden_dim = config['model']['crnn']['hidden_dim']
            bidirectional = True
            self.layer1 = torch.nn.Sequential(
                torch.nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(16),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2, stride=2))
            # 32 = 16 * 2
            self.rnn = nn.LSTM(64, hidden_dim, num_layers = n_layers, bidirectional = bidirectional, batch_first=True)
            self.in_features = hidden_dim*2
        elif self.mode == 'mt_att':
            transformer_layer = nn.TransformerEncoderLayer(
                        d_model = self.time_length, # input feature (frequency) dim after maxpooling 64*xxx -> 64*xxx (MFC*time)
                        nhead = 4, # 4 self-attention layers in each multi-head self-attention layer in each encoder block
                        dim_feedforward = 512, # 2 linear layers in each encoder block's feedforward network: dim 64-->512--->64
                        dropout = 0.4, 
                        activation = 'relu' # ReLU: avoid saturation/tame gradient/reduce compute time
                    )
            # I'm using 4 instead of the 6 identical stacked encoder layrs used in Attention is All You Need paper
            # Complete transformer block contains 4 full transformer encoder layers (each w/ multihead self-attention+feedforward)
            self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=4)
            
            self.in_features = self.time_length
        elif self.mode=='cnn':
            self.layer1 = torch.nn.Sequential(
                torch.nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(32),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2, stride=2))
            self.layer2 = torch.nn.Sequential(
                torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(),
                # torch.nn.MaxPool2d(kernel_size=2, stride=2)
                )

            self.in_features = 64
        else:
            raise ValueError("uci_gas Mode Error")

        self.linear = nn.Linear(self.in_features, self.class_num)

    def attention_net(self, lstm_output, final_state): # lstm_output - (batch size, maxlen, hidden*2), final_state - (1, batch size, hidden*2)
            hidden = final_state.squeeze(0) # (batch size, hidden*2)
            attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2) # (batch size, maxlen)
            soft_attn_weights = F.softmax(attn_weights, 1)
            new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2) # (batch size, hidden*2)

            return new_hidden_state
    
    def forward_smr(self, inputs, embed=False, out_flag=False):
        done = False
        # print('[1] forward_smr', inputs.shape)

        xt = inputs
        # xt = inputs.float().transpose(1,2)
        # # print('[2] forward_smr', xt.shape)
        # # x-> (batch, channel, time)
        # xt = self.spec(xt)

        # print('[3] forward_smr', xt.shape)
        
        if self.mode=='one_way' or self.mode=='bi_dir' or self.mode=='sp_bi_dir':
            # print('xt.shape', xt.shape)
            xt = torch.squeeze(xt)
            # print('xt.shape', xt.shape)
            # batch, 64, 118

            if self.mode=='one_way':
                output = self.tcn(xt)  # input should have dimension (N, C, L)
            elif self.mode=='bi_dir':
                rev_inputs_lst = []
                for input in xt[:]:
                    rev_input = torch.fliplr(input)
                    rev_inputs_lst.append(rev_input)
                rev_inputs = torch.stack(rev_inputs_lst)
                
                y1 = self.tcn(xt)
                y2 = self.tcn(rev_inputs)
                
                output = torch.cat((y1, y2), dim=1)
            else:
                rev_inputs_lst = []
                for input in xt[:]:
                    rev_input = torch.fliplr(input)
                    rev_inputs_lst.append(rev_input)
                rev_inputs = torch.stack(rev_inputs_lst)

                y1 = self.tcn1(xt)
                y2 = self.tcn2(rev_inputs)
                output = torch.cat((y1, y2), dim=1)
            
            # print('y1.shape', y1.shape)
            # print('y1[:, :, -1].shape', y1[:, :, -1].shape)
            # print('y2.shape', y2.shape)
            # print('y2[:, :, -1].shape', y2[:, :, -1].shape)
            # print('output.shape', output.shape)
            
            out = output[:, :, -1]
        elif self.mode=='h_dialated_conv' or self.mode=='dialated_conv':
            # batch, channel, time
            xt = torch.squeeze(xt)

            if self.mode=='h_dialated_conv':
                # print('====> xt.shape', xt.shape)
                x = xt.transpose(1, 2)
                # batch, time, channel
                batch, time, channel = x.shape
                # print('====> x0.shape', x.shape)
                x = x.reshape(batch*time, channel, -1)
                # print('====> x1.shape', x.shape)
                x = torch.squeeze(x)
                # print('====> x0.shape', x.shape)
                z = self.input_fc(x)
                # print('====> z1.shape', z.shape)
                z = z.reshape(batch, time, -1)
                # print('====> z2.shape', z.shape)
                z = z.transpose(1, 2)
                # print('====> z3.shape', z.shape)
            else:
                z = xt
                
            output = self.repr_dropout(self.dialated_conv(z))  # input should have dimension (N, C, L)
            # print('y.shape', y.shape)
            out = output[:, :, -1]
        elif self.mode=='rnn' or self.mode=='brnn' or self.mode=='rnn_att' or self.mode=='brnn_att':
            xt = torch.squeeze(xt)
            # print('xt.shape', xt.shape)
            # batch, 64, 118
            
            z = xt.transpose(1, 2)
            if self.parallel:
                self.rnn.flatten_parameters()
            outputs, (hidden, cell) = self.rnn(z)
            
            # print(outputs.shape)
            outputs = self.lnorm(outputs)

            if self.mode=='rnn_att' or self.mode=='brnn_att':
                out = self.attention_net(outputs, outputs.transpose(0, 1)[-1])
            else:
                out = outputs.transpose(0, 1)[-1]
        elif self.mode=='wavenet':
            xt = torch.squeeze(xt)
            # print('xt.shape', xt.shape)
            # print('======>', xt.shape)
            outputs = self.wavenet(xt)
            # print('======>', outputs.shape)
            out = self.post(outputs[:, :, -1])
            # print('======>', o.shape)
        elif self.mode=='crnn':
            xt = torch.squeeze(xt)
            # print('====> xt : ', xt.shape)
            # N,H,W,C -> N,C,H,W
            xt = xt.permute(0,3,1,2)
            # print('[1]====> xt : ', xt.shape)
            x = self.layer1(xt)
            # print('[2]====>', x.shape)
            
            batch, channel, freq, time = x.shape
            # xt -> (batch, time, channel*freq)
            z = x.reshape(batch, time, -1)
            # print('[z]====>', z.shape)
            
            if self.parallel:
                self.rnn.flatten_parameters()
            outputs, (hidden, cell) = self.rnn(z)
            
            out = outputs[:, -1, :]
        elif self.mode=='mt_att':
            # remove channel dim: 1*64*70 --> 64*70
            xt = torch.squeeze(xt)
            
            # convert maxpooled feature map format: batch * freq * time ---> time * batch * freq format
            # because transformer encoder layer requires tensor in format: time * batch * embedding (freq)
            x = xt.permute(2,0,1) 
            
            # print('x', x.shape)
            
            # finally, pass reduced input feature map x into transformer encoder layers
            transformer_output = self.transformer_encoder(x)
            # print('transformer_output', transformer_output.shape)
            
            # create final feature emedding from transformer layer by taking mean in the time dimension (now the 0th dim)
            # transformer outputs 2x40 (MFCC embedding*time) feature map, take mean of columns i.e. take time average
            transformer_embedding = torch.mean(transformer_output, dim=0) # dim 40x70 --> 40

            # print('transformer_embedding', transformer_embedding.shape)
            
            out = transformer_embedding
        elif self.mode == 'c_mt_att':
            x = self.layer1(xt)
            # print('====>', x.shape)
            x = self.layer2(x)
            # print('====>', x.shape)
            x = self.layer3(x)
            # print('====>', x.shape)
            
            batch, channel, freq, time = x.shape
            # xt -> (batch, time, channel*freq)
            z = x.reshape(batch, -1, time)

            # convert maxpooled feature map format: batch * freq * time ---> time * batch * freq format
            # because transformer encoder layer requires tensor in format: time * batch * embedding (freq)
            x = z.permute(2,0,1)
            
            # print('x', x.shape)
            
            # finally, pass reduced input feature map x into transformer encoder layers
            transformer_output = self.transformer_encoder(x)
            # print('transformer_output', transformer_output.shape)
            
            # create final feature emedding from transformer layer by taking mean in the time dimension (now the 0th dim)
            # transformer outputs 2x40 (MFCC embedding*time) feature map, take mean of columns i.e. take time average
            transformer_embedding = torch.mean(transformer_output, dim=0) # dim 40x70 --> 40

            # print('transformer_embedding', transformer_embedding.shape)
            out = transformer_embedding
                
        elif self.mode=='cnn':
            xt = torch.squeeze(xt)
            # N,H,W,C -> N,C,H,W
            xt = xt.permute(0,3,1,2)
            # print('====> xt.shape', xt.shape)
            x = self.layer1(xt)
            # print('[2] x.shape', x.shape)
            x = self.layer2(x)
            # print('[3] x.shape', x.shape)
            x = F.adaptive_avg_pool2d(x, (1, 1))
            out = torch.squeeze(x)
            # print('out.shape', out.shape)

        if embed:
            if out_flag == False:
                return out
            else:
                o = self.linear(out)
                return out, o
        
        o = self.linear(out)
        # print('o:', o.shape)

        return o
    
    def plot_heatmap(self, arr, fname, pred=''):
        display_flag = False
        save_flag = True
        
        fig = plt.figure(figsize=(8, 6))
        print('arr.shape', arr.shape)
        arr = np.flip(arr.mean(0), axis=0)
        ax = fig.add_subplot(111)

        ax.imshow(arr[::-1], cmap='magma', interpolation='nearest')

        plt.ylim(plt.ylim()[::-1])
        plt.tight_layout()
        ax.text(.99, .98, pred, fontsize=20, 
            color='white',
            fontweight='bold', 
            verticalalignment='top', 
            horizontalalignment='right',
            transform=ax.transAxes)
        if display_flag:
            plt.show()
        if save_flag:
            plt.savefig(fname, format='png')
        # plt.close()
        plt.close(fig)
        # plt.clf()
