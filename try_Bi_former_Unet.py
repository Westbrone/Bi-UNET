import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from .Attention_Caculate import Attention as attn
from .Attention_Caculate import AttentionLePE



class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        """
        x: NHWC tensor
        """
        x = x.permute(0, 3, 1, 2) #NCHW
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) #NHWC

        return x

##### define a Bi-Unet-Block
class BiFormerBlock(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=-1,
                       num_heads=8, n_win=7, qk_dim=None, qk_scale=None,
                       kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='ada_avgpool',
                       topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False, mlp_ratio=4, mlp_dwconv=False,
                       side_dwconv=5, before_attn_dwconv=3, pre_norm=True, auto_pad=False):
        super().__init__()
        qk_dim =  dim
        # modules
        #self.pos_embed = nn.Conv2d(dim, dim, kernel_size=before_attn_dwconv, padding=1, groups=dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        if topk > 0:
            self.attn = attn(dim=dim, num_heads=num_heads, n_win=n_win, qk_dim=qk_dim,
                                        qk_scale=qk_scale, kv_per_win=kv_per_win, kv_downsample_ratio=kv_downsample_ratio,
                                        kv_downsample_kernel=kv_downsample_kernel, kv_downsample_mode=kv_downsample_mode,
                                        topk=topk, param_attention=param_attention, param_routing=param_routing,
                                        diff_routing=diff_routing, soft_routing=soft_routing, side_dwconv=side_dwconv,
                                        auto_pad=auto_pad)
        elif topk == -1:
            self.attn = attn(dim=dim)
        elif topk == -2:
            self.attn = AttentionLePE(dim=dim, side_dwconv=side_dwconv)



        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, int(mlp_ratio * dim)),
                                 DWConv(int(mlp_ratio * dim)) if mlp_dwconv else nn.Identity(),
                                 nn.GELU(),
                                 nn.Linear(int(mlp_ratio * dim), dim)
                                 )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = False

        self.pre_norm = pre_norm#Ture

    def forward(self, x):
        """
        x: NCHW tensor
        """
        # conv pos embedding
        #x = x + self.pos_embed(x)
        # permute to NHWC tensor for attention & mlp
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        # attention & mlp
        if self.pre_norm:
            x = x + self.drop_path(self.attn(self.norm1(x))) # (N, H, W, C)
            x = x + self.drop_path(self.mlp(self.norm2(x))) # (N, H, W, C)


        # permute back
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)#必须控制返回的格式一致
        return x


class BasicLayer_encoder(nn.Module):
    def __init__(self,depth = [2, 2, 2, 2], embed_dim = [64, 128, 320, 512], in_chans = 3,num_classes = 1000,
                 head_dim=32, qk_scale=None, representation_size=None,
                 drop_path_rate=0., drop_rate=0.,
                 use_checkpoint_stages=[],
                 ########
                 n_win=7,
                 kv_downsample_mode='identity',
                 kv_per_wins=[-1, -1, -1, -1],
                 side_dwconv=5,
                 layer_scale_init_value=-1,
                 qk_dims=[64, 128, 256, 512],
                 param_routing=False, diff_routing=False, soft_routing=False,
                 pre_norm=True,
                 before_attn_dwconv=3,
                 auto_pad=False,
                 #-----------------------
                 kv_downsample_kernels=[4, 2, 1, 1],
                 kv_downsample_ratios=[4, 2, 1, 1], # -> kv_per_win = [2, 2, 2, 1]
                 mlp_ratios=[3, 3, 3, 3],
                 param_attention='qkvo',
                 mlp_dwconv=False):
        super().__init__()
        depth = [2, 2, 2, 2]
        embed_dim = [64, 128, 320, 512]
        in_chans = 3
        num_classes = 1000
        topks = [1, 4, 16, -2]
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        ############ downsample layers (patch embeddings) ######################
        self.downsample_layers = nn.ModuleList()
        #stem first step is patch Enbedding
        stem = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim[0] // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(embed_dim[0] // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim[0] // 2, embed_dim[0], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(embed_dim[0]),
        )
        self.downsample_layers.append(stem)
        ########################################################################
        #patch Merging
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.Conv2d(embed_dim[i], embed_dim[i+1], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.BatchNorm2d(embed_dim[i+1])
            )
            self.downsample_layers.append(downsample_layer)
        ##########################################################################
        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        nheads= [dim // head_dim for dim in qk_dims]
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, 8)]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[BiFormerBlock(dim=embed_dim[i], drop_path=int(dp_rates[cur + j]),
                        layer_scale_init_value=layer_scale_init_value,
                        topk=int(topks[i]),
                        num_heads=nheads[i],
                        n_win=n_win,
                        qk_dim=qk_dims[i],
                        kv_per_win=kv_per_wins[i],
                        kv_downsample_ratio=kv_downsample_ratios[i],
                        kv_downsample_kernel=kv_downsample_kernels[i],
                        kv_downsample_mode=kv_downsample_mode,
                        param_attention=param_attention,
                        param_routing=param_routing,
                        diff_routing=diff_routing,
                        soft_routing=soft_routing,
                        mlp_ratio=mlp_ratios[i],
                        mlp_dwconv=mlp_dwconv,
                        side_dwconv=side_dwconv,
                        before_attn_dwconv=before_attn_dwconv,
                        pre_norm=pre_norm,
                        auto_pad=auto_pad) for j in range(depth[i])
            ],
            )
            self.stages.append(stage)
            cur += depth[i]# inflluence dropout

        ##########################################################################
        self.norm = nn.BatchNorm2d(embed_dim[-1])
        # Representation layer
        self.pre_logits = nn.Identity()

        #  final Classifier head
        self.head = nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, x):
        skip_connection = []
        for i in range(4):
            if i !=3:
               x = self.downsample_layers[i](x)
               skip_connection.append(x)
               x = self.stages[i](x)
            else:
                x = self.downsample_layers[i](x)
                x = self.stages[i](x)
        x = self.norm(x)
        #x = self.pre_logits(x)#
        return x, skip_connection



class BasicLayer_decoder(nn.Module):
    def __init__(self, depth=[2, 2, 2], in_chans=3, num_classes=1000, embed_dim=[320, 128, 64],
                 head_dim=32, qk_scale=None, representation_size=None,
                 drop_path_rate=0., drop_rate=0.,
                 n_win=7,
                 kv_downsample_mode='identity',
                 kv_per_wins=[ -1, -1, -1],
                 side_dwconv=5,
                 layer_scale_init_value=-1,
                 param_routing=False, diff_routing=False, soft_routing=False,
                 pre_norm=True,
                 before_attn_dwconv=3,
                 auto_pad=False,
                 #-----------------------
                 kv_downsample_kernels=[4, 2, 1, 1],
                 kv_downsample_ratios=[4, 2, 1, 1], # -> kv_per_win = [2, 2, 2, 1]
                 mlp_ratios=[4, 4, 4, 4],
                 param_attention='qkvo',
                 mlp_dwconv=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        depth = [2, 2, 2]
        qk_dims = [256, 128, 64]
        embed_dim = [512,320, 128,64]
        num_classes = 2
        topks = [16, 4, 1]
        #############################################################################
        self.upsample_layers = nn.ModuleList()
        #patch Merging
        for i in range(3):
            upsample_layer = nn.Sequential(
                nn.ConvTranspose2d(embed_dim[i], embed_dim[i+1], kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
                nn.BatchNorm2d(embed_dim[i+1])
            )
            self.upsample_layers.append(upsample_layer)
        print(upsample_layer)
        ##########################################################################
        self.concat_back_dim = nn.ModuleList()
        skip_dim = [640,320,256,128,128,64]
        j = 0
        for i in range(3):
            back_skip_connection = nn.Linear(skip_dim[j],skip_dim[j+1])
            j = j+2
            self.concat_back_dim.append(back_skip_connection)

         ## #############################################################################
        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        nheads= [dim // head_dim for dim in qk_dims]
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate,6)]
        cur = 0
        for i in range(3):
            stage = nn.Sequential(
                *[BiFormerBlock(dim=embed_dim[i+1], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                        topk=topks[i],
                        num_heads=nheads[i],
                        n_win=n_win,
                        qk_dim=qk_dims[i],
                        qk_scale=qk_scale,
                        kv_per_win=kv_per_wins[i],
                        kv_downsample_ratio=kv_downsample_ratios[i],
                        kv_downsample_kernel=kv_downsample_kernels[i],
                        kv_downsample_mode=kv_downsample_mode,
                        param_attention=param_attention,
                        param_routing=param_routing,
                        diff_routing=diff_routing,
                        soft_routing=soft_routing,
                        mlp_ratio=mlp_ratios[i],
                        mlp_dwconv=mlp_dwconv,
                        side_dwconv=side_dwconv,
                        before_attn_dwconv=before_attn_dwconv,
                        pre_norm=pre_norm,
                        auto_pad=auto_pad) for j in range(depth[i])],
            )
            self.stages.append(stage)
            cur += depth[i]# influence dropout
        print(self.stages)
        ##########################################################################
        embed_dim = [512, 320, 128, 64]
        self.linears = nn.ModuleList([nn.Linear(embed_dim[i + 1], embed_dim[i + 1]) for i in range(3)])
        self.norm = nn.BatchNorm2d(embed_dim[-1])
        # Representation layer
        self.pre_logits = nn.Identity()

        #  final Classifier head
        self.head = nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)
        self.classification_output = nn.Conv2d(in_channels=embed_dim[-1], out_channels=self.num_classes, kernel_size=1,
                                          bias=False)



    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()



#skip connection link
    def forward(self, x):
        evice = torch.device('cuda:0')
        endcoder_instance = BasicLayer_encoder().to(evice)
        x = x.to(evice)
        x, skip_connection =endcoder_instance(x)
        for i in range(3):
            x = self.upsample_layers[i](x)# res = (56, 28, 14, 7), wins = (64, 16, 4, 1)
            x = x.permute(0, 2, 3, 1).contiguous()
            print(x.size())
            skip_connection[2 - i] = skip_connection[2 - i].permute(0, 2, 3, 1).contiguous()
            x = torch.cat([x, skip_connection[2 - i]], -1)
            x = self.concat_back_dim[i](x)
            x = x.permute(0, 3, 1, 2).contiguous()
            skip_connection[2 - i] = skip_connection[2 - i].permute(0, 2, 3, 1).contiguous()
            x = self.stages[i](x)

        x = self.norm(x)

        logits = self.classification_output(x)
        return x