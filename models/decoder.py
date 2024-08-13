import torch
from torch import nn
from models.pos_encoding import get_encoder
import tinycudann as tcnn


class Decoder(nn.Module):
    def __init__(self, cfg, bound, n_class=40):
        super().__init__()
        self.pe_fn = Pos_Encoding(cfg, bound)
        self.pe_dim = self.pe_fn.pe_dim
        self.grid_dim = self.pe_fn.grid_dim

        self.pts_dim = cfg['pts_dim']
        self.hidden_dim = cfg['hidden_dim']
        self.pixel_dim = cfg['pixel_dim']
        self.n_class = n_class

        self.coarse_fn = Coarse(pts_dim=self.pe_dim, 
                                hidden_dim=self.hidden_dim, 
                                feature_dim=self.grid_dim)
        
        self.out_fn = Out(pts_dim=self.pe_dim, feature_dim=self.hidden_dim*2, 
                            hidden_dim=self.hidden_dim, 
                            n_class=self.n_class)

        self.merge = Merge(cfg, hidden_dim=self.hidden_dim, feature_dim=self.pixel_dim, bound=bound)
        
      
class Pos_Encoding(nn.Module):
    def __init__(self, cfg, bound):
        super().__init__()
        # Coordinate encoding
        self.pe_fn, self.pe_dim = get_encoder(cfg['pos']['method'], 
                                              n_bins=cfg['pos']['n_bins'])

        # Sparse parametric grid encoding
        dim_max = (bound[:,1] - bound[:,0]).max()
        self.resolution = int(dim_max / cfg['grid']['voxel_size'])
        self.grid_fn, self.grid_dim = get_encoder(cfg['grid']['method'], 
                                                  log2_hashmap_size=cfg['grid']['hash_size'], 
                                                  desired_resolution=self.resolution)
        print('Grid size:', self.grid_dim)
    
    def forward(self, pts):
        pe = self.pe_fn(pts)
        grid = self.grid_fn(pts)
        return pe, grid
    

class Merge(nn.Module):
    def __init__(self, cfg, hidden_dim=32, feature_dim=64, bound=None):
        super().__init__()
        self.bound = bound
        self.pe_fn, self.pe_dim = get_encoder(cfg['pos']['method'], 
                                              n_bins=cfg['pos']['n_bins'])

        self.decoder = tcnn.Network(n_input_dims=self.pe_dim+feature_dim,
                                    n_output_dims=hidden_dim,
                                    network_config={
                                        "otype": "CutlassMLP", # FullyFusedMLP CutlassMLP 
                                        "activation": "ReLU",
                                        "output_activation": "None",
                                        "n_neurons": hidden_dim,
                                        "n_hidden_layers": 1})

    def forward(self, p, o, features=None):
        n_refer, n_points, C = features.shape

        p = (p - self.bound[:, 0]) / (self.bound[:, 1] - self.bound[:, 0])
        pe = self.pe_fn(p.flatten(0, 1))
        pe = pe.reshape(n_refer, n_points, -1)
        # pe = pe.reshape(n_refer, 1, -1).repeat(1, n_points, 1)

        latents = self.decoder(torch.cat((pe.flatten(0, 1), features.flatten(0, 1)), -1))
        latents = torch.mean(latents.reshape(n_refer, n_points, -1), 0)
        return latents
    

class Coarse(nn.Module):
    def __init__(self, pts_dim, hidden_dim, feature_dim):
        super().__init__()

        self.decoder = tcnn.Network(n_input_dims=pts_dim+feature_dim,
                                    n_output_dims=hidden_dim+1,
                                    network_config={
                                        "otype": "CutlassMLP",
                                        "activation": "ReLU",
                                        "output_activation": "None",
                                        "n_neurons": hidden_dim,
                                        "n_hidden_layers": 1})
        
    def forward(self, pe, features=None):
        return self.decoder(torch.cat((pe, features), -1)).float()
    

class Out(nn.Module):
    def __init__(self, pts_dim, feature_dim, hidden_dim, n_class):
        super().__init__()

        self.color_decoder = tcnn.Network(n_input_dims=pts_dim+feature_dim,
                                          n_output_dims=3,
                                          network_config={
                                            "otype": "CutlassMLP",
                                            "activation": "ReLU",
                                            "output_activation": "None",
                                            "n_neurons": hidden_dim,
                                            "n_hidden_layers": 1})
        
        self.logit_decoder = tcnn.Network(n_input_dims=pts_dim+feature_dim,
                                          n_output_dims=n_class,
                                          network_config={
                                            "otype": "CutlassMLP",
                                            "activation": "ReLU",
                                            "output_activation": "None",
                                            "n_neurons": hidden_dim,
                                            "n_hidden_layers": 1})
        
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, pe, features):
        color = self.sigmoid(self.color_decoder(torch.cat((pe, features), -1)))
        logit = self.logit_decoder(torch.cat((pe, features), -1))
        return color, logit
    