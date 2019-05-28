class Config(object):
    
    def __init__(self):

        # directories
        self.save_dir = 'saved_models/'  # Use to write Neural-Net check-points etc.
        self.top_in_dir = 'data/shape_net_core_uniform_samples_2048/' # Top-dir of where point-clouds are stored.
        self.log_dir = 'log_dir/'
        self.render_dir = 'renders/'

        # input and layer params
        self.z_size = 128
        self.N = [128, 128, 256, 512, 1024, 2048, 2048]
        self.gen_FC_layers = [self.z_size, self.N[0]*96]
        self.gen_conv_layers = [96, 48, 32, 16, 8, 3, 3]
        self.gen_fnet_layers = [[96, 1024, 96*48],[48, 768, 48*32],[32, 256, 32*16],[16, 64, 16*8],[8, 16, 8*3],[3, 6, 3*3]]  
        self.dis_SFC_layers = [3, 64, 128, 256, 512]
        self.dis_FC_layers = [512, 128, 64, 1]
        self.dis_convkernel_width = 1
        self.signal_size = [2048, 3]   
        self.radius = 0.025 # max distance between neighbours
        self.min_nn = [15, 15, 15, 15, 15, 15] # min no. of neighbours (<)
        self.upsampling = [False] + [True]*4 + [False]
        self.upsamp_method = 'aggr' # 'aggr', 'gauss', 'triangle', 'perturb'
        self.upsamp_aggr_method = 'diag' # 'full', 'diag', 'scalar'
        self.upsamp_aggr_nn = 15
        self.sigma_init = 0.1 # for perturb method

        # learning
        self.batch_size = 50
        self.N_iter = 100000
        self.learning_rate_gen = 1e-4
        self.learning_rate_dis = 1e-4
        self.beta = 0.5
        self.dis_clip = (-0.01, 0.01)
        self.wgan = True
        self.dis_n_iter = 5
        self.scale = 1

        # debugging
        self.save_every_iter = 1000
        self.summaries_every_iter = 5
        self.renders_every_iter = 100
        self.N_test = 1

