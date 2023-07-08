from dataclasses import dataclass


@dataclass
class CosyposeCfg:
    """Configuration for self-supervised experiment"""

    ## machine parameters
    #n_cpus: int = int(os.environ.get("N_CPUS", 10))
    n_workers: int = 4

    ## directories
    # project_root = Path(__file__).parent.parent
    # data_dir = project_root / 'data'
    # bop_data_dir = data_dir / 'bop'

    ## dataset parameters
    # dataset = 'ycbv'
    n_symmetries_batch: int = 64

    ## Training
    # batch_size = 32
    # epoch_size = 115200
    # n_epochs = 700
    # n_dataloader_workers = n_workers

    # cosypose data parameters
    object_set: str = None
    urdf_ds_name: str = None
    #cosypose_pretrained_weights_path: str = None
    #cosypose_model_save_dir: str = None

    # cosypose model parameters
    #backbone_str: str = "efficientnet-b3"
    #n_pose_dims: int = 9
    #n_rendering_workers: int = n_workers

    # cosypose method
    #loss_disentangled: bool = True
    #n_points_loss: int = 2600
    #TCO_input_generator: str = "fixed"
    #n_iterations: int = 1
    #min_area: float = None

    #debug : float = False

    def load_from_dict(self, cfg_dict):
        """load parameters from dictionary"""
        self.n_workers = cfg_dict['training']['n_workers']
        self.min_area = cfg_dict['training']['min_area']

        # cosypose data parameters
        self.object_set = cfg_dict['dataset']
        self.urdf_ds_name = cfg_dict['urdf_ds_name']

        ## cosypose model parameters
        #cosypose_cfg = cfg_dict['cosypose_coarse_refine']
        #self.backbone_str = cosypose_cfg['backbone_str']
        #self.n_pose_dims = cosypose_cfg['n_pose_dims']
        #self.n_rendering_workers = cosypose_cfg['n_rendering_workers']

        ## cosypose method
        #self.loss_disentangled = cosypose_cfg['loss_disentangled']
        #self.n_points_loss = cosypose_cfg['n_points_loss']
        #self.n_iterations = cosypose_cfg['n_iterations']
        #self.debug = cosypose_cfg['debug']
