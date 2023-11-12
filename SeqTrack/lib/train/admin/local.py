class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/darcy/PycharmProjects/VideoX/SeqTrack'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/home/darcy/PycharmProjects/VideoX/SeqTrack/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/home/darcy/PycharmProjects/VideoX/SeqTrack/pretrained_networks'
        self.lasot_dir = '/home/darcy/PycharmProjects/VideoX/SeqTrack/data/lasot'
        self.got10k_dir = '/home/darcy/PycharmProjects/VideoX/SeqTrack/data/got10k'
        self.lasot_lmdb_dir = '/home/darcy/PycharmProjects/VideoX/SeqTrack/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/home/darcy/PycharmProjects/VideoX/SeqTrack/data/got10k_lmdb'
        self.trackingnet_dir = '/home/darcy/PycharmProjects/VideoX/SeqTrack/data/trackingnet'
        self.trackingnet_lmdb_dir = '/home/darcy/PycharmProjects/VideoX/SeqTrack/data/trackingnet_lmdb'
        self.coco_dir = '/home/darcy/PycharmProjects/VideoX/SeqTrack/data/coco'
        self.coco_lmdb_dir = '/home/darcy/PycharmProjects/VideoX/SeqTrack/data/coco_lmdb'
        self.imagenet1k_dir = '/home/darcy/PycharmProjects/VideoX/SeqTrack/data/imagenet1k'
        self.imagenet22k_dir = '/home/darcy/PycharmProjects/VideoX/SeqTrack/data/imagenet22k'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/home/darcy/PycharmProjects/VideoX/SeqTrack/data/vid'
        self.imagenet_lmdb_dir = '/home/darcy/PycharmProjects/VideoX/SeqTrack/data/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
