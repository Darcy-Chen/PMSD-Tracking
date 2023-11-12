from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/darcy/PycharmProjects/VideoX/SeqTrack/data/got10k_lmdb'
    settings.got10k_path = '/home/darcy/PycharmProjects/VideoX/SeqTrack/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_extension_subset_path = '/home/darcy/PycharmProjects/VideoX/SeqTrack/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/home/darcy/PycharmProjects/VideoX/SeqTrack/data/lasot_lmdb'
    settings.lasot_path = '/home/darcy/PycharmProjects/VideoX/SeqTrack/data/lasot'
    settings.network_path = '/home/darcy/PycharmProjects/VideoX/SeqTrack/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/darcy/PycharmProjects/VideoX/SeqTrack/data/nfs'
    settings.otb_path = '/home/darcy/PycharmProjects/VideoX/SeqTrack/data/OTB2015'
    settings.prj_dir = '/home/darcy/PycharmProjects/VideoX/SeqTrack'
    settings.result_plot_path = '/home/darcy/PycharmProjects/VideoX/SeqTrack/test/result_plots'
    settings.results_path = '/home/darcy/PycharmProjects/VideoX/SeqTrack/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/darcy/PycharmProjects/VideoX/SeqTrack'
    settings.segmentation_path = '/home/darcy/PycharmProjects/VideoX/SeqTrack/test/segmentation_results'
    settings.tc128_path = '/home/darcy/PycharmProjects/VideoX/SeqTrack/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/darcy/PycharmProjects/VideoX/SeqTrack/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/darcy/PycharmProjects/VideoX/SeqTrack/data/trackingnet'
    settings.uav_path = '/home/darcy/PycharmProjects/VideoX/SeqTrack/data/UAV123'
    settings.vot_path = '/home/darcy/PycharmProjects/VideoX/SeqTrack/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

