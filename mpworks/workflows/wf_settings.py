from monty.design_patterns import singleton

__author__ = 'Anubhav Jain'
__copyright__ = 'Copyright 2013, The Materials Project'
__version__ = '0.1'
__maintainer__ = 'Anubhav Jain'
__email__ = 'ajain@lbl.gov'
__date__ = 'May 10, 2013'

# don't change nnodes unless other parts of code are also changed
# nodes configuration will be ignored on SLURM due to different naming convention (nnodes vs nodes)
QA_VASP = {'nnodes': 2, 'nodes': 2, 'walltime': '48:00:00'}
QA_VASP_SMALL = {'nnodes': 2, 'nodes': 2, 'walltime': '48:00:00'}  # small walltime jobs
QA_DB = {'nnodes': 1, 'nodes' : 1, 'walltime': '2:00:00'}
QA_CONTROL = {'nnodes': 1, 'nodes': 1, 'walltime': '00:30:00'}

@singleton
class WFSettings(object):
    """
    This class stores settings for the Workflows. Use Singleton to enable runtime dynamic chage
    """

    def __init__(self):
        self.MOVE_TO_GARDEN_DEV = False
        self.MOVE_TO_GARDEN_PROD = False
        self.GARDEN = '/project/projectdirs/matgen/garden'

    @property
    def RUN_LOCS(self):
        return [self.GARDEN, self.GARDEN + '/dev',
                '/project/projectdirs/matgen/garden/control_blocks',
                '/project/projectdirs/matgen/scratch',
                '/global/scratch/sd/matcomp/', '/global/homes/m/matcomp',
                '/scratch/scratchdirs/matcomp/', '/scratch2/scratchdirs/matcomp/',
                '/global/scratch/sd/matcomp/aj_tests/',
                '/global/scratch/sd/matcomp/wc_tests/',
                '/global/scratch/sd/matcomp/aj_prod/',
                '/global/scratch2/sd/matcomp/mp_prod/',
                '/global/scratch2/sd/matcomp/mp_prod_hopper/']

