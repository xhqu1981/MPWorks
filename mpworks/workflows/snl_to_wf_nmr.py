from collections import defaultdict

from fireworks import Firework
from fireworks.core.firework import Tracker
from fireworks.utilities.fw_utilities import get_slug
from pymatgen import Composition

from mpworks.firetasks.custodian_task import get_custodian_task
from mpworks.firetasks.nmr_tasks import snl_to_nmr_spec
from mpworks.firetasks.snl_tasks import AddSNLTask
from mpworks.firetasks.vasp_io_tasks import VaspWriterTask, VaspCopyTask, VaspToDBTask
from mpworks.snl_utils.mpsnl import MPStructureNL
from mpworks.workflows.wf_settings import QA_DB, QA_VASP

__author__ = 'Xiaohui Qu'
__copyright__ = 'Copyright 2016, The Materials Project'
__version__ = '0.1'
__maintainer__ = 'Xiaohui Qu'
__email__ = 'xhqu1981@gmail.com'
__date__ = 'May 31, 2016'


"""
This is modified from Wei Chen's snl_to_wf_elastic.
"""


def snl_to_wf_elastic(snl, parameters):
    # parameters["user_vasp_settings"] specifies user defined incar/kpoints parameters
    fws = []
    connections = defaultdict(list)
    cur_fwid = 0
    parameters = parameters if parameters else {}

    snl_priority = parameters.get('priority', 1)
    priority = snl_priority * 2  # once we start a job, keep going!

    f = Composition(snl.structure.composition.reduced_formula).alphabetical_formula
    nick_name = parameters.get("nick_name", f)

    # add the SNL to the SNL DB and figure out duplicate group
    tasks = [AddSNLTask()]
    spec = {'task_type': 'Add to SNL database', 'snl': snl.as_dict(), 
            '_queueadapter': QA_DB, '_priority': snl_priority}
    if 'snlgroup_id' in parameters and isinstance(snl, MPStructureNL):
        spec['force_mpsnl'] = snl.as_dict()
        spec['force_snlgroup_id'] = parameters['snlgroup_id']
        del spec['snl']
    addsnl_fwid = cur_fwid
    cur_fwid += 1
    fws.append(Firework(tasks, spec, 
                        name=get_slug(nick_name + '--' + spec['task_type']), fw_id=cur_fwid))

    parameters["exact_structure"] = True
    # run Triple Jump Structure Relaxation to Converge to a Very Small Force
    geom_fwid = None
    db_fwid = None
    for istep in [1, 2, 3]:
        spec = snl_to_nmr_spec(snl, istep, parameters)
        trackers = [Tracker('FW_job.out'), Tracker('FW_job.error'), Tracker('vasp.out'), Tracker('OUTCAR'),
                    Tracker('OSZICAR'), Tracker('OUTCAR.relax1'), Tracker('OUTCAR.relax2')]
        trackers_db = [Tracker('FW_job.out'), Tracker('FW_job.error')]
        # run GGA structure optimization
        spec['_priority'] = priority
        spec['_queueadapter'] = QA_VASP
        spec['_trackers'] = trackers
        tasks = [VaspWriterTask()]
        if istep >= 2:
            parameters["use_CONTCAR"] = True
            parameters["files"] = "CONTCAR"
            parameters["keep_velocities"] = False
            tasks.append(VaspCopyTask(parameters=parameters))
        tasks.append(get_custodian_task(spec))
        geom_fwid = cur_fwid
        cur_fwid += 1
        fws.append(Firework(tasks, spec, name=get_slug(f + '--' + spec['task_type']),
                            fw_id=geom_fwid))
        geom_task_type = spec['task_type']

        # insert into DB - GGA structure optimization
        spec = {'task_type': 'VASP db insertion', '_priority': priority * 2,
                '_allow_fizzled_parents': True, '_queueadapter': QA_DB, "_dupefinder": DupeFinderDB().to_dict(),
                '_trackers': trackers_db}
        db_fwid = cur_fwid
        cur_fwid += 1
        fws.append(
            Firework([VaspToDBTask()], spec, name=get_slug(f + '--' + spec['task_type']
                                                           + '--' + geom_task_type),
                     fw_id=db_fwid))
        connections[geom_fwid] = [db_fwid]
        if istep == 1:
            connections[addsnl_fwid] = [geom_fwid]

    spec = {'task_type': 'Setup Deformed Struct Task', '_priority': priority,
                '_queueadapter': QA_CONTROL}
    fws.append(Firework([SetupDeformedStructTask()], spec, 
                        name=get_slug(f + '--' + spec['task_type']),fw_id=3))
    connections[2] = [3]

    wf_meta = get_meta_from_structure(snl.structure)
    wf_meta['run_version'] = 'May 2013 (1)'

    if '_materialsproject' in snl.data and 'submission_id' in snl.data['_materialsproject']:
        wf_meta['submission_id'] = snl.data['_materialsproject']['submission_id']

    return Workflow(fws, connections, name=Composition(
        snl.structure.composition.reduced_formula).alphabetical_formula, metadata=wf_meta)
