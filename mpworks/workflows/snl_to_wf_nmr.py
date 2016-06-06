import copy
from collections import defaultdict

from fireworks import Firework
from fireworks.core.firework import Tracker, Workflow
from fireworks.utilities.fw_utilities import get_slug
from pymatgen import Composition

from mpworks.dupefinders.dupefinder_vasp import DupeFinderDB
from mpworks.firetasks.custodian_task import get_custodian_task
from mpworks.firetasks.nmr_tasks import snl_to_nmr_spec, NmrVaspToDBTask, DictVaspSetupTask
from mpworks.firetasks.snl_tasks import AddSNLTask
from mpworks.firetasks.vasp_io_tasks import VaspWriterTask, VaspCopyTask, VaspToDBTask
from mpworks.snl_utils.mpsnl import MPStructureNL, get_meta_from_structure
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


def get_nmr_vasp_fw(fwid, copy_contcar, istep, nick_name, parameters, priority, structure, additional_run_tags):
    spec = snl_to_nmr_spec(structure, istep, parameters, additional_run_tags)
    trackers = [Tracker('FW_job.out'), Tracker('FW_job.error'), Tracker('vasp.out'), Tracker('OUTCAR'),
                Tracker('OSZICAR'), Tracker('OUTCAR.relax1'), Tracker('OUTCAR.relax2')]
    spec['_priority'] = priority
    spec['_queueadapter'] = QA_VASP
    spec['_trackers'] = trackers
    tasks = [DictVaspSetupTask(), get_custodian_task(spec)]
    vasp_fw = Firework(tasks, spec, name=get_slug(nick_name + '--' + spec['task_type']),
                       fw_id=fwid)
    return vasp_fw


def get_nmr_db_fw(nick_name, fwid, prev_task_type, priority, task_class):
    trackers_db = [Tracker('FW_job.out'), Tracker('FW_job.error')]
    spec = {'task_type': 'VASP db insertion', '_priority': priority * 2,
            '_allow_fizzled_parents': True, '_queueadapter': QA_DB, "_dupefinder": DupeFinderDB().to_dict(),
            '_trackers': trackers_db}
    db_fw = Firework([task_class(parameters={"update_input": False})], spec, name=get_slug(nick_name + '--' + spec['task_type'] +
                                                         '--' + prev_task_type),
                     fw_id=fwid)
    return db_fw


def snl_to_wf_nmr(snl, parameters):
    # parameters["user_vasp_settings"] specifies user defined incar/kpoints parameters
    fws = []
    connections = defaultdict(list)
    cur_fwid = 0
    parameters = parameters if parameters else {}

    snl_priority = parameters.get('priority', 1)
    priority = snl_priority * 2  # once we start a job, keep going!

    f = Composition(snl.structure.composition.reduced_formula).alphabetical_formula
    nick_name = parameters.get("nick_name", f)

    if 'exact_structure' in parameters and parameters['exact_structure']:
        structure = snl.structure
    else:
        structure = snl.structure.get_primitive_structure()

    additional_run_tags = []
    # add exact structure run tag automatically if we have a unique situation
    if 'exact_structure' in parameters and parameters['exact_structure'] and \
            snl.structure != structure:
        additional_run_tags.append('exact_structure')

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
                        name=get_slug(nick_name + '--' + spec['task_type']), fw_id=addsnl_fwid))

    parameters["exact_structure"] = True
    # run Triple Jump Structure Relaxation to Converge to a Very Small Force
    geom_calc_fwid = None
    geom_db_fwid = None
    for istep in [1, 2, 3]:
        # Geometry Optimization
        copy_contcar = istep >= 2
        geom_calc_fwid = cur_fwid
        cur_fwid += 1
        vasp_fw = get_nmr_vasp_fw(geom_calc_fwid, copy_contcar, istep, nick_name,
                                  copy.deepcopy(parameters), priority, copy.deepcopy(structure),
                                  additional_run_tags)
        fws.append(vasp_fw)
        geom_task_type = vasp_fw.spec['task_type']
        if istep == 1:
            connections[addsnl_fwid] = [geom_calc_fwid]
        else:
            prev_db_fwid = geom_db_fwid
            connections[prev_db_fwid] = [geom_calc_fwid]

        # insert into DB
        task_class = VaspToDBTask
        prev_task_type = geom_task_type
        geom_db_fwid = cur_fwid
        cur_fwid += 1
        db_fw = get_nmr_db_fw(nick_name, geom_db_fwid, prev_task_type, priority, task_class)
        fws.append(db_fw)
        connections[geom_calc_fwid] = [geom_db_fwid]

    # Calculate NMR Tensors
    for istep in [-1, -2]:
        # -1: Chemical Shift, -2: EFG
        # Geometry Optimization
        nmr_calc_fwid = cur_fwid
        cur_fwid += 1
        vasp_fw = get_nmr_vasp_fw(nmr_calc_fwid, True, istep, nick_name, copy.deepcopy(parameters),
                                  priority, copy.deepcopy(structure), additional_run_tags)
        fws.append(vasp_fw)
        nmr_task_type = vasp_fw.spec['task_type']
        connections[geom_calc_fwid].extend([nmr_calc_fwid])

        # insert into DB
        task_class = NmrVaspToDBTask
        prev_task_type = nmr_task_type
        nmr_db_fwid = cur_fwid
        cur_fwid += 1
        db_fw = get_nmr_db_fw(nick_name, nmr_db_fwid, prev_task_type, priority, task_class)
        fws.append(db_fw)
        connections[nmr_calc_fwid] = [nmr_db_fwid]

    wf_meta = get_meta_from_structure(snl.structure)
    wf_meta['run_version'] = 'June 2016 (1)'

    if '_materialsproject' in snl.data and 'submission_id' in snl.data['_materialsproject']:
        wf_meta['submission_id'] = snl.data['_materialsproject']['submission_id']

    return Workflow(fws, connections, name=nick_name, metadata=wf_meta)
