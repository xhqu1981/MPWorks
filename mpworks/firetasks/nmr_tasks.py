import copy
import hashlib
import json
import os

import shutil
import yaml
from fireworks import FireTaskBase
from fireworks.core.firework import FWAction, Firework, Workflow
from fireworks.utilities.fw_serializers import FWSerializable
from fireworks.utilities.fw_utilities import get_slug
from monty.os.path import zpath
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.io.vasp import Outcar, Kpoints
from monty.io import zopen
import re
import math
from pymatgen.analysis.nmr import NMRChemicalShiftNotation
from pymatgen.io.vasp.sets import DictSet

from mpworks.dupefinders.dupefinder_vasp import DupeFinderVasp
from mpworks.firetasks.vasp_io_tasks import VaspToDBTask, VaspCopyTask
from mpworks.firetasks.vasp_setup_tasks import SetupUnconvergedHandlerTask
from mpworks.workflows.wf_settings import WFSettings
from mpworks.workflows.wf_utils import get_loc

__author__ = 'Xiaohui Qu'
__copyright__ = 'Copyright 2016, The Materials Project'
__version__ = '0.1'
__maintainer__ = 'Xiaohui Qu'
__email__ = 'xhqu1981@gmail.com'
__date__ = 'May 31, 2016'


"""
This is modified from Wei Chen & Joseph Montoya's elastic_tasks.
"""


def _get_nuclear_quadrupole_moment(element, nqm_dict, parameters):
    if element not in nqm_dict:
        return 0.0
    d = nqm_dict[element]
    if len(d) > 1:
        prefered_isotopes = set(parameters.get("isotopes", []))
        pi = prefered_isotopes & set(list(d.keys()))
        if len(pi) == 1:
            return d[list(pi)[0]]
        if len(pi) >= 1:
            raise ValueError("Multiple isotope is requested \"{}\", "
                             "please request only one for each elements".format(list(pi)))
        isotopes = list(d.keys())
        isotopes.sort(key=lambda x: int(x.split("-")[1]), reverse=False)
        return d[isotopes[0]]
    else:
        return list(d.values())[0]


def _config_dict_to_input_set(config_dict, structure, incar_enforce, parameters):
    functional = parameters.get("functional", "PBE")
    pot_map = {"PBE": "PBE", "SCAN": "PBE_52"}
    potcar_functional = pot_map[functional]
    trial_set = DictSet(structure, config_dict=config_dict,
                        user_incar_settings=incar_enforce,
                        potcar_functional=potcar_functional)
    trial_potcar = trial_set.potcar
    all_enmax = [sp.enmax for sp in trial_potcar]
    all_eaug = [sp.eaug for sp in trial_potcar]
    all_elements = [sp.element for sp in trial_potcar]
    num_species = len(all_enmax)
    processed_config_dict = copy.deepcopy(config_dict)
    for k1, pot_values in [("ENCUT", all_enmax), ("ENAUG", all_eaug)]:
        k2 = "{}_ENHANCE_RATIO".format(k1)
        if k2 in config_dict["INCAR"]:
            ratio = 1.0 + config_dict["INCAR"][k2]
            processed_config_dict["INCAR"].pop(k2)
            processed_config_dict["INCAR"][k1] = round(ratio * max(pot_values), 0)
    if "ROPT_PER_ATOM" in config_dict["INCAR"]:
        processed_config_dict["INCAR"].pop("ROPT_PER_ATOM")
        processed_config_dict["INCAR"]["ROPT"] = \
            [config_dict["INCAR"]["ROPT_PER_ATOM"]] * num_species
    if "QUAD_EFG_MAP" in config_dict["INCAR"]:
        nqm_map = processed_config_dict["INCAR"].pop("QUAD_EFG_MAP")
        quad_efg = [_get_nuclear_quadrupole_moment(el, nqm_map, parameters) for el in all_elements]
        processed_config_dict["INCAR"]["QUAD_EFG"] = quad_efg
    vis = DictSet(structure, config_dict=processed_config_dict,
                  user_incar_settings=incar_enforce,
                  potcar_functional=potcar_functional)
    return vis


def _change_garden_setting():
    db_dir = os.environ['DB_LOC']
    db_path = os.path.join(db_dir, 'tasks_db.json')
    with open(db_path) as f:
        db_creds = json.load(f)
        if 'prod' in db_creds['database']:
            WFSettings().MOVE_TO_GARDEN_PROD = True
        elif 'test' in db_creds['database']:
            WFSettings().MOVE_TO_GARDEN_DEV = True
    if 'nmr' not in WFSettings().GARDEN:
        WFSettings().GARDEN = os.path.join(WFSettings().GARDEN, 'nmr')


def _assign_potcar_valence(structure, potcar_dict):
    tri_val_elements = {"Ce", "Dy", "Er", "Eu", "Gd", "Ho", "Lu", "Nd", "Pm", "Pr", "Sm", "Tb", "Tm"}
    di_val_elements = {"Er", "Eu", "Yb"}
    st_elements = set([specie.symbol for specie in structure.species])
    bva = BVAnalyzer()
    valences = bva.get_valences(structure)
    for val, val_elements in [[3, tri_val_elements],
                              [2, di_val_elements]]:
        for el in sorted(val_elements & st_elements):
            if "_" in potcar_dict[el]:
                continue
            el_indices = structure.indices_from_symbol(el)
            cur_el_valences = {valences[i] for i in el_indices}
            if len(cur_el_valences) == 1 and val in cur_el_valences:
                potcar_dict[el] = "{el}_{val:d}".format(el=el, val=val)


def snl_to_nmr_spec(structure, istep_triple_jump, parameters=None, additional_run_tags=()):
    parameters = copy.deepcopy(parameters) if parameters else {}
    spec = {'parameters': parameters}

    module_dir = os.path.abspath(os.path.dirname(__file__))
    if 1 <= istep_triple_jump <= 3:
        config_file = os.path.join(module_dir, "triple_jump_relax_set.yaml")
        config_key = "STEP{}".format(istep_triple_jump)
        config_name = "Triple Jump Relax S{}".format(istep_triple_jump)
    elif istep_triple_jump == -1:
        # NMR Chemical Shit calculations
        config_file = os.path.join(module_dir, "nmr_tensor_set.yaml")
        config_key = "CS"
        config_name = "NMR CS"
    elif istep_triple_jump == -2:
        # NMR Chemical Shit calculations
        config_file = os.path.join(module_dir, "nmr_tensor_set.yaml")
        config_key = "EFG"
        config_name = "NMR EFG"
    else:
        raise ValueError("Unknow Step Index \"{}\"".format(istep_triple_jump))
    with open(config_file) as f:
        parent_config_dict = yaml.load(stream=f)
    config_dict = parent_config_dict[config_key]
    if len(structure) < 64:
        par_num = 4
    else:
        par_num = 8
    if config_name == "NMR CS":
        incar_enforce = {'KPAR': par_num}
    else:
        incar_enforce = {'NPAR': par_num}
    spec['run_tags'] = spec.get('run_tags', [])
    spec['run_tags'].extend(additional_run_tags)
    _assign_potcar_valence(structure, config_dict["POTCAR"])

    mpvis = _config_dict_to_input_set(config_dict, structure,
                                      incar_enforce, parameters=parameters)
    incar = mpvis.incar
    poscar = mpvis.poscar
    potcar = mpvis.potcar

    spec["input_set_config_dict"] = mpvis.config_dict
    spec["input_set_incar_enforce"] = incar_enforce
    spec["custodian_default_input_set"] = mpvis

    # Add run tags of pseudopotential
    spec['run_tags'] = spec.get('run_tags', [potcar.functional])
    spec['run_tags'].extend(potcar.symbols)

    # Add run tags of +U
    u_tags = ['%s=%s' % t for t in
              zip(poscar.site_symbols, incar.get('LDAUU', [0] * len(poscar.site_symbols)))]
    spec['run_tags'].extend(u_tags)

    # add user run tags
    if 'run_tags' in parameters:
        spec['run_tags'].extend(parameters['run_tags'])
        del spec['parameters']['run_tags']

    spec['_dupefinder'] = DupeFinderVasp().to_dict()
    spec['task_type'] = config_name
    spec['vaspinputset_name'] = config_name + ' DictSet'

    return spec

def chemical_shift_spec_to_dynamic_kpt_average_wfs(fw_spec):
    no_jobs_spec = copy.deepcopy(fw_spec)
    no_jobs_spec.pop('jobs', None)
    no_jobs_spec.pop('handlers', None)
    no_jobs_spec.pop('max_errors', None)
    no_jobs_spec.pop('_tasks', None)
    no_jobs_spec.pop('custodian_default_input_set', None)
    no_jobs_spec.pop('prev_task_type', None)
    no_jobs_spec.pop('prev_vasp_dir', None)
    no_jobs_spec.pop('task_type', None)
    no_jobs_spec.pop('vaspinputset_name', None)
    nick_name = no_jobs_spec['parameters']['nick_name']
    priority = no_jobs_spec['_priority']

    cur_fwid = -1
    fws = []

    # Pre Single Kpt CS SCF Task
    scf_spec = copy.deepcopy(no_jobs_spec)
    for k in ["DQ", "ICHIBARE", "LCHIMAG", "LNMR_SYM_RED", "NSLPLINE"]:
        scf_spec['input_set_config_dict']['INCAR'].pop(k, None)
    scf_spec['input_set_config_dict']['INCAR']['ISMEAR'] = 0
    scf_spec['input_set_config_dict']['INCAR']['LCHARG'] = True
    scf_spec['input_set_incar_enforce'] = {"NPAR": fw_spec['input_set_incar_enforce']["KPAR"]}
    scf_spec['task_type'] = 'Pre Kpt CS SCF'
    scf_spec['vaspinputset_name'] = scf_spec['task_type'] + " DictSet"
    scf_spec['prev_task_type'] = fw_spec['task_type']
    scf_spec['prev_vasp_dir'] = os.getcwd()
    scf_tasks = [DictVaspSetupTask()]
    functional = scf_spec["functional"]
    if functional != "PBE":
        scf_tasks.append(ScanFunctionalSetupTask())
    from mpworks.firetasks.custodian_task import get_custodian_task
    scf_tasks.append(get_custodian_task(scf_spec))
    scf_tasks.append(TagFileChecksumTask({"files": ["CHGCAR"]}))
    scf_vasp_fwid = cur_fwid
    cur_fwid -= 1
    vasp_fw = Firework(scf_tasks, scf_spec, name=get_slug(nick_name + '--' + scf_spec['task_type']),
                       fw_id=scf_vasp_fwid)
    fws.append(vasp_fw)

    scf_db_fwid = cur_fwid
    cur_fwid -= 1
    scf_db_type_class = VaspToDBTask
    from mpworks.workflows.snl_to_wf_nmr import get_nmr_db_fw
    scf_db_fw = get_nmr_db_fw(nick_name=nick_name, fwid=scf_db_fwid, prev_task_type=scf_spec['task_type'],
                          priority=priority, task_class=scf_db_type_class)
    fws.append(scf_db_fw)

    # Single Kpt CS Generation
    gen_spec = copy.deepcopy(no_jobs_spec)
    gen_spec['input_set_config_dict']['INCAR']['ISMEAR'] = 0
    gen_spec['input_set_config_dict']['INCAR']['ICHARG'] = 11
    gen_spec['task_type'] = 'Single Kpt CS Generation'
    gen_spec['vaspinputset_name'] = gen_spec['task_type'] + " DictSet"
    gen_tasks = [ChemicalShiftKptsAverageGenerationTask()]
    gen_fwid = cur_fwid
    cur_fwid -= 1
    gen_fw = Firework(gen_tasks, gen_spec,
                      name=get_slug(nick_name + '--' + gen_spec['task_type']),
                      fw_id=gen_fwid)
    fws.append(gen_fw)
    connections = {scf_vasp_fwid: scf_db_fwid,
                   scf_db_fwid: gen_fwid}
    wf = Workflow(fws, connections)
    return wf


class NmrVaspToDBTask(VaspToDBTask):
    _fw_name = "NMR Tensor to Database Task"

    def __init__(self, parameters=None):
        super(NmrVaspToDBTask, self).__init__(parameters)

    def run_task(self, fw_spec):
        _change_garden_setting()
        prev_dir = get_loc(fw_spec['prev_vasp_dir'])
        outcar = Outcar(zpath(os.path.join(prev_dir, "OUTCAR")))
        prev_task_type = fw_spec['prev_task_type']
        nmr_fields = dict()
        update_spec = None
        if prev_task_type == "NMR CS":
            outcar.read_chemical_shifts()
            cs_fiels = {"chemical_shifts": [x.as_dict() for x in outcar.data["chemical_shifts"]["valence_only"]]}
            nmr_fields.update(cs_fiels)
        elif prev_task_type == "Single Kpt CS":
            update_spec = dict()
            update_spec["total_kpts"] = fw_spec['total_kpts']
            update_spec['scf_vasp_dir'] = fw_spec['scf_vasp_dir']
            cs_kpt_name = fw_spec['kpoint_tag']
            update_spec[cs_kpt_name] = dict()
            update_spec[cs_kpt_name]["kpt_vasp_dir"] = prev_dir
            outcar.read_chemical_shifts()
            update_spec[cs_kpt_name]["chemical_shifts"] = outcar.data["chemical_shifts"]
        elif prev_task_type == 'Single Kpt CS Collect':
            for k in ['chemical_shifts', 'manual_kpt_average', 'rmsd',
                      'rmsd_header', 'manual_kpt_data']:
                nmr_fields = fw_spec[k]
            shutil.copytree(prev_dir, "fake_nmr_vasp_files")
            fake_prev_dir = os.path.abspath("fake_nmr_vasp_files")
            fw_spec['prev_vasp_dir'] = fake_prev_dir
            with zopen(zpath(os.path.join(fake_prev_dir, 'FW.json')), 'rt') as f:
                fw_dict = json.load(f)
            fw_dict["prev_task_type"] = "NMR CS"
            with zopen(zpath(os.path.join(fake_prev_dir, 'FW.json')), 'wt') as f:
                json.dump(fw_dict, f, sort_keys=True, indent=4)
        elif prev_task_type == "NMR EFG":
            outcar.read_nmr_efg()
            efg_fields = {"efg": outcar.data["efg"]}
            nmr_fields.update(efg_fields)
        else:
            raise ValueError("Unsupported Task Type: \"{}\"".format(prev_task_type))
        self.additional_fields.update(nmr_fields)
        m_action = super(NmrVaspToDBTask, self).run_task(fw_spec)
        if update_spec is not None:
            update_spec.update(m_action.update_spec)
            m_action.update_spec = update_spec
        return m_action


class TripleJumpRelaxVaspToDBTask(VaspToDBTask):
    _fw_name = "Triple Jump Relax to Database Task"

    def __init__(self, parameters=None):
        super(TripleJumpRelaxVaspToDBTask, self).__init__(parameters)

    def run_task(self, fw_spec):
        _change_garden_setting()
        return super(TripleJumpRelaxVaspToDBTask, self).run_task(fw_spec)


class SetupTripleJumpRelaxS3UnconvergedHandlerTask(SetupUnconvergedHandlerTask):
    _fw_name = "Triple Jump Relax S3 Unconverged Handler Task"

    def run_task(self, fw_spec):
        module_dir = os.path.abspath(os.path.dirname(__file__))
        config_file = os.path.join(module_dir, "triple_jump_relax_set.yaml")
        config_key = "STEP_DYNA3"
        with open(config_file) as f:
            parent_config_dict = yaml.load(stream=f)
        config_dict = parent_config_dict[config_key]
        incar_update = config_dict["INCAR"]
        actions = [{"dict": "INCAR",
                    "action": {"_set": incar_update}}]
        from custodian.vasp.interpreter import VaspModder
        VaspModder().apply_actions(actions)
        parent_action = super(SetupTripleJumpRelaxS3UnconvergedHandlerTask, self).run_task(fw_spec)
        return parent_action


class DictVaspSetupTask(FireTaskBase, FWSerializable):
    _fw_name = "Dict Vasp Input Setup Task"

    def __init__(self, parameters=None):
        parameters = parameters if parameters else {}
        default_files = ["INCAR", "POSCAR", "POTCAR", "KPOINTS"]
        self.update(parameters)
        self.files = parameters.get("files", default_files)

    @staticmethod
    def _sort_structure_by_encut(structure, config_dict):
        # put the larger ENMAX specie first
        trial_vis = DictSet(structure, config_dict=config_dict)
        trial_potcar = trial_vis.potcar
        enmax_dict = {p.symbol.split("_")[0]: p.keywords["ENMAX"] + p.keywords["ENMIN"] * 1.0E-3
                      for p in trial_potcar}
        structure = structure.get_sorted_structure(key=lambda site: enmax_dict[site.specie.symbol], reverse=True)
        return structure

    def run_task(self, fw_spec):
        config_dict = fw_spec["input_set_config_dict"]
        incar_enforce = fw_spec["input_set_incar_enforce"]
        mpsnl = fw_spec["mpsnl"]
        structure = self._sort_structure_by_encut(mpsnl.structure, config_dict)
        vis = DictSet(structure, config_dict=config_dict,
                      user_incar_settings=incar_enforce,
                      sort_structure=False)
        if "INCAR" in self.files:
            vis.incar.write_file("INCAR")
        if "POSCAR" in self.files:
            vis.poscar.write_file("POSCAR")
        if "POTCAR" in self.files:
            vis.potcar.write_file("POTCAR")
        if "KPOINTS" in self.files:
            if "kpoints_enforce" not in fw_spec:
                vis.kpoints.write_file("KPOINTS")
            else:
                fw_spec["kpoints_enforce"].write_file("KPOINTS")
        return FWAction(stored_data={"vasp_input_set": vis.as_dict()})


class ScanFunctionalSetupTask(FireTaskBase, FWSerializable):
    """
    This class is to setup the SCAN functional calculation for the
    hack version of VASP. If official SCAN functional supported VASP
    is release, the run_task() method body can be set to "pass". It
    will make the workflow compatible with the new VASP version.
    """
    _fw_name = "SCAN Functional Setup Task"

    def run_task(self, fw_spec):
        functional = fw_spec.get("functional", "PBE")
        if functional == "SCAN":
            incar_update = {"METAGGA": "Rtpss",
                            "LASPH": True}
            actions = [{"dict": "INCAR",
                        "action": {"_set": incar_update}}]
            from custodian.vasp.interpreter import VaspModder
            VaspModder().apply_actions(actions)

            # guarantee VASP is the hacked version
            fw_env = fw_spec.get("_fw_env", {})
            vasp_cmd = fw_env.get("vasp_cmd", "vasp")
            if os.path.exists(vasp_cmd):
                vasp_path = vasp_cmd
            else:
                vasp_path = shutil.which("vasp")
            hacked_path = "/global/common/matgen/das/vasp/5.3.5-scan-beta/bin/vasp"
            if vasp_path != hacked_path:
                my_name = str(self.__class__).replace("<class '", "").replace("'>", "")
                raise ValueError("'{}' is designed to support the hack of VASP, "
                                 "upon official release of VASP SCAN function, this class"
                                 "should be modified".format(my_name))
        else:
            pass


class ChemicalShiftKptsAverageGenerationTask(FireTaskBase, FWSerializable):
    """
    This class is to spawn the dynamical fws to calculate NMR chemical shfit on each 
    individual and then do K-points weighted average manually.
    """
    _fw_name = "Chemical Shift K-Points Average Generation Task"

    def run_task(self, fw_spec):
        no_jobs_spec = copy.deepcopy(fw_spec)
        no_jobs_spec.pop('jobs', None)
        no_jobs_spec.pop('handlers', None)
        no_jobs_spec.pop('max_errors', None)
        no_jobs_spec.pop('_tasks', None)
        no_jobs_spec.pop('custodian_default_input_set', None)
        no_jobs_spec.pop('task_type', None)
        no_jobs_spec.pop('vaspinputset_name', None)
        nick_name = no_jobs_spec['parameters']['nick_name']
        priority = no_jobs_spec['_priority']
        no_jobs_spec['input_set_config_dict']['INCAR']['ISMEAR'] = 0
        no_jobs_spec['input_set_config_dict']['INCAR']['ICHARG'] = 11
        no_jobs_spec['input_set_incar_enforce'] = {"KPAR": 1}
        no_jobs_spec['task_type'] = 'Single Kpt CS'
        no_jobs_spec['vaspinputset_name'] = no_jobs_spec['task_type'] + " DictSet"
        no_jobs_spec["scf_vasp_dir"] = fw_spec['prev_vasp_dir']

        fws = []
        connections = dict()
        db_fwids = []
        cur_fwid = -1
        prev_dir = get_loc(fw_spec['prev_vasp_dir'])
        scf_kpoint_filename = zpath(os.path.join(prev_dir, 'IBZKPT'))
        whole_kpts = Kpoints.from_file(scf_kpoint_filename)
        no_jobs_spec['total_kpts'] = len(whole_kpts.kpts)

        for (i, kpt) in enumerate(whole_kpts.kpts):
            kweight = int(whole_kpts.kpts_weights[i])
            task_tag = "kpt_#{:d}_weight_{:d}".format(i + 1, kweight)
            comment = "Individual {}th Kpoint for CS Calculation".format(i + 1)
            cur_kpoints = Kpoints(comment=comment, num_kpts=1,
                                  style=Kpoints.supported_modes.Reciprocal,
                                  kpts=[kpt], kpts_weights=[1],
                                  tet_number=0)

            # Individual K-Point Chemical Shift VASP
            kpt_cs_spec = copy.deepcopy(no_jobs_spec)
            kpt_cs_spec['run_tags'].append(task_tag)
            kpt_cs_spec['kpoints_enforce'] = cur_kpoints
            kpt_cs_spec["kpoint_tag"] = task_tag

            kpt_cs_tasks = [DictVaspSetupTask({'files': ['INCAR', "KPOINTS"]}),
                            VaspCopyTask({'files': ['CHGCAR', "POTCAR"],
                                          'use_CONTCAR': True,
                                          'keep_velocities': False})]
            functional = kpt_cs_spec["functional"]
            if functional != "PBE":
                kpt_cs_tasks.append(ScanFunctionalSetupTask())
            kpt_cs_tasks.append(TagFileChecksumTask({"files": ["CHGCAR"]}))
            from mpworks.firetasks.custodian_task import get_custodian_task
            kpt_cs_tasks.append(get_custodian_task(kpt_cs_spec))
            kpt_cs_tasks.append(DeleteFileTask({"files": ["CHGCAR"]}))
            kpt_cs_task_name = get_slug(nick_name + '--' + kpt_cs_spec['task_type'] + "--#{}".format(i))
            kpt_cs_vasp_fwid = cur_fwid  # Links
            cur_fwid -= 1
            vasp_fw = Firework(kpt_cs_tasks, kpt_cs_spec, name=kpt_cs_task_name,
                               fw_id=kpt_cs_vasp_fwid)
            fws.append(vasp_fw)

            # Individual K-Point Chemical Shift VASP DB Insertion
            kpt_cs_db_fwid = cur_fwid  # Links
            cur_fwid -= 1
            kpt_cs_db_type_class = NmrVaspToDBTask
            from mpworks.workflows.snl_to_wf_nmr import get_nmr_db_fw
            kpt_cs_db_fw = get_nmr_db_fw(nick_name=nick_name, fwid=kpt_cs_db_fwid,
                                      prev_task_type=kpt_cs_spec['task_type'],
                                      priority=priority, task_class=kpt_cs_db_type_class)
            fws.append(kpt_cs_db_fw)
            connections[kpt_cs_vasp_fwid] = kpt_cs_db_fwid
            db_fwids.append(kpt_cs_db_fwid)

        # K-Points Average Collect
        collect_spec = copy.deepcopy(no_jobs_spec)
        collect_spec['task_type'] = 'Single Kpt CS Collect'
        collect_spec['vaspinputset_name'] = collect_spec['task_type'] + " DictSet"
        collect_tasks = [ChemicalShiftKptsAverageCollectTask()]
        collect_fwid = cur_fwid
        cur_fwid -= 1
        collect_fw = Firework(collect_tasks, collect_spec,
                              name=get_slug(nick_name + '--' + collect_spec['task_type']),
                              fw_id=collect_fwid)
        fws.append(collect_fw)
        for dbid in db_fwids:
            connections[dbid] = collect_fwid
        wf = Workflow(fws, connections)
        update_spec = {'total_kpts': no_jobs_spec['total_kpts'],
                       "scf_vasp_dir": fw_spec['prev_vasp_dir'],
                       'prev_vasp_dir': fw_spec['prev_vasp_dir'],
                       'prev_task_type': fw_spec['task_type']}
        stored_data = {'total_kpts': no_jobs_spec['total_kpts'],
                       "scf_vasp_dir": fw_spec['prev_vasp_dir']}
        return FWAction(update_spec=update_spec, stored_data=stored_data,
                        detours=wf)

class ChemicalShiftKptsAverageCollectTask(FireTaskBase, FWSerializable):
    """
    This class do K-points weighted chemical shift average from the previous K-points
    specific calculations.
    """
    _fw_name = "Chemical Shift K-Points Average Collect Task"

    def run_task(self, fw_spec):
        kpt_name_pattern = re.compile(r'kpt_#(?P<kpt_no>\d+)_weight_(?P<weight>\d+)')
        kpt_name_weigths = []
        for kpt_name in fw_spec.keys():
            m = kpt_name_pattern.match(kpt_name)
            if m:
                kpt_weight = m.group("weight")
                kpt_name_weigths.append([kpt_name, kpt_weight])
        num_kpts = fw_spec["total_kpts"]
        assert len(kpt_name_weigths) == num_kpts
        num_atoms = len(fw_spec[kpt_name_weigths[0][0]]['chemical_shifts']['valence_only'])
        num_ave_components = 7
        atom_cs_weight_vo_vc = [[list() for _ in range(num_ave_components)]
                                for _ in range(num_atoms)]
        for i_kpt, (kpt_name, weight) in enumerate(kpt_name_weigths):
            kpt_cs = fw_spec[kpt_name]['chemical_shifts']
            for i_atom in range(num_atoms):
                vo_tensor_notation = NMRChemicalShiftNotation.from_dict(kpt_cs['valence_only'][i_atom])
                vc_tensor_notation = NMRChemicalShiftNotation.from_dict(kpt_cs['valence_and_core'][i_atom])
                val_only_tensor_pas = vo_tensor_notation.mehring_values[1:4]
                val_core_tensor_pas = vc_tensor_notation.mehring_values[1:4]
                components = (float(weight),) + val_only_tensor_pas + val_core_tensor_pas
                for i_comp in range(num_ave_components):
                    atom_cs_weight_vo_vc[i_atom][i_comp].append(components[i_comp])
        for i_atom in range(num_atoms):
            for i_comp in range(num_ave_components):
                assert len(atom_cs_weight_vo_vc[i_atom][i_comp]) == num_kpts
        ave_pas_tensors = []
        tensor_rmsd = []
        for i_atom in range(num_atoms):
            atom_ave_tensor = []
            atom_tensor_rmsd = []
            for i_comp in range(1, num_ave_components):
                sum_value = sum([weight * tensor for weight, tensor
                                 in zip(atom_cs_weight_vo_vc[i_atom][0],
                                        atom_cs_weight_vo_vc[i_atom][i_comp])])
                sum_weights = sum(atom_cs_weight_vo_vc[i_atom][0])
                ave_value = sum_value / sum_weights
                atom_ave_tensor.append(ave_value)
                sum_square_dev = sum([weight * ((tensor - ave_value) ** 2) for weight, tensor
                                      in zip(atom_cs_weight_vo_vc[i_atom][0],
                                             atom_cs_weight_vo_vc[i_atom][i_comp])])
                rmsd_value = math.sqrt(sum_square_dev / sum_weights)
                atom_tensor_rmsd.append(rmsd_value)
            ave_pas_tensors.append(atom_ave_tensor)
            tensor_rmsd.append(atom_tensor_rmsd)
        ave_tensor_notations = {"valence_only": [], 'valence_and_core': []}
        for pas in ave_pas_tensors:
            assert len(pas) == 6
            for comp_indices, comp_key in [[range(0, 3), "valence_only"],
                                           [range(3, 6), 'valence_and_core']]:
                sigmas = [pas[i] for i in comp_indices]
                notation = NMRChemicalShiftNotation(*sigmas)
                ave_tensor_notations[comp_key].append(notation.as_dict())
        single_kpt_vasp_calcs = {kpt_name: fw_spec[kpt_name] for kpt_name, weight
                                 in kpt_name_weigths}
        cs_fields = {"chemical_shifts": ave_tensor_notations,
                     "manual_kpt_average": fw_spec,
                     "rmsd": tensor_rmsd,
                     "rmsd_header": ["valence_only_11", "valence_only_22", "valence_only_33",
                                     "valence_and_core_11", "valence_and_core_22", "valence_and_core_33"],
                     "manual_kpt_data": {
                         "total_kpts": fw_spec["total_kpts"],
                         "single_kpt_vasp_calcs": single_kpt_vasp_calcs
                     }}
        stored_data = copy.deepcopy(cs_fields)
        update_spec = copy.deepcopy(cs_fields)
        update_spec['prev_task_type'] = fw_spec['task_type']
        update_spec['prev_vasp_dir'] = fw_spec['scf_vasp_dir']
        for k in ['scf_vasp_dir', 'functional']:
            update_spec[k] = fw_spec[k]
        return FWAction(stored_data=stored_data, update_spec=update_spec)

class TagFileChecksumTask(FireTaskBase, FWSerializable):

    _fw_name = "Tag File Checksum Task"

    def __init__(self, parameters=None):
        parameters = parameters if parameters else {}
        default_files = ["CHGCAR", "WAVCAR"]
        self.update(parameters)
        self.files = parameters.get("files", default_files)

    def run_task(self, fw_spec):
        file_checksums = dict()
        blocksize = 10 * 2 ** 20 # 10 MB
        for fn in self.files:
            with zopen(zpath(fn), 'rb') as f:
                hash = hashlib.sha224()
                for block in iter(lambda: f.read(blocksize), b""):
                    hash.update(block)
                checksum = hash.hexdigest()
                file_checksums[fn] = {"type": "sha224",
                                      "value": checksum}
            with open("checksum.{}.{}".format(fn, checksum[:10]), "w") as f:
                f.write("sha224")
        stored_data = {"file_checksum": file_checksums}
        return FWAction(stored_data=stored_data)


class DeleteFileTask(FireTaskBase, FWSerializable):

    _fw_name = "Delete File Task"

    def __init__(self, parameters=None):
        parameters = parameters if parameters else {}
        default_files = ["CHGCAR", "WAVCAR"]
        self.update(parameters)
        self.files = parameters.get("files", default_files)

    def run_task(self, fw_spec):
        for fn in self.files:
            gzfn = fn + ".gz"
            if os.path.exists(fn):
                os.remove(fn)
            if os.path.exists(gzfn):
                os.remove(gzfn)

