import copy
import hashlib
import json
import os

import shutil
import yaml
from fireworks import FireTaskBase
from fireworks.core.firework import FWAction, Firework
from fireworks.utilities.fw_serializers import FWSerializable
from fireworks.utilities.fw_utilities import get_slug
from monty.os.path import zpath
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.io.vasp import Outcar, zopen
from pymatgen.io.vasp.sets import DictSet

from mpworks.dupefinders.dupefinder_vasp import DupeFinderVasp
from mpworks.firetasks.vasp_io_tasks import VaspToDBTask
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
    tasks = [DictVaspSetupTask()]
    functional = scf_spec["functional"]
    if functional != "PBE":
        tasks.append(ScanFunctionalSetupTask())
    from mpworks.firetasks.custodian_task import get_custodian_task
    tasks.append(get_custodian_task(no_jobs_spec))
    scf_vasp_fwid = cur_fwid  # Links
    cur_fwid -= 1
    vasp_fw = Firework(tasks, scf_spec, name=get_slug(nick_name + '--' + scf_spec['task_type']),
                       fw_id=scf_vasp_fwid)
    fws.append(vasp_fw)

    scf_db_fwid = cur_fwid  # Links
    cur_fwid -= 1
    scf_db_type_class = VaspToDBTask
    from mpworks.workflows.snl_to_wf_nmr import get_nmr_db_fw
    scf_db_fw = get_nmr_db_fw(nick_name=nick_name, fwid=scf_db_fwid, prev_task_type=scf_spec['task_type'],
                          priority=priority, task_class=scf_db_type_class)
    fws.append(scf_db_fw)

    # Single Kpt CS
    kpt_cs_base_spec = copy.deepcopy(no_jobs_spec)
    kpt_cs_base_spec['input_set_config_dict']['INCAR']['ISMEAR'] = 0
    kpt_cs_base_spec['task_type'] = 'Single Kpt CS'
    kpt_cs_base_spec['vaspinputset_name'] = kpt_cs_base_spec['task_type'] + " DictSet"


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
        if prev_task_type == "NMR CS":
            outcar.read_chemical_shifts()
            cs_fiels = {"chemical_shifts": [x.as_dict() for x in outcar.data["chemical_shifts"]["valence_only"]]}
            nmr_fields.update(cs_fiels)
        elif prev_task_type == "NMR EFG":
            outcar.read_nmr_efg()
            efg_fields = {"efg": outcar.data["efg"]}
            nmr_fields.update(efg_fields)
        else:
            raise ValueError("Unsupported Task Type: \"{}\"".format(prev_task_type))
        self.additional_fields.update(nmr_fields)
        return super(NmrVaspToDBTask, self).run_task(fw_spec)


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

        vis.incar.write_file("INCAR")
        vis.poscar.write_file("POSCAR")
        vis.potcar.write_file("POTCAR")
        vis.kpoints.write_file("KPOINTS")
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
        pass

class ChemicalShiftKptsAverageCollectTask(FireTaskBase, FWSerializable):
    """
    This class do K-points weighted chemical shift average from the previous K-points
    specific calculations.
    """
    _fw_name = "Chemical Shift K-Points Average Collect Task"

    def run_task(self, fw_spec):
        pass


class TagFileChecksumTask(FireTaskBase, FWSerializable):

    _fw_name = "Chemical Shift K-Points Average Generation Task"

    def __init__(self, files=None):
        if files is None:
            files = ["POSCAR", "CHGCAR", "POTCAR"]
        self.files = files

    def run_task(self, fw_spec):
        file_checksums = dict()
        blocksize = 10 * 2 ** 20 # 10 MB
        for fn in self.files:
            with zopen(zpath('FW.json'), 'rb') as f:
                hash = hashlib.sha224
                for block in iter(lambda: f.read(blocksize), b""):
                    hash.update(block)
                checksum = hash.hexdigest()
                file_checksums[fn] = {"type": "sha224",
                                      "value": checksum}
            with open("checksum.{}.{}".format(fn, checksum[:10]), "w") as f:
                f.write("sha224")
        stored_data = {"file_chechsum": file_checksums}
        return FWAction(stored_data=stored_data)


class DeleteFileTask(FireTaskBase, FWSerializable):

    _fw_name = "Delete File Task"

    def __init__(self, files=None):
        if files is None:
            files = ["CHGCAR", "WAVCAR"]
        self.files = files

    def run_task(self, fw_spec):
        for fn in self.files:
            gzfn = fn + ".gz"
            if os.path.exists(fn):
                os.remove(fn)
            if os.path.exists(gzfn):
                os.remove(gzfn)

