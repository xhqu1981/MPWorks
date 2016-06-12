import copy
import json
import os
import yaml
from fireworks import FireTaskBase
from fireworks.utilities.fw_serializers import FWSerializable
from monty.os.path import zpath
from pymatgen.io.vasp import Outcar
from pymatgen.io.vasp.sets import DictSet

from mpworks.dupefinders.dupefinder_vasp import DupeFinderVasp
from mpworks.firetasks.vasp_io_tasks import VaspToDBTask
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
    trial_set = DictSet(structure, config_dict=config_dict,
                        user_incar_settings=incar_enforce)
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
                  user_incar_settings=incar_enforce)
    return vis


def snl_to_nmr_spec(structure, istep_triple_jump, parameters=None, additional_run_tags=()):
    parameters = parameters if parameters else {}
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
    if config_name == "NMR CS":
        incar_enforce = {'KPAR': 4}
    else:
        incar_enforce = {'NPAR': 4}
    spec['run_tags'] = spec.get('run_tags', [])
    spec['run_tags'].extend(additional_run_tags)

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


class NmrVaspToDBTask(VaspToDBTask):
    _fw_name = "NMR Tensor to Database Task"

    def __init__(self, parameters=None):
        super(NmrVaspToDBTask, self).__init__(parameters)

    def run_task(self, fw_spec):
        db_dir = os.environ['DB_LOC']
        db_path = os.path.join(db_dir, 'tasks_db.json')
        with open(db_path) as f:
            db_creds = json.load(f)
            if 'prod' in db_creds['database']:
                WFSettings().MOVE_TO_GARDEN_PROD = True
            elif 'test' in db_creds['database']:
                WFSettings().MOVE_TO_GARDEN_DEV = True
        if 'nmr' not in WFSettings.GARDEN:
            WFSettings().GARDEN = os.path.join(WFSettings().GARDEN, 'nmr')
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
        super(NmrVaspToDBTask, self).run_task(fw_spec)


class DictVaspSetupTask(FireTaskBase, FWSerializable):
    _fw_name = "Dict Vasp Input Setup Task"

    def run_task(self, fw_spec):
        config_dict = fw_spec["input_set_config_dict"]
        incar_enforce = fw_spec["input_set_incar_enforce"]
        mpsnl = fw_spec["mpsnl"]
        structure = mpsnl.structure
        vis = DictSet(structure, config_dict=config_dict,
                      user_incar_settings=incar_enforce)

        vis.incar.write_file("INCAR")
        vis.poscar.write_file("POSCAR")
        vis.potcar.write_file("POTCAR")
        vis.kpoints.write_file("KPOINTS")
