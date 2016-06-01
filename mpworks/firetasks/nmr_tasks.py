import copy
import os
import yaml
from pymatgen.io.vasp.sets import DictSet

from mpworks.dupefinders.dupefinder_vasp import DupeFinderVasp

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

def _config_dict_to_input_set(config_dict, config_name, structure, incar_enforce, parameters):
    trial_set = DictSet(structure, name=config_name, config_dict=config_dict,
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
    vis = DictSet(structure, name=config_name, config_dict=processed_config_dict,
                  user_incar_settings=incar_enforce)
    return vis

def snl_to_nmr_spec(snl, istep_triple_jump, parameters=None):
    parameters = parameters if parameters else {}
    spec = {'parameters': parameters}

    module_dir = os.path.abspath(os.path.dirname(__file__))
    if 1<= istep_triple_jump <= 3:
        config_file = os.path.join(module_dir, "triple_jump_relax_set.yaml")
        config_key = "STEP{}".format(istep_triple_jump)
        config_name = "Triple Jump Relax S1"
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

    incar_enforce = {'NPAR': 2}
    if 'exact_structure' in parameters and parameters['exact_structure']:
        structure = snl.structure
    else:
        structure = snl.structure.get_primitive_structure()
    mpvis = _config_dict_to_input_set(config_dict, config_name, structure,
                                      incar_enforce, parameters=parameters)
    exit()




    # mpvis = MPGGAVaspInputSet(user_incar_settings=incar_enforce) if enforce_gga else MPVaspInputSet(user_incar_settings=incar_enforce)

    mpvis = None

    incar = mpvis.get_incar(structure)
    poscar = mpvis.get_poscar(structure)
    kpoints = mpvis.get_kpoints(structure)
    potcar = mpvis.get_potcar(structure)

    spec['vasp'] = {}
    spec['vasp']['incar'] = incar.as_dict()
    spec['vasp']['poscar'] = poscar.as_dict()
    spec['vasp']['kpoints'] = kpoints.as_dict()
    spec['vasp']['potcar'] = potcar.as_dict()

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

    # add exact structure run tag automatically if we have a unique situation
    if 'exact_structure' in parameters and parameters['exact_structure'] and snl.structure != snl.structure.get_primitive_structure():
        spec['run_tags'].extend('exact_structure')

    spec['_dupefinder'] = DupeFinderVasp().to_dict()
    spec['vaspinputset_name'] = mpvis.__class__.__name__
    spec['task_type'] = 'GGA+U optimize structure (2x)' if spec['vasp'][
        'incar'].get('LDAU', False) else 'GGA optimize structure (2x)'

    return spec

