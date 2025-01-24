import os
import yaml
from pybrl2txt.models import Page, Line, Area
import pybrl2txt.braille_to_text as brl2t


def test_case(image_path, cfg_path):
    with open(cfg_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
    
    config['cfg']['logging_level'] = 'WARNING'
    
    text, braille_lines, total_errors = brl2t.parse_image_file(f"{base_dir}/{image_path}", config)
    bname = os.path.basename(image_path).split('.')[0]
    if total_errors != 0:
        print(f"ERRORS: {total_errors} on {bname}")
    else:
        print(f"SUCCESS: {bname}")
    
    with open(f"{base_dir}/{bname}.txt", 'r') as t:
        for ln, line in enumerate(t.readlines()):
            if line == '':
                continue
            if line.strip() != text[ln].strip():
                print(f'  {"-" * 50}')
                orig = line.strip().split(' ')
                trans = text[ln].strip().split(' ')
                diff1 = [item for item in orig if item not in trans]
                diff2 = [item for item in trans if item not in orig]
                print(f"  Orig {len(orig)}, ln {ln}: {diff1}")
                print(f"  Tran {len(trans)}, ln {ln}: {diff2}")
    

args = []
args.extend(['', 'resources', '../src/resources/abbreviations.yml', "abbreviations.png"])
#brl2t.main(args)
base_dir = args[1]
cfg_path = args[2]
image_path = args[3]
test_case(image_path, cfg_path)

print(f'\n{"-" * 80}')
args = []
args.extend(['', 'resources', '../src/resources/abbreviations.yml', "camel-case.png"])
base_dir = args[1]
cfg_path = args[2]
image_path = args[3]
test_case(image_path, cfg_path)

print(f'\n{"-" * 80}')
args = []
args.extend(['', 'resources', 'resources/alfonsina-es.yml', "alfonsina-es.png"])
base_dir = args[1]
cfg_path = args[2]
image_path = args[3]
test_case(image_path, cfg_path)

print(f'\n{"-" * 80}')
args = []
args.extend(['', 'resources', 'resources/abbreviations_brl_single_line.yml', "abbreviations_brl_single_line.png"])
base_dir = args[1]
cfg_path = args[2]
image_path = args[3]
test_case(image_path, cfg_path)

print(f'\n{"-" * 80}')
args = []
args.extend(['', 'resources', 'resources/abbreviations_brl_small.yml', "abbreviations_brl_small.png"])
base_dir = args[1]
cfg_path = args[2]
image_path = args[3]
test_case(image_path, cfg_path)

print(f'\n{"-" * 80}')
args = []
args.extend(['', 'resources', 'resources/brl-fr.yml', "braille-poem-ABC.png"])
base_dir = args[1]
cfg_path = args[2]
image_path = args[3]
test_case(image_path, cfg_path)

print(f'\n{"-" * 80}')
args = []
args.extend(['', 'resources', 'resources/brl_quant_orig.yml', "brl_quant_orig.png"])
base_dir = args[1]
cfg_path = args[2]
image_path = args[3]
test_case(image_path, cfg_path)

#INFO    [pybrl2txt] Image file: technical_new.png, config: ../../tests/resources/brl.yml
print(f'\n{"-" * 80}')
args = []
args.extend(['', 'resources', 'resources/brl.yml', "technical_new.png"])
base_dir = args[1]
cfg_path = args[2]
image_path = args[3]
test_case(image_path, cfg_path)

