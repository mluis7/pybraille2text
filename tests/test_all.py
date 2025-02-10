import os
import sys
import yaml
from pybrl2txt.models import Page, Line, Area
import pybrl2txt.braille_to_text as brl2t
from pybrl2txt.braille_maps import LANG_DFLT



def with_louis(braille_lines, lang, brl2t):
    import louis
    total_errors = 0
    text = []
    for braille_uni_str in braille_lines:
        lou_transl, lou_err = brl2t.call_louis(louis, braille_uni_str, lang)
        total_errors += lou_err
        text.append(lou_transl[0])
    
    return total_errors, text

def compare_text(text, orig_text_paths):
    with open(orig_text_paths, 'r') as t:
        for ln, line in enumerate(t.readlines()):
            if line == '':
                continue
            if line.strip() != text[ln].strip():
                #print(f'  {"-" * 50}')
                orig = line.strip().split(' ')
                trans = text[ln].strip().split(' ')
                diff1 = [item for item in orig if item not in trans]
                diff2 = [item for item in trans if item not in orig]
                print(f"  ln {ln}: Orig {diff1}")
                print(f"  ln {ln}: Tran {diff2}")
    
def test_case(image_path, cfg_path, mode='braille'):
    with open(cfg_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
    
    config['cfg']['logging_level'] = 'WARNING'
    config['cfg']['with_louis'] = mode
    lang = config['parse'].get('lang', LANG_DFLT)
    
    braille_lines, total_errors = brl2t.parse_image_file(f"{base_dir}/{image_path}", config)
    if config['cfg']['with_louis'] == 'text':
        total_errors, text = with_louis(braille_lines, lang, brl2t)
    img_name = os.path.basename(image_path)
    bname = img_name.split('.')[0]
    if total_errors != 0:
        print(f"ERRORS: {total_errors} on {img_name}, lang: '{config['parse'].get('lang', 'en-ueb-g2')}'")
    else:
        print(f"SUCCESS: {img_name}, lang: '{config['parse'].get('lang', 'en-ueb-g2')}'")
    
    if config['cfg']['with_louis'] == 'text':
        compare_text(text, f"{base_dir}/{bname}.txt")
    
    if config['cfg']['with_louis'] == 'braille':
        for b in braille_lines:
            print(b)

print(f'\n{"#" * 70}')
print("Running tests.")
print("SUCCESS means no cell detection error so inaccuracies are\nliblouis usage related (AFAIK, FIXME).")
print("Translated Languages: en-ueb-g2, en-us-g2, es-g2, fr-bfu-g2")
print(f'{"#" * 70}\n')

mode = 'braille'
if len(sys.argv) > 1:
    mode = sys.argv[1]

args = []
args.extend(['', 'resources', '../src/resources/abbreviations.yml', "camel-case.png"])
base_dir = args[1]
cfg_path = args[2]
image_path = args[3]
test_case(image_path, cfg_path, mode=mode)

print(f'\n{"-" * 80}')
args = []
args.extend(['', 'resources', 'resources/abbreviations_brl_single_line.yml', "abbreviations_brl_single_line.png"])
base_dir = args[1]
cfg_path = args[2]
image_path = args[3]
test_case(image_path, cfg_path, mode=mode)

print(f'\n{"-" * 80}')
args = []
args.extend(['', 'resources', 'resources/abbreviations_brl_small.yml', "abbreviations_brl_small.png"])
base_dir = args[1]
cfg_path = args[2]
image_path = args[3]
test_case(image_path, cfg_path, mode=mode)

print(f'\n{"-" * 80}')
args = []
args.extend(['', 'resources', 'resources/brl-fr.yml', "braille-poem-ABC.png"])
base_dir = args[1]
cfg_path = args[2]
image_path = args[3]
test_case(image_path, cfg_path, mode=mode)

print(f'\n{"-" * 80}')
args = []
args.extend(['', 'resources', '../src/resources/abbreviations.yml', "abbreviations.png"])
#brl2t.main(args)
base_dir = args[1]
cfg_path = args[2]
image_path = args[3]
test_case(image_path, cfg_path, mode=mode)

print(f'\n{"-" * 80}')
args = []
args.extend(['', 'resources', 'resources/alfonsina-es.yml', "alfonsina-es.png"])
base_dir = args[1]
cfg_path = args[2]
image_path = args[3]
test_case(image_path, cfg_path, mode=mode)

print(f'\n{"-" * 80}')
args = []
args.extend(['', 'resources', 'resources/brl_quant_orig.yml', "brl_quant_orig.png"])
base_dir = args[1]
cfg_path = args[2]
image_path = args[3]
test_case(image_path, cfg_path, mode=mode)

print(f'\n{"-" * 80}')
args = []
args.extend(['', 'resources', 'resources/brl.yml', "technical_new.png"])
base_dir = args[1]
cfg_path = args[2]
image_path = args[3]
test_case(image_path, cfg_path, mode=mode)

