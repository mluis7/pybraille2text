from pybrl2txt.models import Page, Line, Area
import pybrl2txt.braille_to_text as brl2t

args = []
args.extend(['', 'resources', 'resources/alfonsina-es.yml', "alfonsina-es.png"])
brl2t.main(args)
