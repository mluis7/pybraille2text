from pybrl2txt.models import Page, Line, Area
import pybrl2txt.braille_to_text as brl2t

args = []
args.extend(['', 'resources', '../src/resources/abbreviations.yml', "camel-case.png"])
brl2t.main(args)
