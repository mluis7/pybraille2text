from datetime import datetime
import glob
import locale
from pypdf import PdfReader, mult
import string
import unicodedata as uc
from pybrl2txt.braille_maps import abbr, rev_abbr


cell_map = {
    (1,): 'a', (1,2): 'b', (1,4): 'c', (1,4,5): 'd', (1,5): 'e',
    (1,2,4): 'f', (1,2,4,5): 'g', (1,2,5): 'h', (2,4): 'i', (2,4,5): 'j',
    (1,3): 'k', (1,2,3): 'l', (1,3,4): 'm', (1,3,4,5): 'n', (1,3,5): 'o',
    (1,2,3,4): 'p', (1,2,3,4,5): 'q', (1,2,3,5): 'r', (2,3,4): 's', (2,3,4,5): 't',
    (1,3,6): 'u', (1,2,3,6): 'v', (2,4,5,6): 'w', (1,3,4,6): 'x', (1,3,4,5,6): 'y', (1,3,5,6): 'z',

    (2,): ',', (2,5,6): '.', (3,): "'", (2,3,6): '?',
    (2,5): ':', (2,3,5): '!', (3,6): '-',
    (2,5,6): '.', (2,3): ';', (2,3): ';'
    }

comp_map = {
    ((4,5,6),(3,4)): '/', ((4,5,6),(1,6)): '\\', 
    ((4,6), (1,2,6)): '[', ((4,6), (3,4,5)): ']',
    ((5,),(3,5)): '*', ((4,6), (3,5,6)): '%',
    ((5,), (1,2,6)): '(',((5,), (3,4,5)): ')',  
    }

rev_comp_map = {v:k for k,v in comp_map.items()}
rev_cell_map = {v:k for k,v in cell_map.items()}

numbers = {
    (1,): '1', (1,2): '2', (1,4): '3', (1,4,5): '4', (1,5): '5',
    (1,2,4): '6', (1,2,4,5): '7', (1,2,5): '8', (2,4): '9', (2,4,5): '0',
    }
rev_numbers = {v:k for k,v in numbers.items()}

abbreviations = {}

not_transla = {}

last_word = ''
y_txt_start = 703 #680 #620
xcolumn = [ (68, 160), (343, 434) ] # 68
#xcolumn = [ (54, 140), (325, 420) ] # 68
def visitor_before(op, args, cm, t):
    print(op, args,cm,t)
    pass

def visitor_body(text, cm, tm, fontDict, fontSize):
    '''
    Text is always made of regular ascii characters so the pdf contains 
    a braille character but pypdf returns the ascii.
    The character equivalent tuple is looked up on the dictionary.
    [pdf]    [pypdf]   [cell_map key]
    
    * .        r        (1,2,3,5)
    * *
    * .
    
    Braille can be reconstructed from text looking up the flattened tuple
    on unicodedata
    
    [text]   [tuple]         [unicodedata name]
    
      r     (1,2,3,5)     'BRAILLE PATTERN DOTS-1235'
    
    cm      : current transformation matrix
    tm      : text matrix
    fontDict: font-dictionary
    fontSize: font-size
    '''
    global last_word
    global abbreviations
    
    try:
        x = tm[4]
        y = tm[5]
        braille_str = ''
        braille_tpls = []
        if text in ['many']:
            print(f"text: '{text}'", tm)
        if y < y_txt_start and text not in ['', ' ', '\n'] and '[10' not in text:
            last_word = text
            if (x > xcolumn[0][0] and x < xcolumn[0][1]) or (x > xcolumn[1][0] and x < xcolumn[1][1]):
                #print(f"text: '{text}'")
                
                if last_word not in rev_abbr:
                    abbreviations[text] = None
            elif (x > xcolumn[0][1] and x < xcolumn[1][0]) or (x > xcolumn[1][1]):
                
                uni_name = 'BRAILLE PATTERN DOTS-'
                for c in text:
                    if c == ' ':
                        braille_str += uc.lookup("BRAILLE PATTERN BLANK")
                        braille_tpls.append((0,))
                        continue
                    try:
                        char_tuple = None
                        if c in string.digits:
                            char_tuple = rev_numbers[c]
                        else:
                            char_tuple = rev_cell_map[c]
                        
                        braille_tpls.append(char_tuple)
                        uni_name_suff = ''.join([str(s) for s in char_tuple])
                        braille_str += uc.lookup(uni_name + uni_name_suff)
                    except Exception as e:
                        if c in rev_comp_map:
                            braille_tpls.extend([m for m in rev_comp_map[c]])
                        else:
                            not_transla[last_word] = tuple(braille_tpls) 
                            print(f"ERROR. '{c} ({ord(c)})' not found for {last_word} ({braille_tpls})")
                        pass
                if last_word not in rev_abbr:
                    abbreviations[last_word] = tuple(braille_tpls)
                last_word = None
                #print(f"braille: '{braille_str}', {braille_tpls} ({text})")
            else:
                pass
    except Exception as e:
        pass

with open('/home/lmc/projects/eclipse-workspace/SOPython/lmc/braille_to_text_poc/symbols_list.pdf', "rb") as fd:
    reader = PdfReader(fd)
    print(f"Found {len(reader.pages)} pages")
    for page in reader.pages[2:]:
    #page = reader.pages[2]

        parts = ['', 0, 0, 0, '']
    
        text = page.extract_text( visitor_text=visitor_body) #
    print(abbreviations)
    print('\n')
    print(not_transla)
    #text = page.extract_text()
    #print(text)

