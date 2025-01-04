
__all__ = ['BLANK', 'cell_map', 'numbers', 'prefixes', 'abbr']
BLANK = (0,0,0,0,0,0)

cell_map = {
    (1,): 'a', (1,2): 'b', (1,4): 'c', (1,4,5): 'd', (1,5): 'e',
    (1,2,4): 'f', (1,2,4,5): 'g', (1,2,5): 'h', (2,4): 'i', (2,4,5): 'j',
    (1,3): 'k', (1,2,3): 'l', (1,3,4): 'm', (1,3,4,5): 'n', (1,3,5): 'o',
    (1,2,3,4): 'p', (1,2,3,4,5): 'q', (1,2,3,5): 'r', (2,3,4): 's', (2,3,4,5): 't',
    (1,3,6): 'u', (1,2,3,6): 'v', (2,4,5,6): 'w', (1,3,4,6): 'x', (1,3,4,5,6): 'y', (1,3,5,6): 'z',

    (2,): ',', (2,5,6): '.', (3,): "'", (2,3,6): '?',
    (2,3,5): '!', (3,6): '-',
    (2,5,6): '.', #(2,3): ';', (2,5): ':', (2,5): ':' 
    BLANK: ' '
    
    }

numbers = {
    (1,): '1', (1,2): '2', (1,4): '3', (1,4,5): '4', (1,5): '5',
    (1,2,4): '6', (1,2,4,5): '7', (1,2,5): '8', (2,4): '9', (2,4,5): '0',
    }

prefixes = [(6,), # Upper case prefix 
            (2,5,6),
            #(4,5), # initial letter contraction
            #(4,5,6), # initial letter contraction
            # (5,6), # initial letter contraction
            (3,4,5,6)] # numeral prefix

# extracted from https://www.brailleauthority.org/ueb/symbols_list.pdf
rev_abbr = {'about': ((1,), (1, 2)), 'above': ((1,), (1, 2), (1, 2, 3, 6)), 'according': ((1,), (1, 4)),
        'across': ((1,), (1, 4), (1, 2, 3, 5)), 'after': ((1,), (1, 2, 4)), 
        'afternoon': ((1,), (1, 2, 4), (1, 3, 4, 5)), 'afterward': ((1,), (1, 2, 4), (2, 4, 5, 6)), 
        'again': ((1,), (1, 2, 4, 5)), 'against': ((1,), (1, 2, 4, 5), (3, 4)), 
        'almost': ((1,), (1, 2, 3), (1, 3, 4)), 'already': ((1,), (1, 2, 3), (1, 2, 3, 5)), 
        'also': ((1,), (1, 2, 3)), 'although': ((1,), (1, 2, 3), (1, 4, 5, 6)), 
        'altogether': ((1,), (1, 2, 3), (2, 3, 4, 5)), 'always': ((1,), (1, 2, 3), (2, 4, 5, 6)), 
        'ance': ((4, 6), (1, 5)), 'and': ((1, 2, 3, 4, 6),), 'ar': ((3, 4, 5),), 'as': ((1, 3, 5, 6),),
        # bb/be
        'bb/be': ((2, 3),), #'be ': ((2, 3),), 
        'because': ((2, 3), (1, 4)), 'before': ((2, 3), (1, 2, 4)), 
        'behind': ((2, 3), (1, 2, 5)), 'below': ((2, 3), (1, 2, 3)), 'beneath': ((2, 3), (1, 3, 4, 5)), 
        'beside': ((2, 3), (2, 3, 4)), 'between': ((2, 3), (2, 3, 4, 5)), 'beyond': ((2, 3), (1, 3, 4, 5, 6)), 
        'blind': ((1, 2), (1, 2, 3)), 'braille': ((1, 2), (1, 2, 3, 5), (1, 2, 3)), 'but': ((1, 2),), 
        'can': ((1, 4),), 'cannot': ((4,5,6),(1, 4)), 'cc': ((2,5),),
        # ch/child
        'ch/child': ((1,6),), #'child': ((1, 6),),
        'character': ((5,), (1, 6)), 'children': ((1,6), (1, 3, 4, 5)),
        'con': ((2, 5),), 'conceive': ((2, 5), (1, 4), (1, 2, 3, 6)), 
        'conceiving': ((2, 5), (1, 4), (1, 2, 3, 6), (1, 2, 4, 5)), 'could': ((1, 4),(1, 4, 5)), 
        
        'day': ((5,), (1, 4, 5)),
        'declare': ((1, 4, 5), (1, 4), (1, 2, 3)), 'declaring': ((1, 4, 5), (1, 4), (1, 2, 3), (1, 2, 4, 5)), 
        'deceive': ((1, 4, 5), (1, 4), (1, 2, 3, 6)), 'deceiving': ((1, 4, 5), (1, 4), (1, 2, 3, 6), (1, 2, 4, 5)),
        'dis': ((1, 4, 5),), 'do': ((1, 4, 5),), 'ea': ((2,),), 'ed': ((1,2,4,6),), 'either': ((1, 5), (2, 4)), 
        'en': ((1, 5),), 'ence': ((2, 3), (1, 5)), 'enough': ((2, 6),), 'er': ((1, 2, 4, 5, 6),), 
        'ever': ((5,),(1, 5),), 'every': ((1, 5),), 'father': ((5,), (1, 2, 4),), 'ff': ((2, 3, 5),), 
        'first': ((1, 2, 4), (3, 4)), 'for': ((1,2,3,4,5,6),), 'friend': ((1, 2, 4), (1, 2, 3, 5)), 
        'from': ((1, 2, 4),), 'ful': ((2, 3), (1, 2, 3)), 'gg': ((1, 2, 4, 5),), 'gh': ((1,2,6),), 
        'good': ((1, 2, 4, 5), (1, 4, 5)), 'great': ((1, 2, 4, 5), (1, 2, 3, 5), (2, 3, 4, 5)), 
        'had': ((4, 5 ,6), (1, 2, 5),), 'have': ((1, 2, 5),), 'here': ((5,), (1, 2, 5),), 'herself': ((1, 2, 5), (1, 2, 4, 5, 6), (1, 2, 4)), 
        'him': ((1, 2, 5), (1, 3, 4)), 'himself': ((1, 2, 5), (1, 3, 4), (1, 2, 4)), 'his': ((1, 2, 5),), 
        'immediate': ((2, 4), (1, 3, 4), (1, 3, 4)), 'in': ((3, 5),), 'ing': ((3, 4, 6),), 'it': ((1, 3, 4, 6),),
        'its': ((1, 3, 4, 6), (2, 3, 4)), 'itself': ((1, 3, 4, 6), (1, 2, 4)), 'ity': ((2, 3), (1, 3, 4, 5, 6)),
        'just': ((2, 4, 5),), 'know': ((5,), (1, 3)), 'knowledge': ((1, 3),), 'less': ((4, 6), (2, 3, 4)), 
        'letter': ((1, 2, 3), (1, 2, 3, 5)), 'like': ((1, 2, 3),), 'little': ((1, 2, 3), (1, 2, 3)), 'lord': ((1, 2, 3),),
        'many': ((1, 2, 3), (1, 3, 4),), 'ment': ((2, 3), (2, 3, 4, 5)), 'more': ((1, 3, 4),), 'mother': ((5,),(1,3,4)),

        'much': ((1, 3, 4), (1, 6)), 'must': ((1, 3, 4), (3, 4)), 'myself': ((1, 3, 4), (1, 3, 4, 5, 6), (1, 2, 4)), 
        'name': ((5,), (1, 3, 4, 5),), 'necessary': ((1, 3, 4, 5), (1, 5), (1, 4)), 'neither': ((1, 3, 4, 5), (1, 5), (2, 4)),
        'ness': ((5, 6), (2, 3, 4)), 'not': ((1, 3, 4, 5),), 'of': ((1, 2, 3, 5, 6),), 
        'one': ((5,), (1, 3, 5),), 'oneself': ((5,), (1, 3, 5), (1, 2, 4)), 'ong': ((6, 6), (1, 2, 4, 5)), 
        # ou/out
        'ou/out': ((1, 2, 5, 6),), #'out': ((1, 2, 5, 6),), 
        'ound': ((4, 6), (1, 4, 5)), 'ount': ((4, 6), (2, 3, 4, 5)), 
        'ought': ((5,),(1,2,5,6)), 'ourselves': ((1, 2, 5, 6), (1, 2, 3, 5), (1, 2, 3, 6), (2, 3, 4)), 
        'ow': ((2, 4, 6),), 'paid': ((1, 2, 3, 4), (1, 4, 5)), 'part': ((5,), (1, 2, 3, 4),), 
        
        'people': ((1, 2, 3, 4),), 'perceive': ((1, 2, 3, 4), (1, 2, 4, 5, 6), (1, 4), (1, 2, 3, 6)),
        'perceiving': ((1, 2, 3, 4), (1, 2, 4, 5, 6), (1, 4), (1, 2, 3, 6), (1, 2, 4, 5)),
        'perhaps': ((1, 2, 3, 4), (1, 2, 4, 5, 6), (1, 2, 5)),
 
        'question': ((5,), (1, 2, 3, 4, 5),), 'quick': ((1, 2, 3, 4, 5), (1, 3)), 'quite': ((1, 2, 3, 4, 5),), 
        'rather': ((1, 2, 3, 5),), 'receive': ((1, 2, 3, 5), (1, 4), (1, 2, 3, 6)), 
        'receiving': ((1, 2, 3, 5), (1, 4), (1, 2, 3, 6), (1, 2, 4, 5)), 'rejoice': ((1, 2, 3, 5), (2, 4, 5), (1, 4)), 
        'rejoicing': ((1, 2, 3, 5), (2, 4, 5), (1, 4), (1, 2, 4, 5)), 'right': ((1, 2, 3, 5),), 
        'said': ((2, 3, 4), (1, 4, 5)), 'sh': ((4, 6), (3, 5, 6)), 'shall': ((1, 4, 6),), 
        'should': ((1, 4, 6), (1, 4, 5)), 'sion': ((1, 6), (1, 3, 4, 5)), 'so': ((2, 3, 4),), 
        'some': ((5,), (2, 3, 4),), 'spirit': ((4, 5, 6), (2, 3, 4),), 'st': ((3, 4),), 'still': ((3, 4),), 
        'such': ((2, 3, 4), (1, 6)), 'th': ((1, 4, 5, 6),), 
        
        'the': ((2,3,4,6),), 'their': ((4, 5, 6), (3, 4, 6),), 'themselves': ((2, 3, 4, 6), (1, 3, 4), (1, 2, 3, 6), (2, 3, 4)), 're': ((2, 3, 5),), 
        'these': ((4, 5), (2, 3, 4, 6),), 'this': ((2, 4, 5, 6),), 'those': ((4, 5), (1, 4, 5, 6)), 'through': ((5,), (1, 4, 5, 6)), 
        'thyself': ((1, 4, 5, 6), (1, 3, 4, 5, 6), (1, 2, 4)), 'time': ((5,), (2, 3, 4, 5),), 
        'tion': ((5,6), (1, 3, 4, 5)), 'today': ((2, 3, 4, 5), (1, 4, 5)), 'together': ((2, 3, 4, 5), (1, 2, 4, 5), (1, 2, 3, 5)),
        'tomorrow': ((2, 3, 4, 5), (1, 3, 4)), 'tonight': ((2, 3, 4, 5), (1, 3, 4, 5)), 'under': ((5,), (1, 3, 6),), 
        'upon': ((4,5), (1, 3, 6),), 'us': ((1, 3, 6),), 'very': ((1, 2, 3, 6),), 'was': ((2, 4, 5),), 
        'were': ((2, 3, 5, 6),), 'where': ((2, 3, 5, 6),),
        'wh/which': ((1, 5, 6),), #'which': ((1, 5, 6),), # wh/which
        'will': ((2, 4, 5, 6),), 'with': ((3, 4, 5, 6),), 'word': ((4, 5, 6),(2, 4, 5, 6)), 'work': ((5,),(2, 4, 5, 6)), 
        'world': ((3, 4, 5),(2, 4, 5, 6),), 'would': ((2, 4, 5, 6), (1, 4, 5)), 'you': ((1, 3, 4, 5, 6),), 
        'young': ((5,), (1, 3, 4, 5, 6)), 'your': ((1, 3, 4, 5, 6), (1, 2, 3, 5)), 'yourself': ((1, 3, 4, 5, 6), (1, 2, 3, 5), (1, 2, 4)), 
        'yourselves': ((1, 3, 4, 5, 6), (1, 2, 3, 5), (1, 2, 3, 6), (2, 3, 4))
}

abbr = { v:k for k,v in rev_abbr.items()}

def print_flattened_keys_values():
    sk = {}
    for k, v in abbr.items():
        try:
            res = ''.join([str(idx) for tup in list(k) for idx in tup])
            if res in sk:
                res += 'a'
            sk[res] = v
        except:
            print(k, v)
    
    for k in sorted(sk.keys()):
        print(k, sk[k])

