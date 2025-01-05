'''
Created on Dec 31, 2024

@author: lmc
'''

import sys
import yaml
import cv2
import numpy as np
import unicodedata as uc
from pybrl2txt.models import Page, Line, Area
from pybrl2txt.braille_maps import BLANK, abbr, cell_map, numbers, prefixes as pfx, rev_abbr,\
    UPPER, NUMBER
from future.builtins.misc import isinstance
import logging

logger = logging.getLogger("pybrl2txt")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)-7s [ %(name)s ] %(message)s" ,
#    format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
#    datefmt="%d/%b/%Y %H:%M:%S",
    stream=sys.stdout)
#abbr = { tuple([t for t in v]) : k for k,v in rev_abbr.items()}
#abbr = dict(map(reversed, rev_abbr.items()))

def get_build_detector_params(cv2_params):
    min_area = cv2_params['cv2_cfg']['detect']['min_area']
    max_area = cv2_params['cv2_cfg']['detect']['max_area']
    min_inertia_ratio = cv2_params['cv2_cfg']['detect']['min_inertia_ratio']
    min_circularity = cv2_params['cv2_cfg']['detect'].get('min_circularity', 0.9)
    min_convexity = cv2_params['cv2_cfg']['detect'].get('min_convexity', 0.75)
    # Set up SimpleBlobDetector
    params = cv2.SimpleBlobDetector_Params()
    
    #params.blobColor = 0
# Filter by area (size of the blob)
    params.filterByArea = True
    params.minArea = min_area # 10 Adjust based on dot size
    params.maxArea = max_area # 1000 Filter by circularity
    params.filterByCircularity = True
    params.minCircularity = min_circularity # Adjust for shape of the dots
# Filter by convexity
    params.filterByConvexity = False
    params.minConvexity = min_convexity
# Filter by inertia (roundness)
    params.filterByInertia = True
    params.minInertiaRatio = min_inertia_ratio
    #params.minDistBetweenBlobs = 12
    return params

def show_detection(image, detected_lines, xcell, csize, xmin, ymax):
    """Help to visually debug if lines are correctly detected since dots would be colored by line.
    Black dots represent not correctly detected cells/lines.
    Color will repeat every for lines."""
    
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (178,102,255)]
    while len(colors) < len(detected_lines):
        colors.extend(colors)
    # Draw detected blobs as red circles
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
#    x = xmin
#    for line in detected_lines:
#        for i in range(1,len(line)):
#            output_image = cv2.line(image, ( int(x + csize * i), 50), (int(x + csize * i),ymax), (0, 255, 0), thickness=1)

    for i, line in enumerate(detected_lines):
        output_image = cv2.drawKeypoints(output_image, line, np.array([]), colors[i], cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    logger.info("Showing detection result")
    cv2.imshow(f"Detected blobs.", output_image)
    if cv2.waitKey(0) & 0xFF == ord('q'):  
        cv2.destroyAllWindows() 

def group_by_lines(kp_map, blob_coords, xydiff, page_params):
    """Group coordinates by lines."""
    ycell = page_params.cell_params.ydot
    lines_coord = []
    detected_lines = [[kp_map[blob_coords[0][0], blob_coords[0][1]]]]
    #print(f"new line {'0':>2} at: {int(blob_coords[0][0])},{int(blob_coords[0][1])}")
    # split coordinates by lines
    line_cnt = 1
    for i, d in enumerate(xydiff):
        curr_pt = blob_coords[i + 1]
        if curr_pt[0] >= 714 and curr_pt[1] >= 820 and curr_pt[1] < 851:
            pass
        current_keypoint = kp_map[curr_pt[0], curr_pt[1]]
        if (d[0] < 0 and d[1] >= ycell * 2.5) :# or (curr_pt[0] < prev_pt[0] and curr_pt[1] > prev_pt[1]):
            #print(f"new line {line_cnt:>2} at: {curr_pt}, curr xdiff: {np.round(d)}, {page_params.xmax * -0.3:.0f}, previous: {blob_coords[i]}")
            detected_lines.append([current_keypoint])
            line_cnt += 1
        else:
            detected_lines[-1].append(current_keypoint)
    p0 = 0
    p1 = len(detected_lines[0])
    for j, lc in enumerate(detected_lines):
        if j > 0:
            p0 = p0 + len(detected_lines[j - 1])
        p1 = p0 + len(detected_lines[j])
        lines_coord.append(blob_coords[p0:p1])
    
    for i,d in enumerate(detected_lines):
        if len(d) != len(lines_coord[i]):
            logger.error("ERROR on group by")
    return detected_lines, lines_coord

def get_area_parameters(coords, area_obj: Area):
    """Parameters to help find cells from detected coordinates.
    """
    # x,y differences between contiguous dots. Negative values mean second/third rows in a cell and the start of a line.
    # e.g.: previous point p0=(520,69), current point =(69, 140). xydiff = (-451, 71).
    # xdiff is negative, ydiff is greater than vertical cell size --> current dot is starting a line.
    xydiff = np.array([ (kp[0] - coords[i-1][0], kp[1] - coords[i-1][1]) for i,kp in enumerate(coords) if i > 0 ])
    
    xcoords = coords[coords[:,0] > 1][:,0]
    ycoords = coords[coords[:,1] > 1][:,1]
    # minimum x in the whole image
    xmin = np.array(xcoords).min()
    # max x in the whole image. Represents last dot in a line.
    xmax = np.array(xcoords).max()
    # minimum y in the whole image
    ymin = np.array(ycoords).min()
    ymax = np.array(ycoords).max()
    
    area_obj.xmin = xmin
    area_obj.xmax = xmax
    area_obj.ymin = ymin
    area_obj.ymax = ymax
    
    xuniq_from_diff = np.unique(np.round(xydiff[(xydiff[:,0] > 1)][:,0]))
    # x separation between dots in a cell
    xcell = xuniq_from_diff.min()
    # y separation between dots in a cell
    yuniq_from_diff = np.unique(np.round(xydiff[(xydiff[:,1] > 1)][:,1]))
    ycell = yuniq_from_diff.min()
    # x separation between cells
    xsep = np.unique(xydiff[(xydiff[:,0] > xcell + 1)][:,0]).min()
    
    area_obj.cell_params.xdot = xcell
    area_obj.cell_params.ydot = ycell
    area_obj.cell_params.xsep = xsep
    area_obj.cell_params.csize = round(xcell + xsep)
    
    # If it's a Line, set the y-coord possible values 
    if isinstance(area_obj, Line):
        yuniq = np.unique(np.round(ycoords[(ycoords > 1)]))
        area_obj.ydot14 = yuniq[0]
        if yuniq.size > 1:
            area_obj.ydot25 = yuniq[1]
        if yuniq.size > 2:
            area_obj.ydot36 = yuniq[2]
        logger.debug(f"{area_obj}")
    
    #print(f"xuniq_from_diff: {xuniq_from_diff}\nyuniq: {yuniq_from_diff}")
    return area_obj, xydiff

def get_keypoints(img_path, cv2_params):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    params = get_build_detector_params(cv2_params)
    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params) # Detect blobs
    keypoints = detector.detect(image)
    return keypoints, image

def cell_keypoints_to_braille_indexes(cell, line_params, idx):
    """Return a sorted tuple representing dot indexes in the cell.
    The tuple should map to a text character in a braille_maps dict.
    The index detection logic could probably be simplified/merged but 
    keeping it verbose helps debugging a lot.
    
    Indexes are
    1 4
    2 5
    3 6
    
    Cell    Indexes       Text
    * .
    * .
    * * --> (1,2,3,6) --> 'v'
    
    If the cell has 2 different 'x' values then the whole cell may be determined.
    cell= [[10, 10],[20, 40]]
    
    xcol123 = cell[:,0].min()
    xcol456 = cell[:,0].max()
    
    # min and max could belong to dots in the first or second row.
    ycol14_25 = cell[:,1].min()
    ycol25_36 = cell[:,1].max()
    * .  * *  . .  . .
    . .  * *  * *  * .
    . *  . .  * *  . *
    
    Cells with dots on the second column only need special care since 
    another reference is required to determine which column the dots are in.
    
    (xcol123 == xcol456) = True 
    * .  . .  . *  . .
    * .  * .  . *  . *
    . .  * .  . .  . *
    
    (ycol14_25 == ycol25_36) = True
    single row
    * *  . . . .
    . .  * * . .
    . .  . . * *
    
    (xcol123 == xcol456) and (ycol14_25 == ycol25_36) = True
    Single dot
    * .  . .  . .  . * . .  . .
    . .  * .  . .  . . . *  . .
    . .  . .  * .  . . . .  . *
    """
    is_cell_error = False
    cause = 'UNKNOWN'
    if idx == 0 and line_params.ymax > 800:
        pass
    elif line_params.line_num == 1 and idx == 30:
        pass
    cell_start = get_cell_start(line_params, idx)# - line_params.xmin
    cell_idxs = []
    cell_idx = -1
    
    ydot14 = line_params.ydot14
    ydot25 = line_params.ydot25
    ydot36 = line_params.ydot36
    
    cell_middle = round(cell_start + line_params.cell_params.xdot * 0.8)
    
    xcol123 = cell[:,0].min()
    xcol456 = cell[:,0].max()
    ycol14_25 = cell[:,1].min()
    ycol25_36 = cell[:,1].max()
    
    if xcol123 != xcol456:
        cell = cell[np.lexsort((cell[:,0], cell[:,1]))]
        xucell = np.unique(cell[:,0])
        cell_middle = xucell[0] + round((xucell[1]-xucell[0])/2) 
    
    if len(cell) > 6:
        logger.error(f"ERROR. Cell has more than 6 dots")
        return [-1], True
    elif len(cell) == 0:
        logger.error(f"ERROR. Cell is empty")
        return [-1], True
    
    for cc in cell:
        if xcol123 == xcol456 and ycol14_25 == ycol25_36:
            # Single dot cell
            if cc[0] < cell_middle:
                if cc[1] <= ydot14:
                    cell_idx = 1
                elif cc[1] > ydot14 and cc[1] <= ydot25:
                    cell_idx = 2
                elif cc[1] >= ydot36:
                    cell_idx = 3
            elif cc[0] > cell_middle:
                if cc[1] <= ydot14:
                    cell_idx = 4
                elif cc[1] > ydot14 and cc[1] <= ydot25:
                    cell_idx = 5
                elif cc[1] >= ydot36:
                    cell_idx = 6
        elif xcol123 != xcol456 and ycol14_25 == ycol25_36:
            # 2 dots in a row
            if cc[1] <= ydot14:
                if cc[0] < cell_middle:
                    cell_idx = 1
                elif cc[0] > cell_middle:
                    cell_idx = 4
            elif cc[1] > ydot14 and cc[1] <= ydot25:
                if cc[0] < cell_middle:
                    cell_idx = 2
                elif cc[0] > cell_middle:
                    cell_idx = 5
            elif cc[1] > ydot25 and cc[1] <= ydot36:
                if cc[0] < cell_middle:
                    cell_idx = 3
                elif cc[0] > cell_middle:
                    cell_idx = 6
        elif xcol123 == xcol456 and ycol14_25 != ycol25_36:
            # 2 or 3 dots in a single row
            if cc[0] < cell_middle:
                if cc[1] <= ydot14:
                    cell_idx = 1
                elif cc[1] > ydot14 and cc[1] <= ydot25:
                    cell_idx = 2
                elif cc[1] > ydot25 and cc[1] <= ydot36:
                    cell_idx = 3
            elif cc[0] > cell_middle:
                if cc[1] <= ydot14:
                    cell_idx = 4
                elif cc[1] > ydot14 and cc[1] <= ydot25:
                    cell_idx = 5
                elif cc[1] > ydot25 and cc[1] <= ydot36:
                    cell_idx = 6
        elif xcol123 != xcol456 and ycol14_25 != ycol25_36:
            # 2 or more dots in a different row/column
            if cc[0] < cell_middle:
                if cc[1] <= ydot14:
                    cell_idx = 1
                elif cc[1] > ydot14 and cc[1] <= ydot25:
                    cell_idx = 2
                elif cc[1] > ydot25 and cc[1] <= ydot36:
                    cell_idx = 3
            elif cc[0] > cell_middle:
                if cc[1] <= ydot14:
                    cell_idx = 4
                elif cc[1] > ydot14 and cc[1] <= ydot25:
                    cell_idx = 5
                elif cc[1] > ydot25 and cc[1] <= ydot36:
                    cell_idx = 6
        else:
            cell_idx = -1

        err_msg = f"{line_params}, idx: {idx}, cell_middle: {cell_middle}, kp: {cc}"
        if cell_idx == -1:
            cause = 'not found'
            is_cell_error = True
            logger.warning(f"WARN. cell_idx {cause}. {err_msg}")
        elif cell_idx in cell_idxs:
            cause = 'duplicate'
            is_cell_error = True
            logger.warning(f"WARN. cell_idx {cause}. {err_msg}")
        cell_idxs.append(cell_idx)
        cell_idx = -1
    return tuple(sorted(cell_idxs)), is_cell_error

def translate_cell(cell, line_params, idx):
    """Translate cell coordinates to braille cell indexes."""
    
    error_count = 0
    if cell is BLANK:
        #print(f"SPACE: {idx}, {BLANK}")
        return BLANK, 0
    brl_idx, is_cell_error = cell_keypoints_to_braille_indexes(cell, line_params, idx)
    if is_cell_error:
        error_count += 1
        logger.error(f"index translation error\n{cell} -> {brl_idx}")
    return brl_idx, error_count

def translate_cells(cells, line_params):
    """Translate list of coordinates tuples to list of Braille indexes tuples by word""" 
    word_tuples = []
    word_tpl = []
    total_trn_err = 0
    for idx, cell in enumerate(cells):
        #print(cell)
        tcell, cell_err_count = translate_cell(cell, line_params, idx)
        total_trn_err += cell_err_count
    
        if tcell is not BLANK:
            word_tpl.append(tcell)
        elif len(word_tpl) > 0:
            word_tpl.append(BLANK)
            word_tuples.append(tuple(word_tpl))
            word_tpl = []
    if total_trn_err > 0:
        logger.error(f"Line {line_params.line_num} cell translation errors: {total_trn_err}")
    return word_tuples


def get_cell_start(line_params, idx):
    cell_start = line_params.xmin
#    if idx == 1:
#        cell_start += line_params.cell_params.csize
    if idx >= 1:
        cell_start += (line_params.cell_params.csize) * idx #- line_params.cell_params.xdot * 0.3
    return cell_start

def translate_line(line_coor, ln, page):
    """Return array of cells in a line with BLANK inserted.
    Area parameters are recalculated with specific line values.
    """
    line_params = Line()
    line_params.line_num = ln
    line_params = get_area_parameters(line_coor, line_params)[0]
    cp = line_params.cell_params
    
    # Line min x coordinate is greater than page one so the 
    # line probably starts with dots in the second column (4,5,6)
    # or with spaces.
    if line_params.xmin > page.xmin:
            line_params.xmin = page.xmin
    # Line max x coordinate is less that page one so the line is shorter.
    if line_params.xmax < page.xmax:
            line_params.xmax = page.xmax
    # Calculated cell size for the line is greater than page calculated cell size.
    # It's probably an indicator of page irregularities or bad blob detections
    if line_params.cell_params.csize > page.cell_params.csize:
        logger.warning(f"WARN. Line cell size changed: page csize: {page.cell_params.csize}, line csize {line_params.cell_params.csize}")
        line_params.cell_params.csize = page.cell_params.csize
        
    logger.debug(f"Line: {ln}, X params: xcell {cp.xdot :.0f}, xmin {line_params.xmin:.0f}, xsep {cp.xsep:.0f}, csize {cp.csize:.0f}, xmax {line_params.xmax:.0f}")
    logger.debug(f"Line: {ln}, Y params: ycell {cp.ydot:.0f}, ymin {line_params.ymin:.0f}")
    
    #cell_count_expected = int((line_params.xmax - line_params.xmin) / line_params.cell_params.csize)
    cells = []
    blank_count = 0
    
    idx = 0
    cell_start = get_cell_start(line_params, idx)
    if idx == 0 and cell_start > page.xmin:
        line_params.xmin = page.xmin
        cell_start = line_params.xmin
    
    cell_end = line_params.xmin + line_params.cell_params.csize #  line_params.cell_params.xdot + line_params.cell_params.xsep / 2
    line_end = 0
    
    while line_end <= line_params.xmax:
        #print("line, cell_start, cell_end", ln, cell_start, cell_end)
        cell = line_coor[(line_coor[:, 0] >= cell_start) & (line_coor[:, 0] < cell_end)]
        if len(cell) == 0:# or (len(cell) > 0 and cell[0][0] > cell_end):
            cells.append(BLANK)
            line_end += line_params.cell_params.csize
            blank_count += 1
        else:
            cells.append(cell)
            line_end = cells[-1][0][0]
        idx += 1
        cell_start = get_cell_start(line_params, idx)
        cell_end += line_params.cell_params.csize
    
    logger.debug(f"Line {ln}, Found cells: {len(cells)}, Spaces: {blank_count}, cell width: {line_params.cell_params.csize}, pt count: {len(line_coor)}")
    return cells, line_params, blank_count


def translate_word_text_by_indexes(word_tuples, line_num):
    """Translate cell indexes to text using maps.
    This is the toughest task since Braille reading rules must be applied. 
    """
    
    line_text = []
    line_abbr = ''
    line_other = ''
    total_errors = 0
    prefix = None
    for w, wrd in enumerate(word_tuples):
        #print("wrd", wrd)
        if wrd[-1] is BLANK:
            prefix = None
            wrd = tuple(wrd[:-1])
        wrd_txt = ''
        if len(wrd) == 1:
            if wrd in abbr:
                line_text.append(abbr[wrd])
                line_abbr += f"{abbr[wrd]} "
            elif wrd[0] in cell_map:
                line_text.append(cell_map[wrd[0]])
                line_other += f"{cell_map[wrd[0]]} "
        elif len(wrd) > 1:
            if wrd in abbr:
                line_text.append(abbr[wrd])
                line_abbr += f"{abbr[wrd]} "
            else:
                for char in wrd:
                    if char in pfx.values():
                        prefix = char
                        continue
                    if prefix ==  pfx[NUMBER] and char in numbers:
                        wrd_txt += numbers[char]
                        line_other += numbers[char]
                        continue
                    
                    if char in cell_map:
                        wrd_txt += cell_map[char]
                        line_other += cell_map[char]
                        if prefix ==  pfx[NUMBER]:
                            prefix = None
                if prefix is not None and prefix == pfx[UPPER]:
                    wrd_txt = wrd_txt[0].upper() + ''.join(wrd_txt[1:])
                    prefix = None
                line_text.append(wrd_txt)
                wrd_txt = ''
                line_other += f" {wrd}\n"
        else:
            if wrd == ():
                continue
            else:
                line_text.append("XXXXX")
            total_errors += 1
    
    #print(f" abbr found: {line_abbr}")
    #print(f"other found: {line_other}")
    logger.info(f"Line {line_num}: Tuples to words translations: {len(line_text)}")
    return line_text, total_errors

def main():
    logging.basicConfig(level=logging.DEBUG)
    
    base_dir = '/home/lmc/projects/eclipse-workspace/SOPython/lmc/braille_to_text_poc'
    cfg_path = '../resources/abbreviations.yml'
    cfg_path = '../resources/abbreviations_brl_single_line.yml'
    
    #image_path = "braille-poem2.png"
    #image_path = "braille.jpg"
    #image_path = "technical.jpg"
    #image_path = "contracted_braille_example.webp"
    #image_path = "result2.brf.png"
    #image_path = "result2.brf_ABC.png"
    #image_path = "abbreviations_brl_abc.png"
    image_path = "abbreviations_brl_1line.png"
    image_path = "abbreviations_brl_single_line.png"
    
    
    grade = 2
    round_to = 2
    
    with open(cfg_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
        grade = config['grade']
        round_to = config['parse']['round_to']
        xmin = config['parse'].get('xmin')
        show_detect = config['cv2_cfg']['detect']['show_detect']
    
    img_path = f"{base_dir}/{image_path}"
    logger.info(f"Starting Grade {grade} braille to text translation of {image_path}")
    keypoints, image = get_keypoints(img_path, config)
    # map of keypoints coordinates to keypoints
    kp_map = { (round(kp.pt[0], round_to), round(kp.pt[1], round_to)): kp for kp in keypoints}
    
    areas = np.unique(np.array([round(kp.size) for kp in keypoints]))
    areas_diff = np.diff(areas)
    if areas_diff.size > 0 and areas_diff.max() >= config['cv2_cfg']['detect']['min_area'] * 0.5:
        logger.warning(f"Too many blob sizes detected. Cell detection will probably be poor or bad. Sizes: {areas}")
    # all dots coordinates, sorted to help find lines.
    blob_coords = np.array(list(kp_map.keys()))
    #blob_coords = np.array([kp.pt for kp in keypoints])
    blob_coords = blob_coords[np.lexsort((blob_coords[:,0], blob_coords[:,1]))]
    #blob_coords = np.round(blob_coords, decimals=2)
    
    page, xydiff = get_area_parameters(blob_coords, Page())
    cp = page.cell_params
    if xmin is not None:
        page.xmin = xmin
    
    logger.info(f"Detected blobs: {len(blob_coords)}")
    logger.info(f"Page X params: xcell {cp.xdot :.0f}, xmin {page.xmin:.0f}, xsep {cp.xsep:.0f}, csize {cp.csize:.0f}, xmax {page.xmax:.0f}")
    logger.info(f"Page Y params: ycell {cp.ydot:.0f}, ymin {page.ymin:.0f}")
    logger.info(f"keypoint sizes {areas}")
    # List of list of cells by line_params
    detected_lines, lines_coord = group_by_lines(kp_map, blob_coords, xydiff, page)
   
    if show_detect: 
        show_detection(image, detected_lines, cp.xdot, cp.csize, page.xmin, 400)
    
    text = ''
    lines_words = []
    total = 0
    total_errors = 0
    total_blank = 0
    word_tuples = []
    for ln, line_coor in enumerate(lines_coord):
        cells, line_params, blank_count = translate_line(line_coor, ln, page)
        cp = line_params.cell_params
        if xmin is not None:
            line_params.xmin = xmin
        
        word_tuples.append((ln, translate_cells(cells, line_params)))
        total_blank += blank_count
    
    for ln_wrd_tpl in word_tuples:
        total += len(ln_wrd_tpl[1])
        wrd_text, wrd_errors = translate_word_text_by_indexes(ln_wrd_tpl[1], ln_wrd_tpl[0])
        text += ' '.join(wrd_text)
        lines_words.append(wrd_text)
        total_errors += wrd_errors
                
        text += '\n'

    found_count = 0
    not_found_count = 0
    not_found_text = []
    found_text = []
    missing = []
    #for wr in [q for q in text.split(' ') if q not in ['', '\n']]:
    #for wr in [q for q in text.split(' ') if q not in ['', '\n']]:
    for words in lines_words:
        for wr in [w for w in words if w != '']:
            if wr in rev_abbr:
                found_text.append(wr)
                found_count += 1
            else:
                not_found_text += f"{wr}"
                not_found_count += 1                
        found_text += '\n'
        not_found_text += ' '
            
    logger.info(f"Total words: {total} (spaces: {total_blank}), total_errors: {total_errors}")
    logger.info(f"total processed words: {not_found_count + found_count}")
    logger.info(f"total found words: {found_count}\nNot found words: {not_found_count}\n")
    print(f'Translated abbreviations {"-" * 80}')
    print(' '.join(found_text))
    print(f'Incorrect translations {"-" * 80}')
    print(''.join(not_found_text))
    
#    for ra in rev_abbr:
#        if ra not in found_text:
#            missing.append(ra)
#    print(f'Missing abbreviations{"-" * 80}')
#    print('\n'.join(missing))

    print(f'All text {"-" * 80}')
    print(text)
if __name__ == '__main__':
    main()
    
