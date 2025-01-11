'''
Created on Dec 31, 2024

@author: lmc
'''

import sys
import yaml
import cv2
import numpy as np
import unicodedata as uc
import louis
from pybrl2txt.models import Page, Line, Area
from pybrl2txt.braille_maps import BLANK
from future.builtins.misc import isinstance
import logging

logger = logging.getLogger("pybrl2txt")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)-7s [%(name)s] %(message)s" ,
    stream=sys.stdout)
uni_prefix = 'BRAILLE PATTERN DOTS-'

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
    for j, _ in enumerate(detected_lines):
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
    xydiff = np.diff(coords, axis=0)
    
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
    xsep = np.unique(xydiff[(xydiff[:,0] > xcell)][:,0]).min()
    
    area_obj.cell_params.xdot = xcell
    area_obj.cell_params.ydot = ycell
    area_obj.cell_params.xsep = xsep
    area_obj.cell_params.csize = round(xcell + xsep)
    
    # If it's a Line, set the y-coord possible values 
    if isinstance(area_obj, Line):
        area_obj.cell_count = round((area_obj.xmax + area_obj.cell_params.xdot - area_obj.xmin)/area_obj.cell_params.csize) + 1
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


def cell_to_braille_indexes_no_magic(cell, line_params, idx):
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
    cell_idxs = []
    cell_idx = -1
    ydot14 = line_params.ydot14
    ydot25 = line_params.ydot25
    ydot36 = line_params.ydot36
    if len(cell) == 0:
        return cell_idxs, True
    cell_start, _ = get_cell_start_end(None, line_params, idx)
    cell_middle = round(cell_start + line_params.cell_params.xdot * 0.8)
    xcol123 = cell[:, 0].min()
    xcol456 = cell[:, 0].max()
    ycol14_25 = cell[:, 1].min()
    ycol25_36 = cell[:, 1].max()
    if xcol123 != xcol456:
        cell = cell[np.lexsort((cell[:, 0], cell[:, 1]))]
        xucell = np.unique(cell[:, 0])
        cell_middle = xucell[0] + round((xucell[1] - xucell[0]) / 2)
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
    
    return cell_idxs, is_cell_error

def cell_keypoints_to_braille_indexes(cell, line_params, idx):
    """Cell must contain the correct points, if not, it must be fixed in the previous step.
    
    Cell is normalized using cell and line parameters: ``xmin, ymin, xdot and ydot``.
    First, substract xmin and ymin from corresponding cell coordinates. Then, divide
    those values by x dot and y dot distances.
    e.g.:
    xmin = 190
    ymin = 42
    xdot = ydot = 12
    
    [[190.  42.]
     [190.  54.]
     [190.  65.]
     [202.  65.]]
     
     Magic! normalized values should match a key on ref_cell map!!!
     Normalized values should return 0 or 1 for x and 0, 1, 2 for y.
     
     [[0. 0.]
      [0. 1.]
      [0. 2.]
      [1. 2.]]
    
    **cell_to_braille_indexes_no_magic** method can be tried instead if too many errors occur.
    """
    ref_cell = {(0,0): 1, (1,0): 4, # (2,0): 4,
                (0,1): 2, (1,1): 5, # (2,1): 5,
                (0,2): 3, (1,2): 6, # (2,2): 6,
                }

    if len(cell) > 6:
        logger.error(f"ERROR. Cell has more than 6 dots")
        return (-1,), True
    
    if len(cell) == 0:
        return BLANK, False
    
    # convenience checks to place breakpoints for debugging.
    if idx == 0: #and line_params.ymax > 800:
        pass
    if line_params.line_num == 0 and idx in [3, 6, 11]:
        pass
    
    cell_start, _ = get_cell_start_end(None, line_params, idx)
    
    celln = cell.copy()
    xcu = np.unique(celln[:, 0])
    cxmin = celln[:,0].min()
    #
    # Cell start fixing rules
    #    Attempt to handle: blob detection from images of unknown source, 
    #    lines of different length, irregular or to much white space between cells,
    #    unknown Braille grade, etc.
    #
    # - It's not the first cell and has 2 unique x coordinates so cell starts at the min of them
    # - Calculated cell start is less than cell min x for more than a cell size.
    if (len(xcu) == 2 and idx > 0) or (cxmin - cell_start) > line_params.cell_params.csize:
        cell_start = cxmin
    # - Cell cannot start after x min
    if cell_start > cxmin:
        cell_start = cxmin
    
    # cell normalization
    celln[:,0] -= cell_start 
    celln[:,1] -= line_params.ymin
    celln = celln[np.lexsort((celln[:, 1], celln[:, 0]))]
    celln[:,0] = np.round(np.divide(celln[:,0], line_params.cell_params.xdot))
    celln[:,1] = np.round(np.divide(celln[:,1], line_params.cell_params.ydot))

    # get dot indexes or -1 if not found
    cell_idxs = [ref_cell.get((x[0],x[1]), -1)  for x in celln]
    
    is_cell_error = -1 in cell_idxs
    dup_cell_error = len(cell_idxs) > len(set([x for x in cell_idxs]))
    if is_cell_error or dup_cell_error:
        cause = 'duplicate ' if dup_cell_error else ''
        logger.error(f"Line: {line_params.line_num}/{idx} - {cause}index translation error on\ncell: {cell}\nnorm: {celln} -> ids: {cell_idxs}")

    #cell_idxs, is_cell_error = cell_to_braille_indexes_no_magic(cell, line_params, cell_start, idx)
    return tuple(sorted(cell_idxs)), is_cell_error or dup_cell_error

def get_cell_start_end(page, line_params, idx):
    cell_start = line_params.xmin

    if idx == 0 and page is not None and cell_start > page.xmin:
        # WARNING: sensitive value
        line_params.xmin = page.xmin
        cell_start = line_params.xmin
    if idx > 0:
        cell_start += (line_params.cell_params.csize) * idx
    
    cell_end = cell_start + line_params.cell_params.csize
    
    return cell_start, cell_end

def translate_line(line_coor, ln, page):
    """Return array of cells in a line with BLANK inserted.
    Area parameters are recalculated with specific line values.
    """
    line_params = Line()
    line_params.line_num = ln
    line_params = get_area_parameters(line_coor, line_params)[0]
    cp = line_params.cell_params
    cp.normalized = page.cell_params.normalized
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
        logger.warning(f"Line/Page params differ. Page : {page.cell_params}")
        logger.warning(f"Line/Page params differ. Line : {line_params.cell_params}")
        #line_params.cell_params.csize = page.cell_params.csize
    
    text = ''
    error_count = 0
    cells = []
    wrd_cells = []
    line_coor = line_coor[np.lexsort((line_coor[:, 1], line_coor[:, 0]))]
    line_diff = np.diff(line_coor, axis=0)
    # FIXME: offset added to fix word split. 'else' -> 'el se'
    line_wrd_idx = line_coor[:-1][line_diff[:,0] > cp.csize * 1.1]
    pwi = None
    # split line in words
    for wi in line_wrd_idx:
        if pwi is None:
            wrd_cells.append(line_coor[line_coor[:,0] <= wi[0]])
        else:
            wrd_cells.append(line_coor[(line_coor[:,0] > pwi[0]) & (line_coor[:,0] <= wi[0])])
        pwi = wi
    wrd_cells.append(line_coor[line_coor[:,0] > pwi[0]])
    
    #split words in cells
    for i, wrdc in enumerate(wrd_cells):
        cs = wrdc[:,0].min()
        ce = cs + cp.xsep
        # FIXME extend the limit pixel to cover rounding problems.
        while ce < wrdc[:,0].max() + cp.csize + 1:
            cll = wrdc[(wrdc[:,0] >= cs) & (wrdc[:,0] < ce)]
            cs = ce
            ce = cs + cp.csize
            if len(cll) > 0:
                cells.append(cll)
        # add "space" between words
        cells.append([])
        pass
        
    # translate coordinates to Braille indexes
    idxs = []
    for idx, cell in enumerate(cells):
        if line_params.cell_params.normalized:
            brl_idx, is_cell_error = cell_keypoints_to_braille_indexes(cell, line_params, idx)
        else:
            brl_idx, is_cell_error = cell_to_braille_indexes_no_magic(cell, line_params, idx)
        idxs.append(brl_idx)
        error_count +=  1 if is_cell_error else 0
        
    text += call_louis(idxs, ln, page.lang)
    return text, line_params, error_count

def get_replacement_for_unknown_indexes():
    """Return a inverted question mark ¿ for not translated indexes"""
    
    unk_replace = ''
    for c in ['45', '56', '236']:
        unk_replace += uc.lookup(f'{uni_prefix}{c}')
    
    return unk_replace

def call_louis(word_tuples, line_num, lang='en'):
    """Convert index tuples to unicode characters for the whole line,
    then supply that braille unicode text to python-louis to get translated text.
    
    Returns '¿' for each index if word contains at least one -1 index.
    
    For a custom path to tables set LOUIS_TABLEPATH
    """
    
    #TODO: build dict of available languages dynamically.
    languages = {'es': 'es-g2.ctb', 'en': 'en-ueb-g2.ctb', 'fr': 'fr-bfu-g2.ctb'}
    logger.debug(f"Locale table: {languages[lang]}")
    braille_uni_str = ''
    for w, wrd in enumerate(word_tuples):
        if -1 in wrd:
            braille_uni_str += get_replacement_for_unknown_indexes() * len(wrd)
            continue
        res = ''.join(sorted([str(idx) for idx in wrd]))
        
        try:
            uni_name = f'{uni_prefix}{res}'
            if wrd == (0,):
                uni_name = 'BRAILLE PATTERN BLANK'
            braille_uni_str += uc.lookup(uni_name)
            #logger.debug(f"Line {line_num}: cell: {wrd}, unicode : {uni_name}", exc_info=False)
        except Exception as e:
            logger.debug(f"Line {line_num}: Index to unicode conversion.: {e}, cell: {wrd}", exc_info=False)
            # Add an inverted question mark (¿)for not found unicode characters.
            braille_uni_str += get_replacement_for_unknown_indexes()
    #logger.debug(f"Line {line_num}: Braille unicode text: {braille_uni_str}")
    lou_transl = louis.backTranslate([languages[lang]], braille_uni_str)
    return lou_transl[0]

def parse_image_file(cfg_path, img_path):
    with open(cfg_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
        grade = config['grade']
        round_to = config['parse']['round_to']
        lang = config['parse'].get('lang', 'en')
        xmin = config['parse'].get('xmin')
        normalized = config['parse'].get('normalized', True)
        dot_min_sep = config['parse'].get('dot_min_sep')
        show_detect = config['cv2_cfg']['detect']['show_detect']
        if config.get('cfg') is not None:
            logging_level = logging.getLevelName(config.get('cfg').get('logging_level', 'INFO').upper())
            logger.setLevel(logging_level)
    logger.info(f"Starting '{lang}' Grade {grade} braille to text translation of {img_path}")
    keypoints, image = get_keypoints(img_path, config)
    
    text = ''
    ln = -1
    total = 0
    total_errors = 0
    try:
        # map of keypoints coordinates to keypoints
        kp_map = { (round(kp.pt[0], round_to), round(kp.pt[1], round_to)): kp for kp in keypoints}
        areas = np.unique(np.array([round(kp.size) for kp in keypoints]))
        areas_diff = np.diff(areas)
        if areas_diff.size > 0 and areas_diff.max() >= config['cv2_cfg']['detect']['min_area'] * 0.5:
            logger.warning(f"Too many blob sizes detected. Cell detection will probably be poor or bad. Sizes: {areas}")
        
        # all dots coordinates, sorted to help find lines.
        blob_coords = np.array(list(kp_map.keys()))
        blob_coords = blob_coords[np.lexsort((blob_coords[:, 0], blob_coords[:, 1]))]
        
        page, xydiff = get_area_parameters(blob_coords, Page())
        cp = page.cell_params
        if xmin is not None:
            page.xmin = xmin
        if not normalized:
            cp.normalized = normalized
        if dot_min_sep is not None:
            cp.dot_min_sep = dot_min_sep
        if page.xmax / cp.csize > 40:
            logger.warning("More than 40 cells per line could be found exceeding the recommended 40.")
        page.lang = lang
        logger.info(f"Detected blobs: {len(blob_coords)}, max cells per line: {(page.xmax - page.xmin)/cp.csize:.0f}")
        logger.info(f"Page X params: xcell {cp.xdot :.0f}, xmin {page.xmin:.0f}, xsep {cp.xsep:.0f}, csize {cp.csize:.0f}, xmax {page.xmax:.0f}")
        logger.info(f"Page Y params: ycell {cp.ydot:.0f}, ymin {page.ymin:.0f}")
        logger.info(f"keypoint sizes {areas}")
        
        # List of list of cells by line_params
        detected_lines, lines_coord = group_by_lines(kp_map, blob_coords, xydiff, page)
        
        if show_detect:
            show_detection(image, detected_lines, cp.xdot, cp.csize, page.xmin, 400)
        
        # lines to cell by index
        for ln, line_coor in enumerate(lines_coord):
            lntext, _, err_cnt = translate_line(line_coor, ln, page)
            text += lntext
            total_errors += err_cnt
            text += '\n'
    
    except Exception as e:
        logger.error(f"Critical error while parsing line {ln}: {e}", exc_info=logger.isEnabledFor(logging.DEBUG))
    
    return text, total, total_errors

def main(args):
    logging.basicConfig(level=logging.DEBUG)
    
    base_dir = ''
    cfg_path = ''
    image_path = ""
    #image_path = "camel-case.png"

    if len(args) >= 4:
        base_dir = args[1]
        cfg_path = args[2]
        image_path = args[3]
    else:
        base_dir = '../../tests/resources'
        cfg_path = '../resources/abbreviations.yml'
        #    image_path = "abbreviations.png"
        image_path = "camel-case.png"
    
    
    img_path = f"{base_dir}/{image_path}"
    
    text, total, total_errors = parse_image_file(cfg_path, img_path)

    logger.info(f"Total words: {total}, total_errors: {total_errors}")

    print(f'\n{"-" * 80}')
    print(text)
if __name__ == '__main__':
    main(sys.argv)
    
