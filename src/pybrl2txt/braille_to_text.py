'''
Braille to text translation module

Main steps

     Braille image --> blob coordinates --> Braille cell indexes --> Unicode Braille --> text

Details

- Get a sorted `numpy` array of blob coordinates from `opencv` detected keypoints.
- Group coordinates by line.
- Find cells representing a character
- Translate cell coordinates to Braille character indexes (1,2,3 for the first column, 4,5,6 for the second).
- Translate dot indexes to Unicode Braille characters.
- Translate Unicode Braille to English text with python-louis (liblouis).

Created on Dec 31, 2024

@author: lmc
'''

import sys
from future.builtins.misc import isinstance
import logging
import yaml
import cv2
import numpy as np
import unicodedata as uc
import louis
from pybrl2txt.brl2txt_logging import addLoggingLevel
from pybrl2txt.models import Page, Line, Area
from pybrl2txt.braille_maps import BLANK, lou_languages, dots_rels, MAX_LINE_CELLS,\
    cell_specs,LANG_DFLT


logger = logging.getLogger("pybrl2txt")
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)-7s [%(name)s] %(message)s" ,
    stream=sys.stdout)
addLoggingLevel('TRACE', logging.DEBUG - 5)

uni_prefix = 'BRAILLE PATTERN DOTS-'

def get_build_detector_params(cv2_params):
    '''
    
    :param cv2_params:
    '''
    param_map = {"blobColor":  "blob_color",
                "filterByArea":  "filter_by_area",
                "filterByCircularity":  "filter_by_circularity",
                "filterByColor":  "filter_by_color",
                "filterByConvexity":  "filter_by_convexity",
                "filterByInertia":  "filter_by_inertia",
                "maxArea":  "max_area",
                "maxCircularity":  "max_circularity",
                "maxConvexity":  "max_convexity",
                "maxInertiaRatio":  "max_inertia_ratio",
                "maxThreshold":  "max_threshold",
                "minArea":  "min_area",
                "minCircularity":  "min_circularity",
                "minConvexity":  "min_convexity",
                "minDistBetweenBlobs":  "min_dist_between_blobs",
                "minInertiaRatio":  "min_inertia_ratio",
                "minRepeatability":  "min_repeatability",
                "minThreshold":  "min_threshold",
                "thresholdStep":  "threshold_step"}
    
    min_area = cv2_params['detect']['min_area']
    max_area = cv2_params['detect']['max_area']
    min_inertia_ratio = cv2_params['detect']['min_inertia_ratio']
    min_circularity = cv2_params['detect'].get('min_circularity', 0.9)
    min_convexity = cv2_params['detect'].get('min_convexity', 0.75)
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
    
#    params.minThreshold = 10;
#    params.maxThreshold = 50;
    params.minDistBetweenBlobs = cv2_params['detect'].get('min_dist_between_blobs', 5)
    params.minRepeatability = 2
    
    #logger.debug([f"{k}: {params.__getattribute__(k)}" for k in param_map])
    return params

def get_keypoints(img_path, cv2_params):
    """
    Detect Braille dots coordinates from image.
    :param img_path:
    :param cv2_params:
"""

    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    #ret,image = cv2.threshold(image,64,255,cv2.THRESH_BINARY)
    params = get_build_detector_params(cv2_params)
    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params) # Detect blobs
    keypoints = detector.detect(image)
    return keypoints, image

def show_detection(image, detected_lines, page, cv2_params):
    """
    Help to visually debug if lines are correctly detected since dots would be colored by line.
    Black dots represent not correctly detected cells/lines.
    Color will repeat every four lines.
    
    :param image:
    :param detected_lines:
    :param page:
    :param cv2_params:
"""
    
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (178,102,255)]
    while len(colors) < len(detected_lines):
        colors.extend(colors)
    # Draw detected blobs as red circles
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if cv2_params.get('basic', False):
        output_image = cv2.drawKeypoints(output_image, detected_lines, np.array([]), colors[0], cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    else:
        for i, line in enumerate(detected_lines):
            clr = colors[i]
            # Draw a vertical line between cell
            if cv2_params.get('enabled', False) and cv2_params.get('with_cell_end', False): 
                line_params = page.lines_params[i]
                yminl = int(line_params.ymin - 3)
                ymaxl = int(line_params.ymax + 3)
                
                for rc in page.ref_cells:
                    x0y0 = (int(rc[1][0] + line_params.cp.xdot), yminl)
                    x1y1 = (int(rc[1][0] + line_params.cp.xdot), ymaxl)
                    output_image = cv2.line(output_image, x0y0 , x1y1, (0, 0, 0), thickness=1)
    
            output_image = cv2.drawKeypoints(output_image, line, np.array([]), clr, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        if cv2_params.get('with_ref_dots', True)  and len(page.ref_cells) > 0:
            kpy = page.ymin - page.cp.ydot * 0.8
            clr = (255,102,178)
            rline = []
            for rc in page.ref_cells:
                kp0 = cv2.KeyPoint(rc[0][0], kpy, rc[0][1])
                kp1 = cv2.KeyPoint(rc[1][0], kpy, rc[1][1])
                rline.append(kp0)
                rline.append(kp1)
            output_image = cv2.drawKeypoints(output_image, rline, np.array([]), clr, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            
            

    logger.info("Showing detection result")
#    (h, w) = image.shape[:2]
#    new_width = 1048
#    if h < new_width:
#        aspect_ratio = h / w
#        new_height = int(new_width * aspect_ratio)
#        output_image = cv2.resize(output_image, (new_width, new_height))
#    cv2.namedWindow('Detected blobs', cv2.WINDOW_NORMAL)
#    cv2.resizeWindow('Detected blobs', new_width, new_height)
    cv2.imshow("Detected blobs", output_image)
    if cv2.waitKey(0) & 0xFF == ord('q'):  
        cv2.destroyAllWindows() 

def group_by_lines(kp_map, blob_coords, xydiff, page_params):
    """
    Group coordinates by lines.
    :param kp_map:
    :param blob_coords:
    :param xydiff:
    :param page_params:
"""
    ycell = page_params.cp.ydot
    lines_coord = []
    detected_lines = [[kp_map[blob_coords[0][0], blob_coords[0][1]]]]
    # split keypoints by lines
    for i, d in enumerate(xydiff):
        curr_pt = blob_coords[i + 1]
        if curr_pt[0] >= 714 and curr_pt[1] >= 820 and curr_pt[1] < 851:
            pass
        current_keypoint = kp_map[curr_pt[0], curr_pt[1]]
        if (d[0] < 0 and d[1] >= int(ycell * 2.1)) :
            # Add a new line
            detected_lines.append([current_keypoint])
        else:
            detected_lines[-1].append(current_keypoint)
    p0 = 0
    p1 = len(detected_lines[0])
    # split coordinates by lines
    for j, _ in enumerate(detected_lines):
        if j > 0:
            p0 = p0 + len(detected_lines[j - 1])
        p1 = p0 + len(detected_lines[j])
        line_coor = blob_coords[p0:p1]
        line_coor = line_coor[np.lexsort((line_coor[:, 1], line_coor[:, 0]))]
        lines_coord.append(line_coor)
    
    for i,d in enumerate(detected_lines):
        if len(d) != len(lines_coord[i]):
            logger.error(f"ERROR on group by. Keypoint line length {len(d)} defers from coordinates line length. {len(lines_coord[i])}")
        
        areas = np.unique(np.array([round(kp.size) for kp in d]))
        
        get_line_area_parmeters(lines_coord[i], i, page_params, areas)

    return detected_lines, lines_coord


def get_cell_dim_from_norm(xcell, xuniq_diff, dimension):
    """
    Return a list of cell dimension values taking into account
    Braille specification limits if possible.
    @see: pybrl2txt.braille_maps.cell_specs, pybrl2txt.braille_maps.dots_rels
    Parameters
    ----------
    :param xcell: cell to get dimensions for
    :param xuniq_diff: x cell distances for current line
    :dimension: possible values are: dot_size, cell_size, sep_size
    """
    
    if dimension == 'cell_sep':
        left_val = dots_rels[f'{dimension}_min'] * 0.8
        right_val = dots_rels[f'{dimension}_max'] * 1.2
    elif dimension == 'dot_size':
        left_val = xuniq_diff.min()
        right_val = xuniq_diff.min() * 1.2
    elif dimension == 'cell_size':
        left_val = dots_rels[f'{dimension}_min'] * 0.92
        right_val = dots_rels[f'{dimension}_max'] * 1.08
    return xuniq_diff[(xuniq_diff >= xcell * left_val) & (xuniq_diff < xcell * right_val)]

def is_diff_eq(x1, x2, ref):
    '''
    Distance between x1, x2 is equal(ish) to ref separation.
    Comparison is done with a margin of 2 pixels (empirical).
    :param x1:
    :param x2:
    :param ref: a cell dimension like xsep or csize
    '''
    return round(abs(x1 - x2)) >= ref - 2 and abs(x1 - x2) <= ref + 2 

def is_diff_ge(x1, x2, ref):
    '''
    Distance between x1, x2 is greater or equal than ref separation.
    :param x1:
    :param x2:
    :param ref: a cell dimension like xsep or csize
    '''
    return round(abs(x1 - x2)) >= ref

def is_diff_gt(x1, x2, ref):
    '''
    Distance between x1, x2 is greater than ref separation.
    :param x1:
    :param x2:
    :param ref: a cell dimension like xsep or csize
    '''
    return round(abs(x1 - x2)) > ref

def is_gt_le(val, lower_lim, upper_lim):
    return val > lower_lim and val <= upper_lim

def get_ref_cell_tuple(celli, xref_coords, area):
    '''
    Given cell coordinates, return a tuple with complete x cell coordinates.
    Cells with a single x coordinate are fixed taking previous cell into account.
    The second added coordinate is given a different keypoint size that could be used
    when showing detection or to distinguish potentially problematic cells.
    kpsize = 1 - regular cells with 2 different coordinates
    kpsize = 2 - regular cells with multiple coordinates but 2 useful values.
    kpsize = 3 - cells with a single dot that were fixed.
    kpsize = 5 - cells with a single dot with a guessed fix.
    :param celli:
    :param xref_coords:
    :param area:
    '''
    kpsize = 1
    # this is probably the case for most points when the image has several lines
    if len(celli) == 2:
        p0 = celli.min(), kpsize
        p1 = celli.max(), kpsize
    elif len(celli) == 1:
    # cell with single coordinate found
        if len(xref_coords) > 0:
            # dots 1,2,3, prev dots 4,5,6
            if is_diff_eq(celli.min(), xref_coords[-1][1][0], area.cp.xsep):
                p0 = celli.min(), kpsize
                p1 = celli.min() + area.cp.xdot, kpsize * 3
            # dots 4,5,6, prev dots 4,5,6
            elif is_diff_eq(celli.min(), xref_coords[-1][1][0], area.cp.csize):
                p0 = celli.min() - area.cp.xdot, kpsize * 3
                p1 = celli.min(), kpsize
            # any dots 4,5,6 prev dots 4,5,6
            elif is_diff_ge(celli.min(), xref_coords[-1][1][0], area.cp.csize):
                p1 = celli.min() + area.cp.xdot, kpsize * 3
                p0 = celli.min(), kpsize
            # catch all (empirical)
            else:
                p1 = celli.min() + area.cp.xdot, kpsize * 5
                p0 = celli.min(), kpsize
        else:
            # dots 4,5,6 in the first cell
            p0 = celli.min() - area.cp.xdot, kpsize * 3
            p1 = celli.min(), kpsize
    elif len(celli) == 0: # no point found, it's a space
        p0 = xref_coords[-1].pt[0] + area.cp.xsep + 1, kpsize * 10
        p1 = p0[0] + area.cp.xdot, kpsize * 10
    else:
        # More than 2 coordinates but just 2 useful values
        # e.g. [15.9, 24.3, 24.5, 24.6]
        p0 = celli.min(), kpsize
        p1 = celli.max(), kpsize
    return p0, p1

def get_area_parameters(coords, area_obj: Area):
    """
    Page or Line area parameters to help find cells from detected coordinates.
    
    Keypoint coordinates can be rounded but cell parameters must be not.
    Given a page with xmax=1100 and xmin=34 and expecting 40 cells per line maximum (BANA specifications):
    
    cell_size = (1100 - 34)/40 = 26.65
    Rounding to 26
    line_length_calculated = 26 * 40 = 1040
    
    A whole cell could get lost depending on image resolution
    cell_size - line_length_calculated = (1100 - 34) - 1040 = 26
    
    :param coords: page or line dots coordinates
    :param area_obj: Page or Line objects
    """
    # x,y differences between contiguous dots. Negative values mean second/third rows in a cell and the start of a line.
    # e.g.: previous point p0=(520,69), current point =(69, 140). xydiff = (-451, 71).
    # xdiff is negative, ydiff is greater than vertical cell size --> current dot is starting a line.
    xydiff = np.diff(coords, axis=0)
    
    #xcoords = np.unique(coords[coords[:,0] > 1][:,0])
    xcoords = coords[:,0]
    ycoords = np.unique(coords[:,1])
    # minimum x in the whole image
    xmin = xcoords.min()
    # max x in the whole image. Represents last dot in a line.
    xmax = xcoords.max()
    # minimum y in the whole image
    ymin = ycoords.min()
    ymax = ycoords.max()
    
    area_obj.xmin = xmin
    area_obj.xmax = xmax
    area_obj.ymin = ymin
    area_obj.ymax = ymax
    if isinstance(area_obj, Page):
        area_obj.cp.dot_size_spec = ((xmax - xmin)/MAX_LINE_CELLS)/dots_rels['cell_size_min']
        # First pass. Use a low value to allow correct line detection
        xms = 1
    else:
        xms = area_obj.cp.dot_size_spec
    
    # get X diffs for values greater than expected dot size
    xuniq_diff = np.unique(np.round(xydiff[(xydiff[:,0] >= xms * 0.8)][:,0]))
    
    # x separation between dots in a cell
    xcell = np.average(get_cell_dim_from_norm(1, xuniq_diff, 'dot_size'))
    # see dots_rels for size relationship between xdot and cell size
    # which is between 2.6 and 3.05 according to BANA. (2.4, 3) used to narrow limits a bit.
    csizes = get_cell_dim_from_norm(xcell, xuniq_diff, 'cell_size')
    
    ycell = xcell
    
    # x separation between cells
    # WARNING: rounding issues compensation - sensitive value
    #x_seps0 = xuniq_diff[xuniq_diff > xcell * 1.3]
    x_seps = get_cell_dim_from_norm(xcell, xuniq_diff, 'cell_sep')
    if len(x_seps) > 0:
        #xsep = np.average(np.unique(xuniq_diff[(xuniq_diff >= xcell * 1.3) & (xuniq_diff < xcell * 2)]))
        xsep = np.unique(x_seps).min()

    else:
        logger.warning(f"Line coordinates have weird X values. Setting xsep to {xcell *  1.4:.0f}")
        xsep = round(xcell *  1.4)
    
    area_obj.cp.xdot = xcell
    area_obj.cp.ydot = ycell
    area_obj.cp.xsep = xsep
    if len(csizes) == 0:
        # Fallback value but the other is preferred 
        area_obj.cp.csize = xcell + xsep
    else:
        area_obj.cp.csize = np.average(csizes)

    # If it's a Line, set the y-coord possible values 
    if isinstance(area_obj, Line):
        area_obj.cell_count = round((area_obj.xmax + int(area_obj.cp.xdot) - area_obj.xmin)/int(area_obj.cp.csize)) + 1
        yuniq = np.unique(np.round(ycoords[(ycoords > 1)]))
        # used by cell_to_braille_indexes_no_magic method
        area_obj.ydot14 = yuniq[0]
        if yuniq.size > 1:
            area_obj.ydot25 = yuniq[1]
        if yuniq.size > 2:
            area_obj.ydot36 = yuniq[2]
        logger.debug(f"{area_obj}")
    
    xref_coords = []
    if isinstance(area_obj, Page):
        xcoor_sort = np.unique(sorted(xcoords.copy()))
        xcs_diff = np.diff(xcoor_sort)
        # index of diff representing cell separation or greater
        xcs_diff = xcoor_sort[:-1][xcs_diff > area_obj.cp.xdot * 1.1]
        pxci = None
        for xci in xcs_diff:
            if pxci is None:
                celli = xcoor_sort[(xcoor_sort <= xci)]
            else:
                celli = xcoor_sort[(xcoor_sort > pxci) & (xcoor_sort <= xci)]
            
            p0, p1 = get_ref_cell_tuple(celli, xref_coords, area_obj)
            pxci = xci
            xref_coords.append([p0, p1])
        
        # Add remaining cell if any
        celli = xcoor_sort[(xcoor_sort > pxci)]
        if len(celli) > 0:
            p0, p1 = get_ref_cell_tuple(celli, xref_coords, area_obj)
            xref_coords.append([p0, p1])
        xref_coords = np.array(xref_coords)
        area_obj.ref_cells = xref_coords
    return area_obj, xydiff

def get_line_area_parmeters(line_coor, ln, page, areas):
    """
    Get Line Area parameters with corrections if necessary.
    
    Parameters
    ----------
    :param line_coor: ndarray
            numpy array of coordinates by line
    :param ln: int
            line number
        page: Page
    :param areas: list 
            Page Area parameters
    
    Returns
    -------
        line_params: Line
            Line object with corrected Area parameters if necessary.
        """
    
    page.lines_params[ln] = Line()
    
    line_params = page.lines_params[ln]
    line_params.line_num = ln
    line_params.cp.blob_sizes = areas
    line_params.cp.dot_size_spec =  page.cp.dot_size_spec
    line_params = get_area_parameters(line_coor, line_params)[0]
    line_params.cp.normalized = page.cp.normalized

    # Calculated cell size for the line is greater than page calculated cell size.
    # It's probably an indicator of page irregularities or bad blob detections
    if line_params.cp.csize > page.cp.csize:
        logger.warning(f"Page/Line {ln} params differ. Page : {page.cell_params}")
        logger.warning(f"Page/Line {ln} params differ. Line : {line_params.cell_params}") 
        #line_params.cell_params.csize = page.cell_params.csize
    page.lines_params[ln] = line_params
    return line_params

def cell_to_braille_indexes_no_magic(cell, page, ln, idx):
    """
    Return a sorted tuple representing dot indexes in the cell.
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
    
    
    :param cell: cell to translate
    :param page: Page object
    :param ln: int line number
    :param idx: int cell index in the line
    """
    line_params = page.lines_params[ln]
    is_cell_error = False
    cause = 'UNKNOWN'
    cell_idxs = []
    cell_idx = -1
    ydot14 = line_params.ydot14
    ydot25 = line_params.ydot25
    ydot36 = line_params.ydot36
    if len(cell) == 0:
        return cell_idxs, True
    cell_start, _ = get_cell_start_end(cell, page, ln, idx)
    cell_middle = round(cell_start + line_params.cp.xdot * 0.8)
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
                elif is_gt_le(cc[1], ydot14, ydot25):
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
        err_msg = f"idx: {idx}, cell_middle: {cell_middle}, kp: {cc}"
        if cell_idx == -1:
            cause = 'not found'
            is_cell_error = True
            logger.warning(f"Ln {ln}, cell_idx {cause}. {err_msg}")
        elif cell_idx in cell_idxs:
            cause = 'duplicate'
            is_cell_error = True
            logger.warning(f"Ln {ln}, cell_idx {cause}. {err_msg}")
        cell_idxs.append(cell_idx)
        cell_idx = -1
    
    return cell_idxs, is_cell_error

def cell_keypoints_to_braille_indexes(cell, page, ln, idx):
    """
    Cell must contain the correct points, if not, it must be fixed in the previous step.
    
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
    
    :param cell: ndarray cell to translate
    :param page: Page object
    :param ln: int line number
    :param idx: int cell index in the line
    """
    ref_cell = {(0,0): 1, (1,0): 4, # (2,0): 4,
                (0,1): 2, (1,1): 5, # (2,1): 5,
                (0,2): 3, (1,2): 6, # (2,2): 6,
                }

    line_params = page.lines_params[ln]
    if len(cell) > 6:
        logger.error(f"ERROR. Cell has more than 6 dots")
        return (-1,), True
    
    if len(cell) == 0:
        return BLANK, False
    
    # convenience checks to place breakpoints for debugging.
    if idx == 8: #and line_params.ymax > 800:
        pass
    if line_params.line_num == 4 and idx in [2,3]:
        pass
    
#    if len(cell) == 1:
#        ref = page.ref_cells[(page.ref_cells[:,:,0] <= cell[:,0])]
#    else:
#        ref = page.ref_cells[(page.ref_cells[:,:,0] >= cell[:,0]) & (page.ref_cells[:,:,0] <= cell[:,1])]
    celln = cell.copy()
    cell_start, _ = get_cell_start_end(celln, page, ln, idx)
    
    # cell normalization
    celln[:,0] -= cell_start 
    celln[:,1] -= line_params.ymin
    celln = celln[np.lexsort((celln[:, 1], celln[:, 0]))]
    celln[:,0] = np.round(np.divide(celln[:,0], line_params.cp.xdot))
    celln[:,1] = np.round(np.divide(celln[:,1], line_params.cp.ydot))

    # WARNING: Normalization/rounding issues give x max = 2
    # [[1 0][2 2]] fixed to [[0 0][1 2]]
    if celln[:,0].max() == 2:
        celln[:,0] -= 1
    
    # get dot indexes or -1 if not found
    cell_idxs = np.array([ref_cell.get((x[0],x[1]), -1)  for x in celln])
    
    logger.trace(f"dots to indexes idx: {idx}: cell_start: {cell_start}, cell: {cell}, celln{celln}, cell_idxs: {cell_idxs}")
    is_cell_error = -1 in cell_idxs
    dup_cell_error = len(cell_idxs) > len(set([x for x in cell_idxs]))
    if is_cell_error or dup_cell_error:
        cause = 'duplicate ' if dup_cell_error else ''
        logger.error(f"Line: {line_params.line_num}/{idx} - {cause}index translation error on cell: {cell} norm: {celln} -> ids: {cell_idxs}")
        logger.warning(f"Line: {line_params.line_num}/{idx} - index translation error fix attempt")
        # try to fix broken cell
        cell_idxs, is_cell_error = cell_to_braille_indexes_no_magic(cell, page, ln, idx)

    return tuple(sorted(cell_idxs)), is_cell_error or dup_cell_error

def get_cell_start_end(cell, page, ln, idx):
    """
    Calculate cell start x value given the line/page CellParameters
    and the cell index within the word.
    
    :param cell: cell to get start, end x coordinate value
    :param page: Page object
    :param ln: int line number
    :param idx: int cell index in the line
    """
    line_params = page.lines_params[ln]
    cell_start = line_params.xmin

    # Line starts with indentation and first cell may have dots on right column only
    # e.g. Line starts with capitalized letter cell (6,)
    if idx == 0 and page is not None and cell_start > page.xmin:
        num_cell_shift = round((cell_start - page.xmin)/line_params.cp.csize)
        
        if num_cell_shift >= 1:
            cell_start = page.xmin + (num_cell_shift * line_params.cp.csize)
            line_params.xmin = cell_start
        else:
            # WARNING: sensitive value
            line_params.xmin = page.xmin
            cell_start = line_params.xmin
    
    if idx > 0:
        cell_start += (line_params.cp.csize) * idx
    
    cell_end = cell_start + line_params.cp.csize
    
    if cell is not None:
        xcu = np.unique(cell[:, 0])
        cxmin = cell[:,0].min()
        #
        # Cell start fixing rules
        #    Attempt to handle: blob detection from images of unknown source, 
        #    lines of different length, irregular or too much white space between cells,
        #    unknown Braille grade, etc.
        #
        # - It's not the first cell and has 2 unique x coordinates so cell starts at the min of them
        # - Calculated cell start is less than cell min x for more than a cell size.
        if (xcu.shape[0] == 2 and idx > 0) or (cxmin - cell_start) > line_params.cp.csize:
            cell_start = cxmin
            cell_end = cell[:,0].max()
        # - Cell cannot start after x min
        if cell_start > cxmin:
            cell_start = cxmin
        # FIXME: cell cannot start more than xdot less than cxmin
        if (cxmin - cell_start) >= line_params.cp.csize:
            cell_start = cxmin - line_params.cp.csize
    
    return cell_start, cell_end

def translate_line(line_coor, ln, page):
    """
    Return array of cells in a line with BLANK inserted.
    Area parameters are recalculated with specific line values.
    
    Line coordinates are split into words and then words into cells. 
    
    :param line_coor: line dot coordinates
    :param ln: linr number
    :param page: Page object
    """
    line_params = page.lines_params[ln]
    cp = line_params.cp
    
    error_count = 0
    cells = []
    wrd_cells = []
    # IMPORTANT: line coordinates are already sorted by y then x to get cell ordered coordinates 
    line_diff = np.diff(line_coor, axis=0)
    # FIXME: offset added to fix word split. 'else' -> 'el se'
    # line coordinates differences greater than cell size. Represent end of a word
    line_wrd_idx = line_coor[:-1][line_diff[:,0] >= cp.csize * 1.5]
    # FIXME: better calculation of word delimiter
    #line_wrd_idx = line_coor[:-1][(line_diff[:,0] > line_diff[:,0][line_diff[:,0] >= cp.csize * 0.9].min())]
    # Empty index list - word distance is weird
    if len(line_wrd_idx) == 0:
        logger.error(f"Ln {ln}. No words found so Braille image is probably bad or it's a single word. Assuming it's a single word.")
        wrd_cells.append(line_coor)

    
    pwi = None
    # split line in words
    for wi in line_wrd_idx:
        if pwi is None:
            ccoor = line_coor[line_coor[:,0] <= wi[0]]
        else:
            # array of coordinates between previous and current word delimiter indexes.
            ccoor = line_coor[(line_coor[:,0] > pwi[0]) & (line_coor[:,0] <= wi[0])]
        wrd_cells.append(ccoor)
        pwi = wi
    if pwi is not None:
        # append remaining coordinates if any
        wrd_cells.append(line_coor[line_coor[:,0] > pwi[0]])
    
    # how many cells represent 1 space
    # rounded difference between first x coor of last word minus last x coord of the preceding word divide by csize
    if len(wrd_cells) >= 2:
        space_cells = round((wrd_cells[-1][:,0].min() - wrd_cells[-2][:,0].max()) / cp.xdot, 1)
        logger.trace(f"Ln: {ln}, words: {len(wrd_cells)}, space size: {space_cells} (aprox. {space_cells * cell_specs['x_dot_to_dot'][0]:.2f} mm)")
    
    is_csize_fixed = False
    last_full_cell = None
    #split words in cells
    for i, wrdc in enumerate(wrd_cells):
        # Expected cell start and end
        # WARNING: inaccuracies will impact translation- sensitive value
        cs = wrdc[:,0].min()
        
        if ln == 0 and i == 2:
            pass
        ce = cs + cp.xsep
        cn = 0
        # [[52.5  3. ] [68.   1. ]]
        # [[ 92.    1. ] [107.5   3. ]]
        refs = page.ref_cells[(page.ref_cells[:,0,0] >= wrdc[:,0].min() - cp.xsep) & (page.ref_cells[:,0,0] <= wrdc[:,0].max() + cp.xsep)]
        for rc in refs:
        #for rc in page.ref_cells[(page.ref_cells[:,0,0] <= wrdc[:,0].max())]:
            cs = rc[0][0]
            ce = rc[1][0]
        #while ce < wrdc[:,0].max() + cp.csize:
            cll = wrdc[(wrdc[:,0] >= cs) & (wrdc[:,0] <= ce)]
            # cell has dots in both columns
            has_full_row = len(np.unique(cll[:, 0])) == 2
            if len(cll) > 0:
#                if cll[:,0].max() - cll[:,0].min() > cp.xsep:
#                    logger.warning(f"Ln: {ln}. Cell {cn} in word {i} contains more keypoints than it should. (max - min) {cll[:,0].max() - cll[:,0].min()} > xsep {cp.xsep}. Fixing...")
#                    cll = wrdc[(wrdc[:,0] >= cs) & (wrdc[:,0] < cs + cp.xsep * 0.8)]
#                    has_full_row = len(np.unique(cll[:, 0])) == 2
                cells.append(cll)
                cn += 1
                # Fix cell size for small xdot distance (probably low res image) - once per line
                if cp.xdot < 9 and not is_csize_fixed and last_full_cell is not None and len(np.unique(cll[:,0])) == 2:
                    cxu = np.unique(cll[:,0])
                    nxu = np.unique(last_full_cell[:,0])
                    # 2 contiguous cell with dots in both columns
                    if len(cxu) == 2 and len(nxu) == 2:
                        new_csize = np.round(np.unique(cxu).min() - np.unique(nxu).min(), 1)
                        if new_csize > 0 and new_csize != cp.csize and new_csize < cp.csize * 1.2 and abs(new_csize - cp.csize) > 0.2:
                            logger.warning(f"Ln: {ln}, word {i}, Cell {cn} - fixed csize {cp.csize} to {new_csize}.")
                            cp.csize = new_csize
                            is_csize_fixed = True
                    
                if not is_csize_fixed and has_full_row:
                    last_full_cell = cll
            # Next cell starts at the end of the previous
#            if has_full_row:
#                cs = cll[:,0].max() + cp.xsep * 0.95
#            else:
#                cs = ce
#            ce = cs + cp.csize
        # add "space" between words
        cells.append(np.array([]))
    is_csize_fixed = False
        
    # translate cell coordinates to Braille indexes
    idxs = []
    for idx, cell in enumerate(cells):
        if line_params.cp.normalized:
            brl_idx, is_cell_error = cell_keypoints_to_braille_indexes(cell, page, ln, idx)
        else:
            brl_idx, is_cell_error = cell_to_braille_indexes_no_magic(cell, page, ln, idx)
        idxs.append(brl_idx)
        error_count +=  1 if is_cell_error else 0
        
    text, braille_uni_str, lou_err = call_louis(idxs, ln, page.lang)
    error_count += lou_err
    return text, braille_uni_str, error_count

def get_replacement_for_unknown_indexes():
    """Return a inverted question mark ¿ for not translated indexes"""
    
    unk_replace = ''
    for c in ['45', '56', '236']:
        unk_replace += uc.lookup(f'{uni_prefix}{c}')
    
    return unk_replace

def call_louis_to_Braille_ascii(braille_uni_str, lang='en-ascii'):
    '''
    Translate Uncode Braille to Braille ASCII
    :param braille_uni_str: str Unicode Braille
    :param lang: str Language to use for translation
    '''
    # to Braille ASCII
    return louis.backTranslate(lou_languages[lang], braille_uni_str, typeform=louis.emph_1)[0]

def call_louis(word_tuples, line_num, lang=LANG_DFLT):
    """
    Convert index tuples to unicode characters for the whole line,
    then supply that braille unicode text to python-louis to get translated text.
    
    Returns '¿' for each index if word contains at least one -1 index.
    
    For a custom path to tables set LOUIS_TABLEPATH
    
    :param word_tuples: ndarray word coordinates tuples
    :param line_num: line number
    :param lang: str Language to use for translation
    """
    
    #TODO: build dict of available languages dynamically.
    err_count = 0
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
            logger.trace(f"Line {line_num}: cell: {wrd}, unicode : {uni_name}")
        except Exception as e:
            logger.debug(f"Line {line_num}: Index to unicode conversion.: {e}, cell: {wrd}", exc_info=False)
            # Add an inverted question mark (¿)for not found unicode characters.
            braille_uni_str += get_replacement_for_unknown_indexes()
            err_count += 1
    logger.trace(f"Line {line_num}: Braille unicode text: {braille_uni_str}")
    # louis.dotsIO|louis.ucBrl
    # setting mode=louis.noUndefined will remove \<numbers>/ errors from translation
    # but the error count would be lost
    lou_transl = louis.backTranslate(lou_languages[lang], braille_uni_str)
    
    # translated string contains an untranslated sequence like \45/
    # just one error per line is counted even if there are more
    eib = lou_transl[0].find('\\')
    eis = lou_transl[0].find('/')
    if eib != -1 and eis != -1:
        if lou_transl[0][eib+1: eis].isdigit():
            err_count += 1
            
        
    return lou_transl[0], braille_uni_str, err_count

def  parse_keypoints(config, keypoints):
    """
    Translation main process. Parse coordinates to obtain cell dot indexes and translate those to text_lines.
    
    Parameters
    ----------
    :param config: dict
        Configuration object

    :param keypoints: object
        keypoints detected by opencv

    Returns
    -------
    text_lines : float
        list of text by lines
    braille_lines : float
        list of braille unicode by lines
    total_error: int
        errors count

    detected_lines: list
        list of opencv keypopints by line
    page: Page
        Page area parameters
    all_lines_params: list
        list of Line area parameters
    """
    
    round_to = config['parse']['round_to']
    lang = config['parse'].get('lang',LANG_DFLT)
    xmin = config['parse'].get('xmin')
    normalized = config['parse'].get('normalized', True)
    dot_min_sep = config['parse'].get('dot_min_sep')
    
    detected_lines = None
    text_lines = []
    braille_lines = []
    ln = -1
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
        
        page= Page()
        page.cp.blob_sizes = areas
        page, xydiff = get_area_parameters(blob_coords, page)
        cp = page.cp
        # FIXME: force xmin from configuration file
        if xmin is not None:
            page.xmin = xmin
        if not normalized:
            cp.normalized = normalized
        if dot_min_sep is not None:
            cp.dot_min_sep = dot_min_sep
        if (page.xmax - page.xmin)/ cp.csize > MAX_LINE_CELLS:
            logger.warning("More than 40 cells per line could be found exceeding the recommended 40.")
        page.lang = lang
        logger.info(f"Detected blobs: {len(blob_coords)}, max cells per line: {(page.xmax - page.xmin)/cp.csize:.0f}")
        logger.info(f"Page X params: xcell {cp.xdot :.2f}, xmin {page.xmin:.0f}, xsep {cp.xsep:.2f}, csize {cp.csize:.2f}, xmax {page.xmax:.0f}")
        logger.info(f"Page Y params: ycell {cp.ydot:.2f}, ymin {page.ymin:.0f}")
        logger.info(f"keypoint sizes {areas}")
        
        # List of list of cells by line_params
        detected_lines, lines_coord = group_by_lines(kp_map, blob_coords, xydiff, page)
        
        # lines to cell by index
        for ln, line_coor in enumerate(lines_coord):
            lntext, lntxt_braille, err_cnt = translate_line(line_coor, ln, page)
            
            text_lines.append(lntext)
            braille_lines.append(lntxt_braille)
            total_errors += err_cnt
    
    except Exception as e:
        logger.error(f"Critical error while parsing lines: {e}", exc_info=logger.isEnabledFor(logging.DEBUG))
        return [], 1, [], None, None
    
    return text_lines,braille_lines,total_errors, detected_lines, page


def parse_image_file(img_path, config):
    '''
    Parse image given its path.
    
    :param img_path: str Path to the image
    :param config: dict Configuration dictionary
    '''
    if config.get('cfg') is not None:
        logging_level = logging.getLevelName(config.get('cfg').get('logging_level', 'INFO').upper())
        logger.setLevel(logging_level)
    
    keypoints, image = get_keypoints(img_path, config['cv2_cfg'])
    text_lines, braille_lines, total_errors, detected_lines, page = parse_keypoints(config, keypoints)
    if config['cv2_cfg']['show_detect']['enabled']:
        if config['cv2_cfg']['show_detect'].get('basic', False):
            show_detection(image, keypoints, page, config['cv2_cfg']['show_detect'])
        else:
            show_detection(image, detected_lines, page, config['cv2_cfg']['show_detect'])
            
    return text_lines, braille_lines, total_errors

def main(args):
    """Main standard method.
    """

    base_dir = ''
    cfg_path = ''
    image_path = ""

    if len(args) >= 4:
        base_dir = args[1]
        cfg_path = args[2]
        image_path = args[3]
    else:
        base_dir = '../../tests/resources'
        cfg_path = '../resources/abbreviations.yml'
        image_path = "camel-case.png"
    
    
    img_path = f"{base_dir}/{image_path}"
    
    with open(cfg_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    
    logging_level = logging.getLevelName(config.get('cfg').get('logging_level', 'INFO').upper())
    logger.setLevel(logging_level)
    
    grade = config['grade']
    lang = config['parse'].get('lang', LANG_DFLT)
    logger.info(f"Starting '{lang}' Grade {grade} braille to text_lines translation")
    logger.info(f"Image file: {img_path.split('/')[-1]}, config: {cfg_path}")
    
    text, braille_lines, total_errors = parse_image_file(img_path, config)
    
    logger.info(f"Total_errors: {total_errors}")

    print(f'\n{"-" * 80}')
    print('\n'.join(text))
    
#    print(f'\n{"-" * 80}')
#    for brl_ln in braille_lines:
#        #print(call_louis_to_Braille_ascii(brl_ln), f"{lang}-ascii")
#        print(brl_ln)
if __name__ == '__main__':
    main(sys.argv)
    
