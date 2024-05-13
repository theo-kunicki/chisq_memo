import numpy as np
import numpy.ma as ma

def modified_z_scores(data):
    """
    Computes the modified z-scores for an array of data.

    This function calculates the modified z-scores for each data point in the input array, 
    which is a robust measure of outliers. The modified z-score uses the median absolute deviation 
    (MAD) instead of the standard deviation, making it more resilient to outliers in the data.

    Parameters:
    - data (numpy.ma.MaskedArray): A masked array of data points for which to calculate the modified Z-scores.

    Returns:
    - numpy.ma.MaskedArray: An array of modified z-scores for each data point in the input array.
                            If MAD is 0, returns an array of zeros with the same shape as the input.
    """

#     data = ma.masked_invalid(data)  # Mask any invalid (NaN) values
    median = ma.median(data)
    deviations = ma.abs(data - median)
    mad = ma.median(deviations)
    if mad == 0:
        return ma.zeros_like(data)
    modified_z_scores = 0.6745 * (data - median) / mad
    return modified_z_scores

def modified_z_score(point, data):
    """
    Calculates the modified z-score of a single data point given an array of data.

    This function computes the modified z-score for a specific data point against an array of data, 
    using the median absolute deviation (MAD) for robustness. It is useful for assessing how 
    anomalous a single point is within the context of a given dataset.

    Parameters:
    - point (float): The data point for which the modified z-score is to be calculated.
    - data (numpy.ma.MaskedArray): The array of data points against which the score is calculated.

    Returns:
    - float: The modified z-score of the specified data point.
    """

#     data = ma.masked_invalid(data)  # Mask any invalid (NaN) values
    median = ma.median(data)
    deviations = ma.abs(data - median)
    mad = ma.median(deviations)
    score = 0.6745 * (point - median) / mad
    return score

def value_from_modified_z_score(data, desired_z_score):
    """
    Determines the data value that corresponds to a specified modified z-score in a given dataset.

    Given an array of data and a desired modified z-score, this function calculates the data value that 
    would have that z-score in the context of the provided dataset. It utilizes the median and 
    median absolute deviation (MAD) of the dataset for the calculation, providing a robust measure 
    even in the presence of outliers.

    Parameters:
    - data (numpy.ma.MaskedArray): The array of data points used for the calculation.
    - desired_z_score (float): The modified z-score for which the corresponding data value is desired.

    Returns:
    - float: The data value that corresponds to the specified modified z-score within the dataset. 
             Returns the median of the dataset if MAD is 0.
    """
#     data = ma.masked_invalid(data)  # Mask any invalid (NaN) values
    median = ma.median(data)
    deviations = ma.abs(data - median)
    mad = ma.median(deviations)
    if mad == 0:
        return median
    value = median + (desired_z_score * mad) / 0.6745
    return value

def iteratively_flag(chisqs):
    """
    Iteratively flags outliers in chi-square datasets based on modified z-scores.

    This function applies an iterative flagging process to two chi-square datasets ('Jee' and 'Jnn'). 
    It calculates modified z-scores for each data point in the datasets and flags those with a z-score above 4.0. 
    The process repeats until no new flags are added.

    Parameters:
    - chisqs (dict): A dictionary containing two numpy masked arrays under the keys 'Jee' and 'Jnn', 
                     representing chi-square datasets of two polarizations.

    Returns:
    - dict: A dictionary with the same structure as the input, where data points have been flagged 
            (masked) if deemed outliers based on the iterative flagging process.
    """
    chisq_ee = chisqs['Jee']
    chisq_nn = chisqs['Jnn']
    last_chisq_ee = ma.zeros(chisq_ee.shape)
    last_chisq_nn = ma.zeros(chisq_nn.shape)

    while True:
        if np.all(chisq_ee.mask == last_chisq_ee.mask) and np.all(chisq_nn.mask == last_chisq_nn.mask):
            break
        last_chisq_ee = chisq_ee.copy()
        last_chisq_nn = chisq_nn.copy()
        
        xx_new_flags = (modified_z_scores(chisq_ee)>4.0).data
        yy_new_flags = (modified_z_scores(chisq_nn)>4.0).data

        chisq_ee = ma.masked_array(chisq_ee.data, mask=(chisq_ee.mask + xx_new_flags))
        chisq_nn = ma.masked_array(chisq_nn.data, mask=(chisq_nn.mask + yy_new_flags))
    return {'Jee': chisq_ee, 'Jnn': chisq_nn}

def adjacent_cells(chisqs, i, j):
    """
    Identifies unflagged (non-masked) adjacent cells around a specified cell in a chi-square dataset, 
    considering special handling near coarse-band edges or centers.

    This function takes into account the layout of coarse bands, avoiding the inclusion of cells 
    that are directly adjacent to coarse-band edges or centers. It explores left, up, right, and down 
    directions from the specified cell, extending the search if a coarse band is encountered, 
    until unflagged cells are found or edges of the dataset are reached.

    Parameters:
    - chisqs (numpy.ndarray): A numpy masked array representing a chi-square dataset.
    - i (int): The row index of the specified cell.
    - j (int): The column index of the specified cell.

    Returns:
    - list of tuples: A list containing tuples for each unflagged adjacent cell, where each tuple contains 
                      the cell's value, row index, and column index.

    Raises:
    - IndexError: If the specified indices are out of the bounds of the dataset or are negative.
    """
    if i >= chisqs.shape[0] or j >= chisqs.shape[1]:
        raise IndexError('one or both of your indices are greater than the extent of the array')
    if i < 0 or j < 0:
        raise IndexError('indices must be non-negative for this function')
        
    adjacent_cells = []

    coarse_bands = np.zeros(chisqs.shape, dtype=bool)
    for time in coarse_bands:
        for k, element in enumerate(time):
            remainder = k%32
            if remainder in [0, 1, 16, 30, 31]:
                time[k] = True

    # left
    m, n = i, j-1
    if n < 0: # we are on the leftmost edge
        pass
    elif coarse_bands[m, n]: # we are adjacent to a coarse-band edge or center
        for k in range(1,4):
            if n-k >= 0 and not coarse_bands[m, n-k]:
                n = n-k
                if not chisqs.mask[m, n]:
                    adjacent_cells.append((chisqs[m, n], m, n))
                break
    else:
        if not chisqs.mask[m, n]:
            adjacent_cells.append((chisqs[m, n], m, n))
            
    # up
    m, n = i-1, j
    if m < 0: # we are on the top
        pass
    else:
        if not chisqs.mask[m, n]:
            adjacent_cells.append((chisqs[m, n], m, n))

    # right
    m, n = i, j+1
    if n >= chisqs.shape[1]: # we are on the rightmost edge
        pass
    elif coarse_bands[m, n]: # we are adjacent to a coarse-band edge or center
        for k in range(1, 4):
            if n+k < chisqs.shape[1] and not coarse_bands[m, n+k]:
                n = n+k
                if not chisqs.mask[m,n]:
                    adjacent_cells.append((chisqs[m, n], m, n))
                break
    else:
        if not chisqs.mask[m, n]:
            adjacent_cells.append((chisqs[m, n], m, n))
        
    # down
    m, n = i+1, j
    if m >= chisqs.shape[0]: # we are on the bottom
        pass
    else:
        if not chisqs.mask[m, n]:
            adjacent_cells.append((chisqs[m, n], m, n))
    
    return adjacent_cells


def watershed(chisqs):
    """
    Flags outliers in chi-square datasets using a watershed-like algorithm, following an initial iterative 
    flagging process.

    The function starts by applying `iteratively_flag` to preliminarily flag outliers in the 'Jee' and 'Jnn' 
    datasets based on modified z-scores. It then employs a watershed-like algorithm, which iteratively 
    expands the flagging from already flagged cells to their adjacent unflagged cells based on a modified 
    z-score criterion. This expansion mimics the watershed technique in image processing, where regions grow 
    from seeded points. The process continues until no new cells meet the criteria for flagging.

    This method is particularly effective in flagging contiguous outlier regions that might not be identified 
    in the initial flagging phase due to their local context within the dataset.

    Parameters:
    - chisqs (dict): A dictionary containing two numpy masked arrays under the keys 'Jee' and 'Jnn', 
                     representing two different chi-square datasets.

    Returns:
    - dict: A dictionary with the same structure as the input, where additional data points have been 
            flagged based on the watershed-like algorithm, in addition to the initial iterative flagging. 
            This includes updates to the masks of the 'Jee' and 'Jnn' arrays to reflect the expanded 
            flagging regions.
    """
    chisqs = iteratively_flag(chisqs) # preliminary flagging by iterative z-score
    
    chisq_ee = chisqs['Jee']
    chisq_nn = chisqs['Jnn']
    
    coarse_bands = np.zeros(chisq_ee.shape)
    for time in coarse_bands:
        for k, element in enumerate(time):
            remainder = k%32
            if remainder in [0, 1, 16, 30, 31]:
                time[k] = True
    
    existing_flags_ee = chisq_ee.mask - coarse_bands
    existing_flags_nn = chisq_nn.mask - coarse_bands

    added_flags_xx = True
    while added_flags_xx:
        added_flags_xx = False
        for i, time in enumerate(chisq_ee):
            for j, element in enumerate(chisq_ee[i]):
                if existing_flags_ee[i,j] == True:
                    for cell in adjacent_cells(chisq_ee, i, j):
                        if modified_z_score(cell[0], chisq_ee) > 2:
                            # flag the point
                            if not existing_flags_ee[cell[1], cell[2]]:
                                existing_flags_ee[cell[1], cell[2]] = True
                                added_flags_xx = True

    added_flags_yy = True
    while added_flags_yy:
        added_flags_yy = False
        for i, time in enumerate(chisq_nn):
            for j, element in enumerate(chisq_nn[i]):
                if existing_flags_nn[i,j] == True:
                    for cell in adjacent_cells(chisq_nn, i, j):
                        if modified_z_score(cell[0], chisq_nn) > 2:
                            # flag the point
                            if not existing_flags_nn[cell[1], cell[2]]:
                                existing_flags_nn[cell[1], cell[2]] = True
                                added_flags_yy = True
                        

    flags_ee = existing_flags_ee
    flags_nn = existing_flags_nn
    
    return {'Jee': ma.masked_array(chisq_ee.data, mask=flags_ee+coarse_bands),
            'Jnn': ma.masked_array(chisq_nn.data, mask=flags_nn+coarse_bands)}