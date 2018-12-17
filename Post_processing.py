import numpy as np
from scipy import ndimage
'''Post Processing applied to our analysis'''

def create_matrix(label):
    '''This function transforms a list of label into a matrix of label'''
    
    label = np.array(label)
    label_per_line = int(np.sqrt(label.shape))
    matrix_label = label.reshape((label_per_line, label_per_line),order='F')
    
    return matrix_label

def create_list(matrix_label):
    '''This function transforms a matrix of label into a list of label'''
    
    # Create the list
    list_label = (matrix_label.T).tolist()
    # Flatten the lists
    label = [y for x in list_label for y in x]
    
    return label

def complete_lines(label,threshold):
    ''' The function controls for each columns and rows the number of road squares. 
        If the number is large (>= threshold), the entire column/row is labeled as ROAD.
        
        INPUT: list of label, the threshold
        OUTPUT: the new list of label'''
    
    # Create a matrix of label
    matrix_label = create_matrix(label)
    
    # Column with more than threshold ones are considered as ROAD
    matrix_label[:,np.where(matrix_label.sum(axis=0)>=threshold)[0]] = 1
  
    
    # Rows with more than threshold ones are considered as ROAD
    matrix_label[np.where(matrix_label.sum(axis=1)>=threshold)[0],:] = 1
    
    # Create the list
    label = create_list(matrix_label)
    
    return label


def remove_isolated_connected_component(label,size_min):
    ''' The function detects the connected components. 
        If a component of 1s has a size smaller than size_min, it is entirely set to 0.
        
        INPUT: list of label, the size_min
        OUTPUT: the new list of label'''
    
    #Create a matrix of label
    matrix_label = create_matrix(label)
    
    # now identify the objects and remove those above a threshold
    Zlabeled,Nlabels = ndimage.measurements.label(matrix_label)
    label_size = [(Zlabeled == label).sum() for label in range(Nlabels + 1)]
    
    # now remove the labels
    for label,size in enumerate(label_size):
        if size < size_min:
            matrix_label[Zlabeled == label] = 0
    
    # Create the list
    label = create_list(matrix_label)
    
    return label



def complete_lines_almostfull(label,max_zeros):
    ''' The function controls for each non-road square its neighbors. 
        If they are classified as ROAD with a certain pattern, the considered square is labeled as ROAD.
        
        INPUT: List of labels, the max_zeros
        OUTPUT: New list of labels'''
    
    
    # Create a matrix of label
    matrix_label = create_matrix(label)   
   
    # Fix columns
    rows,columns = matrix_label.shape
    for column in range(columns):
        count = 0
        start = 0
        end = 0
        for row in range(rows):
            if (matrix_label[row,column] == 1) and (start ==0):
                start = 1
            elif (matrix_label[row,column] == 1) and (start ==1) and (count>0):
                end = 1
            elif (matrix_label[row,column] == 0) and (start ==1) and (end==0):
                count = count + 1
            
            if end ==1:
                if count < max_zeros:
                    matrix_label[row-count:row,column] = 1
                start = 1
                end = 0
                count = 0
    
    # Fix rows
    for row in range(rows):
        count = 0
        start = 0
        end = 0
        for column in range(columns):
            if (matrix_label[row,column] == 1) and (start ==0):
                start = 1
            elif (matrix_label[row,column] == 1) and (start ==1) and (count>0):
                end = 1
            elif (matrix_label[row,column] == 0) and (start ==1) and (end==0):
                count = count + 1
            
            if end ==1:
                if count < max_zeros:
                    matrix_label[row,column-count:column] = 1
                start = 1
                end = 0
                count = 0
    
    
    # Create the list
    label = create_list(matrix_label)
    
    return label



def clean_garbage_vert(label,max_distance, size_image):
    ''' The function controls for each column, entirely labeled as road, its neighbors. 
        If they are classified as noisy roads (SEE THE CODE FOR A BETTER UNDERSTANDING) they are set to 0
        
        INPUT: List of labels, the max_distance to be considered for the neighbors, the size of the considered image
        OUTPUT: New list of labels'''
    
    # Create a matrix of label
    matrix_label = create_matrix(label)   
    
    # Column with all one values
    full_columns = np.where(matrix_label.sum(axis=0) == size_image)[0]
    for column in full_columns:   
        if (column < max_distance) and (matrix_label[:,column+1].sum(axis=0) < size_image):
            count = matrix_label[:,column+1:column+max_distance+1].sum(axis=1)
            for k in range(count.shape[0]):
                if count[k] < max_distance:
                    matrix_label[k,column+1:column+max_distance] = 0
        
        elif (column > size_image - max_distance) and (matrix_label[:,column-1].sum(axis=0) < size_image):
            count = matrix_label[:,column-max_distance:column].sum(axis=1)
            for k in range(count.shape[0]):
                if count[k] < max_distance:
                    matrix_label[k,column-max_distance:column] = 0
        
        elif (column >= max_distance) and (column <= size_image - max_distance):
            if matrix_label[:,column+1].sum(axis=0) < size_image:
                count = matrix_label[:,column+1:column+max_distance+1].sum(axis=1)
                for k in range(count.shape[0]):
                    if count[k] < max_distance:
                        matrix_label[k,column+1:column+max_distance] = 0
        
            if matrix_label[:,column-1].sum(axis=0) < size_image:            
                count = matrix_label[:,column-max_distance:column].sum(axis=1)
                for k in range(count.shape[0]):
                    if count[k] < max_distance:
                        matrix_label[k,column-max_distance:column] = 0
        
  
    # Create the list
    label = create_list(matrix_label)
    
    return label    


def clean_garbage_hor(label,max_distance, size_image):
    ''' The function controls for each row, entirely labeled as road, its neighbors. 
        If they are classified as noisy roads (SEE THE CODE FOR A BETTER UNDERSTANDING) they are set to 0
        
        INPUT: List of labels, the max_distance to be considered for the neighbors, the size of the considered image
        OUTPUT: New list of labels'''
    
    # Create a matrix of label
    matrix_label = create_matrix(label)
    
    # Column with all one values
    full_rows = np.where(matrix_label.sum(axis=1) == size_image)[0]
    for row in full_rows:   
        if (row < max_distance) and (matrix_label[row+1,:].sum() < size_image):
            count = matrix_label[row+1:row+max_distance+1,:].sum(axis=0)
            for k in range(count.shape[0]):
                if count[k] < max_distance:
                    matrix_label[row+1:row+max_distance,k] = 0
        
        elif (row > size_image - max_distance) and (matrix_label[row-1,:].sum() < size_image):
            count = matrix_label[row-max_distance:row,:].sum(axis=0)
            for k in range(count.shape[0]):
                if count[k] < max_distance:
                    matrix_label[row-max_distance:row,k] = 0
        
        elif (row >= max_distance) and (row <= size_image - max_distance):
            if matrix_label[row+1,:].sum() < size_image:
                count = matrix_label[row+1:row+max_distance+1,:].sum(axis=0)
                for k in range(count.shape[0]):
                    if count[k] < max_distance:
                        matrix_label[row+1:row+max_distance,k] = 0
        
            if matrix_label[row-1,:].sum() < size_image:            
                count = matrix_label[row-max_distance:row,:].sum(axis=0)
                for k in range(count.shape[0]):
                    if count[k] < max_distance:
                        matrix_label[row-max_distance:row,k] = 0
        
    
    # Create the list
    label = create_list(matrix_label)
    
    return label 


def complete_lines_almostfull(label,max_zeros):
    ''' The function controls for each non-road square its neighbors. 
        If they are classified as ROAD with a certain pattern, the considered square is labeled as ROAD.
        
        INPUT: List of labels, the max_zeros
        OUTPUT: New list of labels'''
    
    
    # Create a matrix of label
    matrix_label = create_matrix(label)   
   
    # Fix columns
    rows,columns = matrix_label.shape
    for column in range(columns):
        count = 0
        start = 0
        end = 0
        for row in range(rows):
            if (matrix_label[row,column] == 1) and (start ==0):
                start = 1
            elif (matrix_label[row,column] == 1) and (start ==1) and (count>0):
                end = 1
            elif (matrix_label[row,column] == 0) and (start ==1) and (end==0):
                count = count + 1
            
            if end ==1:
                if count < max_zeros:
                    matrix_label[row-count:row,column] = 1
                start = 1
                end = 0
                count = 0
    
    # Fix rows
    for row in range(rows):
        count = 0
        start = 0
        end = 0
        for column in range(columns):
            if (matrix_label[row,column] == 1) and (start ==0):
                start = 1
            elif (matrix_label[row,column] == 1) and (start ==1) and (count>0):
                end = 1
            elif (matrix_label[row,column] == 0) and (start ==1) and (end==0):
                count = count + 1
            
            if end ==1:
                if count < max_zeros:
                    matrix_label[row,column-count:column] = 1
                start = 1
                end = 0
                count = 0
    
    
    # Create the list
    label = create_list(matrix_label)
    
    return label


def clean_garbage_vert(label,max_distance, size_image):
    ''' The function controls for each column, entirely labeled as road, its neighbors. 
        If they are classified as noisy roads (SEE THE CODE FOR A BETTER UNDERSTANDING) they are set to 0
        
        INPUT: List of labels, the max_distance to be considered for the neighbors, the size of the considered image
        OUTPUT: New list of labels'''
    
    # Create a matrix of label
    matrix_label = create_matrix(label)   
    
    # Column with all one values
    full_columns = np.where(matrix_label.sum(axis=0) == size_image)[0]
    for column in full_columns:   
        if (column < max_distance) and (matrix_label[:,column+1].sum(axis=0) < size_image):
            count = matrix_label[:,column+1:column+max_distance+1].sum(axis=1)
            for k in range(count.shape[0]):
                if count[k] < max_distance:
                    matrix_label[k,column+1:column+max_distance] = 0
        
        elif (column > size_image - max_distance) and (matrix_label[:,column-1].sum(axis=0) < size_image):
            count = matrix_label[:,column-max_distance:column].sum(axis=1)
            for k in range(count.shape[0]):
                if count[k] < max_distance:
                    matrix_label[k,column-max_distance:column] = 0
        
        elif (column >= max_distance) and (column <= size_image - max_distance):
            if matrix_label[:,column+1].sum(axis=0) < size_image:
                count = matrix_label[:,column+1:column+max_distance+1].sum(axis=1)
                for k in range(count.shape[0]):
                    if count[k] < max_distance:
                        matrix_label[k,column+1:column+max_distance] = 0
        
            if matrix_label[:,column-1].sum(axis=0) < size_image:            
                count = matrix_label[:,column-max_distance:column].sum(axis=1)
                for k in range(count.shape[0]):
                    if count[k] < max_distance:
                        matrix_label[k,column-max_distance:column] = 0
        
  
    # Create the list
    label = create_list(matrix_label)
    
    return label    


def complete_semilines(label,threshold, size_image):
    ''' The function takes a column entirely labeled as row. Then for each row, the function splits that row
        in a left row and right row (with respect to the previous column). Then if that "subrow" is sufficiently 
        labeled as road (> a percentage of the total length of that "subrow") it is entirely classified as road
        
        INPUT: list of label, the percentage of the total length used as threshold, the size of the image
        OUTPUT: the new list of label'''
    
    # Create a matrix of label
    matrix_label = create_matrix(label)
    
    # Rows with all one values
    full_rows = np.where(matrix_label.sum(axis=1) == size_image)[0]
    for row in full_rows:   
        for column in range(size_image):
            if matrix_label[row+1:,column].sum() > np.abs(size_image-row)*threshold :
                matrix_label[row+1:,column] = 1
            if matrix_label[:row-1,column].sum() > np.abs(size_image-row)*threshold :
                matrix_label[:row-1,column] = 1
    
    # Columns with all one values
    full_columns = np.where(matrix_label.sum(axis=0) == size_image)[0]
    for column in full_columns:   
        for row in range(size_image):
            if (column < size_image - 1) and (matrix_label[row,column+1:].sum() > np.abs(size_image-column)*threshold):
                matrix_label[row,column+1:] = 1
            if (column > 0) and (matrix_label[row,:column-1].sum() > np.abs(size_image-column)*threshold) :
                matrix_label[row,:column-1] = 1
                
    
    # Create the list
    label = create_list(matrix_label)
    
    return label

def post_processing(label,threshold,size_min,verbarg,horbarg,size_image):
    label = complete_lines(label,threshold)
    label = remove_isolated_connected_component(label,size_min)
    label = clean_garbage_vert(label,verbarg,size_image)
    label = clean_garbage_hor(label,horbarg,size_image)
    label = remove_isolated_connected_component(label,size_min)
    #label = complete_semilines(label,0.85, size_image)
    return label

def remove_border(imgs,new_size):
    old_size = imgs.shape[0]
    remove = int((old_size - new_size)/2)
    imgs = imgs.reshape(old_size,old_size)
    new_img = imgs[remove:-remove,remove:-remove]
    return new_img