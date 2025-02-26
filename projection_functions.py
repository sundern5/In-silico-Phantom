import numpy as np
from tqdm import tqdm
from skimage.transform import iradon
import cv2 as cv

def create_sinogram(nodes_2d, det_spacing=0.1, det_width=10, det_step_ang=1, det_radius=100, det_origin=None):
    """radon transform.

    Create a sinogram from a set of 2D nodes (For 2D meshes)
    Assumes parallel beams
    Parameters
    ----------
    nodes_2d : n x 2 array
        Numpy array containing all node points of mesh within a certain z range
        Z range should not be provided
    det_spacing: optional
        space between consecutive detectors (Effective pixel/voxel size)
    det_width: optional
        total length of detector / resulting image
    det_step_ang :  optional
        Angle movement in degrees between acquisitions
    det_radius : optional
        Radius about the center of mass of scanning object
    -------    
    Returns
    -------
    sinogram: ndarray
        returns sinogram array
        rows - number of pixels of the final image
        columns - number of angle steps
    """

    if(np.shape(nodes_2d)[1]!=2):
        raise ValueError('The input node array must have 2 columns')
    
    bin_edges = np.arange(start=0,stop=det_width+det_spacing,step=det_spacing)  ## same as detector points
    det_steps = int(180/det_step_ang)       ## number of acquisitions per rotation

    if(det_origin is None):
        det_origin = np.mean(nodes_2d,axis=0)   ## point about which detector is rotated   

    sinogram = np.zeros((det_steps,int(det_width/det_spacing)))

    for i in range(0,det_steps):

        rad_ang = np.pi*i*det_step_ang/180      ## converting iteration angle to radians
        detector_mid_pt = det_origin-np.array([det_radius*np.cos(rad_ang),det_radius*np.sin(rad_ang)])

        detector_ang = (90+i)*np.pi/180

        ## estimate detector start and end points for each rotation

        detector_start_pt = detector_mid_pt-np.array([det_width*np.cos(detector_ang)/2,det_width*np.sin(detector_ang)/2])
        detector_end_pt = detector_mid_pt+np.array([det_width*np.cos(detector_ang)/2,det_width*np.sin(detector_ang)/2])

        detector_vec = detector_end_pt-detector_start_pt    
        detector_vec = detector_vec/np.linalg.norm(detector_vec)        ## unit vector of detector

        tmp_arr = np.zeros(np.shape(nodes_2d)[0])
        
        ## project each node onto detector

        for j in range(0,np.shape(nodes_2d)[0]):
            original_vec = nodes_2d[j,:]-detector_start_pt
            tmp_arr[j] = np.dot(original_vec,detector_vec)
        
        hist, _ = np.histogram(tmp_arr, bins=bin_edges)     ## create sinogram intensity through number of nodes_2d projected onto a region
        sinogram[i,:] = hist

    sinogram = cv.GaussianBlur(sinogram, (3,3), 0)      ## blur sinogram to remove discreetness of nodes

    sinogram = sinogram.T

    return sinogram

def create_3D_sinogram(nodes, z_spacing=0.1, det_spacing=0.1, det_width=10, det_step_ang=1, det_radius=100):
    """radon transform.

    Create a sinogram from a set of 3D nodes
    Assumes parallel beams and does not go through helical motion
    Parameters
    ----------
    nodes: n x 3 array
        Numpy array containing all node points of mesh
    z_spacing: optional
        Value to move the detector by between consecutive rotations
    det_spacing: optional
        space between consecutive detectors (Effective pixel/voxel size)
    det_width: optional
        total length of detector / resulting image
    det_step_ang :  optional
        Angle movement in degrees between acquisitions
    det_radius : optional
        Radius about the center of mass of scanning object
    -------    
    Returns
    -------
    3D sinogram: ndarray
        returns 3D sinogram array
        rows - number of pixels of the final image
        columns - number of angle steps
        stride - number of z steps
    """

    z_min = np.nanmin(nodes[:,2])       ## find the z-range of the mesh (height)
    z_max = np.nanmax(nodes[:,2])

    z_height = z_max-z_min

    z_min = z_min-0.05*z_height       ## set sinogram boundaries with 5% margin
    z_max = z_max+0.05*z_height 

    z_bins = np.arange(start=z_min,stop=z_max+z_spacing,step=z_spacing)         ## zet effective number of z voxels
    bin_num = np.digitize(nodes[:,2], bins=z_bins)      ## separate nodes per z-zone

    det_steps = int(180/det_step_ang)

    det_origin = np.mean(nodes,axis=0)   ## point about which detector is rotated

    sinogram_3D = np.zeros((int(det_width/det_spacing),det_steps,np.shape(z_bins)[0]))

    for i in tqdm(range(0,np.nanmax(bin_num)), desc="Creating sinogram..."):
        indx = np.array(np.where(bin_num==i+1))
        indx = indx.flatten()

        if(np.shape(indx)[0]==0):
            continue

        bin_nodes = nodes[indx,0:2]

        ## calling above 2D sinogram function
        sinogram_3D[:,:,i] = create_sinogram(bin_nodes,det_spacing,det_width,det_step_ang,det_radius,det_origin=det_origin[0:2])

    return sinogram_3D

def invert_3d_sinogram(sinogram_3D, det_step_ang=1, filter_type='hann'):
    """3D inverse radon transform.

    Create an image from a 3D sinogram (stacks of 2D sinograms)
    Parameters
    ----------
    sinogram :  3D array
        3D sinogram array. 3rd dimension is along the z 
        i.e. process iterate along 3rd dimensions processing sinogram[:,:,i] as a regular sinogram
    -------    
    Returns
    -------
    3D CT image: ndarray
        returns conventional 3D medical image array
    """
    theta = np.array(np.arange(start=0.0, stop=180.0, step=det_step_ang))

    final_img = np.zeros((np.shape(sinogram_3D)[0],np.shape(sinogram_3D)[0],np.shape(sinogram_3D)[2]))

    for i in tqdm(range(0,np.shape(sinogram_3D)[2]), desc="Inverse Radon..."):
        final_img[:,:,i]  = iradon(sinogram_3D[:,:,i], theta=theta, filter_name=filter_type)

    return final_img