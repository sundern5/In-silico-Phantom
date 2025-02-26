import numpy as np
import nibabel as nib 
import projection_functions as p

nodes = np.genfromtxt('nodes.txt',delimiter=',')        ## example nodes
nodes = nodes[:,1:]

width = np.nanmax([np.nanmax(nodes[:,0])-np.nanmin(nodes[:,0]),np.nanmax(nodes[:,1])-np.nanmin(nodes[:,1])])
det_width = np.round(width*1.2)     ## find required image width

z_spacing = 2.0     ## set z-spacing (mm)
det_spacing = 1.0   ## set voxel width (mm)
det_step_ang = 1    ## set angle between consecutive acqusitions (degrees)

sinogram_3D = p.create_3D_sinogram(nodes=nodes,z_spacing=z_spacing,det_spacing=det_spacing,det_width=det_width,\
                                   det_step_ang=det_step_ang)

nifti_img = nib.Nifti1Image(sinogram_3D, affine = np.eye(4))
nib.save(nifti_img, 'test_sinogram.nii')

final_img = p.invert_3d_sinogram(sinogram_3D=sinogram_3D,det_step_ang=det_step_ang,filter_type='hann')

nifti_img = nib.Nifti1Image(final_img, affine = np.eye(4))
nib.save(nifti_img, 'test_image.nii')