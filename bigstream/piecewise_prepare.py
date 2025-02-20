import os, tempfile
import bigstream.utility as ut
import numpy as np
from itertools import product
from bigstream.transform import apply_transform_to_coordinates
from bigstream.align import alignment_pipeline
import time
import json
import pickle
from tqdm import tqdm


def serialize_slices(lst):
    new_lst = []
    for item in lst:
        if isinstance(item, slice):
            item = f"slice({item.start}, {item.stop}, {item.step})"
        elif isinstance(item, list):
            item = serialize_slices(item)
        new_lst.append(item)
    return new_lst

def prepare_distributed_piecewise_alignment_pipeline(
    fix,
    mov,
    fix_spacing,
    mov_spacing,
    output_dir,
    blocksize,
    overlap=0.5,
    fix_mask=None,
    mov_mask=None,
    foreground_percentage=0.5,
    static_transform_list=[],
    cluster=None,
    cluster_kwargs={},
    temporary_directory=None,
    write_path=None,
    write_group_interval=30,
    **kwargs,
):
    """
    Piecewise alignment of moving to fixed image.
    Overlapping blocks are given to `alignment_pipeline` in parallel
    on distributed hardware. Can include random, rigid, affine, and
    deformable alignment. Inputs can be numpy or zarr arrays. Output
    is a single displacement vector field for the entire domain.
    Output can be returned to main process memory as a numpy array
    or written to disk as a zarr array.

    Parameters
    ----------
    fix : ndarray
        the fixed image

    mov : ndarray
        the moving image; `fix.shape` must equal `mov.shape`
        I.e. typically piecewise affine alignment is done after
        a global affine alignment wherein the moving image has
        been resampled onto the fixed image voxel grid.

    fix_spacing : 1d array
        The spacing in physical units (e.g. mm or um) between voxels
        of the fixed image.
        Length must equal `fix.ndim`

    mov_spacing : 1d array
        The spacing in physical units (e.g. mm or um) between voxels
        of the moving image.
        Length must equal `mov.ndim`

    steps : list of tuples in this form [(str, dict), (str, dict), ...]
        For each tuple, the str specifies which alignment to run. The options are:
        'random' : run `random_affine_search`
        'rigid' : run `affine_align` with `rigid=True`
        'affine' : run `affine_align`
        'deform' : run `deformable_align`
        For each tuple, the dict specifies the arguments to that alignment function
        Arguments specified here override any global arguments given through kwargs
        for their specific step only.

    blocksize : iterable
        The shape of blocks in voxels

    overlap : float in range [0, 1] (default: 0.5)
        Block overlap size as a percentage of block size

    fix_mask : binary ndarray (default: None)
        A mask limiting metric evaluation region of the fixed image
        Assumed to have the same domain as the fixed image, though sampling
        can be different. I.e. the origin and span are the same (in physical
        units) but the number of voxels can be different.

    mov_mask : binary ndarray (default: None)
        A mask limiting metric evaluation region of the moving image
        Assumed to have the same domain as the moving image, though sampling
        can be different. I.e. the origin and span are the same (in physical
        units) but the number of voxels can be different.

    static_transform_list : list of numpy arrays (default: [])
        Transforms applied to moving image before applying query transform
        Assumed to have the same domain as the fixed image, though sampling
        can be different. I.e. the origin and span are the same (in physical
        units) but the number of voxels can be different.

    cluster : ClusterWrap.cluster object (default: None)
        Only set if you have constructed your own static cluster. The default behavior
        is to construct a cluster for the duration of this function, then close it
        when the function is finished.

    cluster_kwargs : dict (default: {})
        Arguments passed to ClusterWrap.cluster
        If working with an LSF cluster, this will be
        ClusterWrap.janelia_lsf_cluster. If on a workstation
        this will be ClusterWrap.local_cluster.
        This is how distribution parameters are specified.

    temporary_directory : string (default: None)
        Temporary files are created during alignment. The temporary files will be
        in their own folder within the `temporary_directory`. The default is the
        current directory. Temporary files are removed if the function completes
        successfully.

    write_path : string (default: None)
        If the transform found by this function is too large to fit into main
        process memory, set this parameter to a location where the transform
        can be written to disk as a zarr file.

    write_group_interval : float (default: 30.)
        The time each of the 27 mutually exclusive write block groups have
        each round to write finished data.

    kwargs : any additional arguments
        Arguments that will apply to all alignment steps. These are overruled by
        arguments for specific steps e.g. `random_kwargs` etc.

    Returns
    -------
    field : nd array or zarr.core.Array
        Local affines stitched together into a displacement field
        Shape is `fix.shape` + (3,) as the last dimension contains
        the displacement vector.
    """

    # temporary file paths and create zarr images
    temporary_directory = tempfile.TemporaryDirectory(
        prefix='.', dir=temporary_directory or os.getcwd(),
    )
    zarr_blocks = [128,] * fix.ndim

    # zarr files for initial deformations
    new_list = []
    for iii, transform in enumerate(static_transform_list):
        if transform.shape != (4, 4) and len(transform.shape) != 1:
            path = temporary_directory.name + f'/deform{iii}.zarr'
            transform = ut.numpy_to_zarr(transform, tuple(zarr_blocks) + (transform.shape[-1],), path)
        new_list.append(transform)
    static_transform_list = new_list

    # determine fixed image slices for blocking
    blocksize = np.array(blocksize)
    nblocks = np.ceil(np.array(fix.shape) / blocksize).astype(int)
    overlaps = np.round(blocksize * overlap).astype(int)
    indices, slices = [], []
    for (i, j, k) in np.ndindex(*nblocks):
        start = blocksize * (i, j, k) - overlaps
        stop = start + blocksize + 2 * overlaps
        start = np.maximum(0, start)
        stop = np.minimum(fix.shape, stop)
        coords = tuple(slice(x, y) for x, y in zip(start, stop))

        foreground = True
        if fix_mask is not None:
            start = blocksize * (i, j, k)
            stop = start + blocksize
            ratio = np.array(fix_mask.shape) / fix.shape
            start = np.round( ratio * start ).astype(int)
            stop = np.round( ratio * stop ).astype(int)
            mask_crop = fix_mask[tuple(slice(a, b) for a, b in zip(start, stop))]
            if not np.sum(mask_crop) / np.prod(mask_crop.shape) >= foreground_percentage:
                foreground = False

        if foreground:
            indices.append((i, j, k,))
            slices.append(coords)

    # determine foreground neighbor structure
    new_indices = []
    neighbor_offsets = np.array(list(product([-1, 0, 1], repeat=3)))
    for index, coords in zip(indices, slices):
        neighbor_flags = {tuple(o): tuple(index + o) in indices for o in neighbor_offsets}
        new_indices.append((index, coords, neighbor_flags))
    indices = new_indices

    # establish all keyword arguments
    # steps = [(a, {**kwargs, **b}) for a, b in steps]

    # #print type of all variables
    # print("steps type: ", type(steps), flush=True)
    # print(steps, flush=True)
    
    #print("nblocks type: ", type(nblocks), flush=True)
    #print('blocksize type: ', type(blocksize), flush=True)
    #print('overlaps type: ', type(overlaps), flush=True)
    #print('indicies type: ', type(indices), flush=True)
    #print(indices, flush=True)
    
    
    # print('starting write of pipeline config to json file', flush=True)
    # pipeline_config = {
    #     'steps': serialize_slices(steps),
    # }
    
    try:
        np.savetxt(output_dir +'/blocksize.txt', blocksize)
    except: pass
    try:
        np.save(output_dir +'/blocksize.npy', blocksize)
    except: pass
    try:
        np.savetxt(output_dir +'/overlaps.txt', overlaps)
    except: pass
    try:
        np.save(output_dir +'/overlaps.npy', overlaps)
    except: pass
    try:
        np.savetxt(output_dir +'/nblocks.txt', nblocks)
    except: pass
    try:
        np.save(output_dir +'/nblocks.npy', nblocks)
    except: pass
    
    
    print('writing pipeline config to json file', flush=True)
    

    # Save pipeline_config to a json file
    # with open('/results/pipeline_config.json', 'w') as f:
    #     json.dump(pipeline_config, f)    

    non_empty_indices = get_non_empty_indices(indices, mov, mov_mask)

    print("Got non empty indices", flush=True)
    print(non_empty_indices, flush=True)

    indices_dict = {}
    
    for i, indexed_config in enumerate(indices):
        #write to separate pkl files
        index, coords, neighbor_flags = indexed_config

        print("i: ", i, flush=True)
        if index in non_empty_indices:
            
            print("non empty index: ", index, flush=True)
            # before writing indices file, update neighbor flags to exclude empty blocks
            for neighbor in neighbor_flags:
                
                if neighbor_flags[neighbor]: # if expecting a neighbor block
                    neighbor_index = tuple(np.array(index) + np.array(neighbor))
                    print("expecting neighbor block: ", neighbor_index, flush=True)


                    if neighbor_index not in non_empty_indices:
                        print("neighbor block empty: ", neighbor_index, flush=True) 
                        neighbor_flags[neighbor] = False

            indexed_config = (index, coords, neighbor_flags)  # update indexed_config with new neighbor_flags
            indices_dict[str(i)] = indexed_config[1]
            if not os.path.exists(output_dir +f'/distribute/{str(i)}'):
                os.makedirs(output_dir + f'/distribute/{str(i)}')
            
            with open(output_dir + f'/distribute/{str(i)}/indices.pkl', 'wb') as f:
                pickle.dump(indexed_config, f)
    
    with open(output_dir +'/indices_dict.pkl', 'wb') as f:
        pickle.dump(indices_dict, f)              
    #print('finished writing pipeline config to json file', flush=True)
    
    # write blocksize to npy file
    
    
    # submit all alignments to cluster
    # futures = cluster.client.map(
    #     align_single_block, indices,
    #     static_transform_list=static_transform_list,
    # )


def get_non_empty_indices(indices, mov, mov_mask):
    # Calculate the ratio for scaling just once
    ratio = np.array(mov_mask.shape) / np.array(mov.shape)

    # Determine the indices of the non-empty blocks in the moving image
    non_empty_indices = []

    pbar = tqdm(total=len(indices), desc="Processing indices")

    for indexed_config in indices:
        index, coords, neighbor_flags = indexed_config
        # Extract start and stop points from slices
        starts = [s.start for s in coords]
        stops = [s.stop for s in coords]

        # Convert to numpy arrays
        starts = np.array(starts)
        stops = np.array(stops)

        # Scale starts and stops according to the ratio
        starts = np.round(ratio * starts).astype(int)
        stops = np.round(ratio * stops).astype(int)

        # Creating slices for each dimension
        mask_crop = mov_mask[tuple(slice(s, e) for s, e in zip(starts, stops))]

        # Check for any non-zero element in the mask_crop
        if np.any(mask_crop):
            non_empty_indices.append(index)

        # Update the progress bar
        pbar.update(1)

    # Close the progress bar
    pbar.close()

    return non_empty_indices






# closure for alignment pipeline
def align_single_block(
    fix,
    mov,
    fix_spacing,
    mov_spacing,
    steps,
    nblocks,
    blocksize,
    overlaps,
    indices,
    static_transform_list,
    fix_mask=None,
    mov_mask=None,
    temporary_directory=None,
    **kwargs
):

    # temporary file paths and create zarr images
    temporary_directory = tempfile.TemporaryDirectory(
        prefix='.', dir=temporary_directory or os.getcwd(),
    )
    
    zarr_blocks = [128,] * fix.ndim
    fix_zarr_path = temporary_directory.name + '/fix.zarr'
    mov_zarr_path = temporary_directory.name + '/mov.zarr'
    fix_mask_zarr_path = temporary_directory.name + '/fix_mask.zarr'
    mov_mask_zarr_path = temporary_directory.name + '/mov_mask.zarr'
    fix_zarr = ut.numpy_to_zarr(fix, zarr_blocks, fix_zarr_path)
    mov_zarr = ut.numpy_to_zarr(mov, zarr_blocks, mov_zarr_path)
    fix_mask_zarr = None
    if fix_mask is not None: fix_mask_zarr = ut.numpy_to_zarr(fix_mask, zarr_blocks, fix_mask_zarr_path)
    mov_mask_zarr = None
    if mov_mask is not None: mov_mask_zarr = ut.numpy_to_zarr(mov_mask, zarr_blocks, mov_mask_zarr_path)

    # print some feedback
    print("Block index: ", indices[0], "\nSlices: ", indices[1], flush=True)

    # get the coordinates, read fixed data
    block_index, fix_slices, neighbor_flags = indices
    fix = fix_zarr[fix_slices]

    # get fixed image block corners in physical units
    fix_block_coords = []
    for corner in list(product([0, 1], repeat=3)):
        a = [x.stop-1 if y else x.start for x, y in zip(fix_slices, corner)]
        fix_block_coords.append(a)
    fix_block_coords = np.array(fix_block_coords)
    fix_block_coords_phys = fix_block_coords * fix_spacing

    # parse initial transforms
    # recenter affines, read deforms, apply transforms to crop coordinates
    new_list = []
    mov_block_coords_phys = np.copy(fix_block_coords_phys)
    for transform in static_transform_list[::-1]:
        if transform.shape == (4, 4):
            mov_block_coords_phys = apply_transform_to_coordinates(
                mov_block_coords_phys, [transform,],
            )
            transform = ut.change_affine_matrix_origin(transform, fix_block_coords_phys[0])
        else:
            ratio = np.array(transform.shape[:-1]) / fix_zarr.shape
            start = np.round( ratio * fix_block_coords[0] ).astype(int)
            stop = np.round( ratio * (fix_block_coords[-1] + 1) ).astype(int)
            transform_slices = tuple(slice(a, b) for a, b in zip(start, stop))
            transform = transform[transform_slices]
            spacing = ut.relative_spacing(transform, fix, fix_spacing)
            origin = spacing * start
            mov_block_coords_phys = apply_transform_to_coordinates(
                mov_block_coords_phys, [transform,], spacing, origin
            )
        new_list.append(transform)
    static_transform_list = new_list[::-1]

    # get moving image crop, read moving data 
    mov_block_coords = np.round(mov_block_coords_phys / mov_spacing).astype(int)
    mov_start = np.min(mov_block_coords, axis=0)
    mov_stop = np.max(mov_block_coords, axis=0)
    mov_start = np.maximum(0, mov_start)
    mov_stop = np.minimum(np.array(mov_zarr.shape)-1, mov_stop)
    mov_slices = tuple(slice(a, b) for a, b in zip(mov_start, mov_stop))
    mov = mov_zarr[mov_slices]

    # XXX if input masks are zarr arrays this doesn't work, nothing at paths
    # read masks
    fix_mask, mov_mask = None, None
    if fix_mask_zarr is not None:
        ratio = np.array(fix_mask_zarr.shape) / fix_zarr.shape
        start = np.round( ratio * fix_block_coords[0] ).astype(int)
        stop = np.round( ratio * (fix_block_coords[-1] + 1) ).astype(int)
        fix_mask_slices = tuple(slice(a, b) for a, b in zip(start, stop))
        fix_mask = fix_mask_zarr[fix_mask_slices]
    if mov_mask_zarr is not None:
        ratio = np.array(mov_mask_zarr.shape) / mov_zarr.shape
        start = np.round( ratio * mov_start ).astype(int)
        stop = np.round( ratio * mov_stop ).astype(int)
        mov_mask_slices = tuple(slice(a, b) for a, b in zip(start, stop))
        mov_mask = mov_mask_zarr[mov_mask_slices]

    # get moving image origin
    mov_origin = mov_start * mov_spacing - fix_block_coords_phys[0]

    # run alignment pipeline
    transform = alignment_pipeline(
        fix, mov, fix_spacing, mov_spacing, steps,
        fix_mask=fix_mask, mov_mask=mov_mask,
        mov_origin=mov_origin,
        static_transform_list=static_transform_list,
        **kwargs
    )

    # Ensure that transform is a list of arrays, 
    # Required to handle alignment_pipeline being passed return_format='indepedent'
    if not isinstance(transform, list):
        transform = [transform]  # Convert transform to a list with one element if it's not already a list

    processed_transforms = []  # List to store processed transforms

    for single_transform in transform:
        # Ensure transform is a vector field
        if single_transform.shape == (4, 4):
            single_transform = ut.matrix_to_displacement_field(
                single_transform, fix.shape, spacing=fix_spacing,
            )

        print("TRANSFORM SHAPE, type: ", single_transform.shape, single_transform.dtype, flush=True)

        # Append processed transform to the list
        processed_transforms.append(single_transform)

    # Return the list of processed transforms
    if len(processed_transforms) == 1:
        # Return single transform if only one was returned from the alignment pipeline
        return processed_transforms[0]
    return processed_transforms
# END CLOSURE

def apply_weights_to_transform(single_transform, indices, nblocks, blocksize, overlaps):

    block_index, fix_slices, neighbor_flags = indices

    # Create the standard weight array
    core = tuple(x - 2*y + 2 for x, y in zip(blocksize, overlaps))
    pad = tuple((2*y - 1, 2*y - 1) for y in overlaps)
    weights = np.pad(np.ones(core, dtype=np.float64), pad, mode='linear_ramp')

    # Rebalance if any neighbors are missing
    if not np.all(list(neighbor_flags.values())):

        # Define overlap slices
        slices = {}
        slices[-1] = tuple(slice(0, 2*y) for y in overlaps)
        slices[0] = (slice(None),) * len(overlaps)
        slices[1] = tuple(slice(-2*y, None) for y in overlaps)

        missing_weights = np.zeros_like(weights)
        for neighbor, flag in neighbor_flags.items():
            if not flag:
                neighbor_region = tuple(slices[-1*b][a] for a, b in enumerate(neighbor))
                region = tuple(slices[b][a] for a, b in enumerate(neighbor))
                missing_weights[region] += weights[neighbor_region]

        # Rebalance the weights
        weights = weights / (1 - missing_weights)
        weights[np.isnan(weights)] = 0.  # edges of blocks are 0/0
        weights = weights.astype(np.float32)

    # Crop weights if block is on edge of domain
    for i in range(3):
        region = [slice(None),]*3
        if block_index[i] == 0:
            region[i] = slice(overlaps[i], None)
            weights = weights[tuple(region)]
        if block_index[i] == nblocks[i] - 1:
            region[i] = slice(None, -overlaps[i])
            weights = weights[tuple(region)]

    # Crop any incomplete blocks (on the ends)
    if np.any(weights.shape != single_transform.shape[:-1]):
        crop = tuple(slice(0, s) for s in single_transform.shape[:-1])
        weights = weights[crop]

    # Apply weights
    single_transform = single_transform * weights[..., None]

    return single_transform

