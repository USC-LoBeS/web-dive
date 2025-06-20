import numpy as np
import pandas as pd
from tqdm import tqdm
import nibabel as nib
import dipy
from scipy.spatial import cKDTree
from dipy.tracking import utils
from tslearn.metrics import dtw_path
from dipy.tracking.streamline import length
from dipy.segment.clustering import QuickBundles
from dipy.segment.featurespeed import ResampleFeature
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from dipy.tracking.streamline import length, transform_streamlines, Streamlines


def get_len(file):
    df = pd.read_csv(file)
    return df.shape[0]

def assignment_map_(target_bundle, model_bundle, no_disks,threshold=None,method=None,affine =[],bundle_shape=None):
        """
        Calculates assignment maps of the target bundle
        Returns
        -------
        indx : ndarray
            Assignment map of the target bundle streamline point indices to the
            model bundle centroid points.

        """


        if method=="Center":
            threshold=np.inf
            m_feature = ResampleFeature(nb_points=no_disks)
            m_metric = AveragePointwiseEuclideanMetric(m_feature)
            qb = QuickBundles(threshold=threshold, metric=m_metric)
            clusters = qb.cluster(target_bundle)
            centroids = Streamlines(clusters.centroids)
            _, indx = cKDTree(centroids.get_data(), 1, copy_data=True).query(
            target_bundle.get_data(), k=1)
            return np.array(indx)
        
        if method=="Meta":
            if np.array_equal(affine, target_bundle.affine) :
                mask = create_mask_from_trk(target_bundle,bundle_shape)
                nifti_image = nib.Nifti1Image(mask, np.linalg.inv(affine))
            else:
                #print("Bundle Density Doing")
                nifti_image = bundle_density(target_bundle,bundle_shape,affine)
                #print("Bundle Density Done",target_bundle,bundle_shape,affine)
                mask = nifti_image.get_fdata()
        
            dtw_points_sets = perform_dtw(model_bundle=target_bundle, subject_bundle=model_bundle,num_segments=no_disks, affine=affine)
            bundle=target_bundle.streamlines
            points = [item for sublist in bundle for item in sublist]
            segments = segment_bundle(nifti_image.get_fdata(), dtw_points_sets, no_disks)
            segmented_bundle=np.zeros(mask.shape)
            numIntensities = len(segments)
            intensities = [i+1 for i in range(numIntensities)]
            for i,j in zip(segments,intensities):
                segmented_bundle+=((i)*j)
            unique_values = np.unique(segmented_bundle)
            nifti_data = segmented_bundle
            voxels = np.round(nib.affines.apply_affine(np.linalg.inv(affine), points))
            nifti_dict = {(i, j, k): val for (i, j, k), val in np.ndenumerate(nifti_data)}
            master_color = list(map(lambda vox: nifti_dict[abs(vox[0]), abs(vox[1]), abs(vox[2])], voxels))
            master_color = np.asarray(master_color).astype(int)
            return master_color
def reorient_streamlines(m_centroid, s_centroids):
    """
    Reorients the subject centroids based on the model centroid.
    Args:
        m_centroid (np.ndarray): Model centroid
        s_centroids (list): List of subject centroids
    Returns:
        oriented_s_centroids (list): List of reoriented subject centroids
    """

    def is_flipped(m_centroid, s_centroid):
        """
        checks if subjects centroid is flipped compared to the model centroid.
        """
        start_distance = np.linalg.norm(m_centroid[0] - s_centroid[-1])
        start = np.linalg.norm(m_centroid[-1] - s_centroid[0])

        end_distance = np.linalg.norm(m_centroid[-1] - s_centroid[-1])
        end = np.linalg.norm(m_centroid[0] - s_centroid[0])

        if (start_distance < end_distance) and (start < end):
            return True
        return False

    oriented_s_centroids = []
    for s_centroid in s_centroids:
        if is_flipped(m_centroid, s_centroid):
            oriented_s_centroids.append(s_centroid[::-1])
        else:
            oriented_s_centroids.append(s_centroid)

    return oriented_s_centroids


def perform_dtw(model_bundle, subject_bundle, num_segments, affine=None):
    """
    This function performs Dynamic Time Warping (DTW) on two tractogram (.trk)
    files in same space.

    Args:
        tbundle (str): path to a template .trk file
        sbundle (str): Path to a subject .trk file
        num_segments (int): number of points (N+1) of template centroid to segment the bundle (N)

    Returns:
        dict: dictionary containing the corresponding points.
    """

    # reference_image = nib.load(mask_img)

    ## Trasform the Template bundle to the subject space world cordinates and then to the subject voxel space cordinates:
    ##load_tractogram(model_bundle, "same", bbox_valid_check=False)
    model_streamlines = model_bundle.streamlines
    transformed_model_bundles = transform_streamlines(model_streamlines, np.linalg.inv(affine))

    m_feature = ResampleFeature(nb_points=num_segments)
    m_metric = AveragePointwiseEuclideanMetric(m_feature)
    m_qb = QuickBundles(threshold=np.inf, metric=m_metric)
    m_centroid = m_qb.cluster(transformed_model_bundles).centroids
    print('Model: Centroid length... ', np.mean([length(streamline) for streamline in m_centroid]))

    ## Trasform the Subject bundle to the subject voxel cordinates:
    subject_streamlines = subject_bundle.streamlines
    transformed_subject_bundles = transform_streamlines(subject_streamlines, np.linalg.inv(affine))
    s_feature = ResampleFeature(nb_points=500)
    s_metric = AveragePointwiseEuclideanMetric(s_feature)
    s_qb = QuickBundles(threshold=np.inf, metric=s_metric)
    s_centroid = s_qb.cluster(transformed_subject_bundles).centroids
    print('Subject: Centroid length... ', np.mean([length(streamline) for streamline in s_centroid]))

    ## Create multiple centroids from subject bundle using QuickBundles
    num_clusters = 100
    feature = ResampleFeature(nb_points=500)
    metric = AveragePointwiseEuclideanMetric(feature)
    qb = QuickBundles(threshold=2., metric=metric, max_nb_clusters=num_clusters)
    centroids = qb.cluster(transformed_subject_bundles).centroids

    ## Check if the centroids are flipped compared to the model centroid
    s_centroid = reorient_streamlines(m_centroid, s_centroid)
    centroids = reorient_streamlines(m_centroid, centroids)

    ## Compute the correspondence between the model and the subject centroids using DTW
    dtw_corres = []
    for idx, (m_centroid, s_centroid) in enumerate(zip(m_centroid, s_centroid)):
        pathDTW, similarityScore = dtw_path(m_centroid, s_centroid)
        x1, y1, z1 = m_centroid[:, 0], m_centroid[:, 1], m_centroid[:, 2]
        x2, y2, z2 = s_centroid[:, 0], s_centroid[:, 1], s_centroid[:, 2]
        corres = dict()
        for (i, j) in pathDTW:
            key = (x1[i], y1[i], z1[i])
            value = (x2[j], y2[j], z2[j])
            if key in corres:
                corres[key].append(value)
            else:
                corres[key] = [value]
        centroid_corres = []
        for key in corres.keys():
            t = len(corres[key]) // 2
            centroid_corres.append(corres[key][t])
        dtw_corres.append(np.array(centroid_corres))

    ## Establish correspondence between dtw_corres and centroids of the subject bundle
    s_corres = []
    for idx, centroid in enumerate(centroids):

        s_centroid = np.squeeze(centroid)
        s_ref  = np.squeeze(dtw_corres)
        pathDTW, similarityScore = dtw_path(s_ref, s_centroid)
        x1, y1, z1 = s_ref[:, 0], s_ref[:, 1], s_ref[:, 2]
        x2, y2, z2 = s_centroid[:, 0], s_centroid[:, 1], s_centroid[:, 2]
        corres = dict()
        for (i, j) in pathDTW:
            key = (x1[i], y1[i], z1[i])
            value = (x2[j], y2[j], z2[j])
            if key in corres:
                corres[key].append(value)
            else:
                corres[key] = [value]

        centroid_corres = []
        for key in corres.keys():
            t = len(corres[key]) // 2
            centroid_corres.append(corres[key][t])
        s_corres.append(np.array(centroid_corres))

    ## combine correspondences
    combined_corres = dtw_corres + s_corres

    ## Remove centroids that are shorter than the threshold
    data = []
    for streamline in combined_corres:
        data.append(length(streamline))
    mean_length = np.mean(data)
    std_length = np.std(data)
    print("Average streamlines length", np.mean(data))
    print("Standard deviation", std_length)
    threshold = mean_length - 1 * std_length
    indices = np.where(data < threshold)
    final_corres = [sl for idx, sl in enumerate(combined_corres) if idx not in indices[0]]

    ## Compute pairwise distances between corresponding points of the final centroids
    corresponding_points = np.array(final_corres)
    pairwise_distances = np.zeros((corresponding_points.shape[1], corresponding_points.shape[0], corresponding_points.shape[0]))
    for i in range(corresponding_points.shape[1]):
        for j in range(corresponding_points.shape[0]):
            for k in range(j + 1, corresponding_points.shape[0]):
                pairwise_distances[i, j, k] = np.linalg.norm(corresponding_points[j, i] - corresponding_points[k, i])
    pairwise_distances[pairwise_distances == 0] = np.nan
    mean_distances = np.nanmean(pairwise_distances, axis=(1, 2))
    std_distances = np.nanstd(pairwise_distances, axis=(1, 2))
    excluded_idx = np.where(std_distances <= 3.5)[0]

    ## Filter the final_corres based on pairwise distances that have std <= 3.5
    excluded_start = excluded_idx[0]
    excluded_end = excluded_idx[-1]

    filtered_arrays = []
    for idx, array in enumerate(final_corres):
        combined_array = []
        if excluded_start > 1:
            start_point = array[0]
            end_point = array[excluded_start]
            side_1_points = np.linspace(start_point, end_point, excluded_start + 1)[1:-1]
            combined_array.extend(array[0:1])
            combined_array.extend(side_1_points)
        elif excluded_start <= 1:
            combined_array.extend(array[0:excluded_start])
        combined_array.extend(array[excluded_start:excluded_end+1])
        if num_segments - excluded_end > 1:
            start_point = array[excluded_end]
            end_point = array[-1]
            side_2_points = np.linspace(start_point, end_point, num_segments - excluded_end)[1:-1]
            combined_array.extend(side_2_points)
            combined_array.extend(array[-1:])
        elif num_segments - excluded_end == 1:
            combined_array.extend(array[-1:])

        filtered_arrays.append(np.array(combined_array))
    print("Total number filtered centroids:", len(filtered_arrays))
    return filtered_arrays



def segment_bundle(bundle_data, dtw_points_sets, num_segments):
    """
    Parcellate white matter bundle into num_segments based on DTW points.

    Parameters:
    -----------
    bundle_data: A bundle mask as a NumPy array.
    dtw_points_sets: list of ndarrays of shape (num_segments, 3) which are the corresponding DTW points.
    num_segments (int): required number of segments.

    Returns:
    --------
    segments: A list of labels, where each label corresponds to a segment.
    """
    segments = [np.zeros_like(bundle_data, dtype=bool) for _ in range(num_segments+1)]

    for dtw_points in tqdm(dtw_points_sets):
        for i in range(num_segments):
            if i == 0:
                plane_normal = (dtw_points[i+1] - dtw_points[i]).astype(float)
                for x, y, z in np.argwhere(bundle_data):
                    point = np.array([x, y, z])
                    if np.dot(point - dtw_points[i], -plane_normal) >= 0:
                        segments[i][x, y, z] = True

            ## 1st plane >>>
            if i < num_segments - 2 and i >= 0:
                plane_normal = (dtw_points[i+1] - dtw_points[i]).astype(float)
                next_plane_normal = (dtw_points[i+1 + 1] - dtw_points[i+1]).astype(float)
                for x, y, z in np.argwhere(bundle_data):
                    point = np.array([x, y, z])
                    if np.dot(point - dtw_points[i], plane_normal) >= 0 and np.dot(point - dtw_points[i+1], -next_plane_normal) >= 0:
                        segments[i+1][x, y, z] = True

            ## 2nd plane - end
            elif i == num_segments - 2:
                plane_normal = (dtw_points[i] - dtw_points[i-1]).astype(float)
                for x, y, z in np.argwhere(bundle_data):
                    point = np.array([x, y, z])
                    if np.dot(point - dtw_points[i-1], plane_normal) >= 0:
                        segments[i+1][x, y, z] = True

            ## end plane >>>
            elif i == num_segments - 1:
                plane_normal = (dtw_points[i] - dtw_points[i-1]).astype(float)
                for x, y, z in np.argwhere(bundle_data):
                    point = np.array([x, y, z])
                    if np.dot(point - dtw_points[i], plane_normal) >= 0:
                        segments[i+1][x, y, z] = True

    ######## catching remaining voxels ########
    arrays = np.array(segments)
    sum_array = np.sum(arrays, axis=0)
    remaining_voxels = sum_array.copy()
    if np.any(remaining_voxels):
        for x, y, z in np.argwhere(sum_array >= 2):
            for seg in segments:
                seg[x, y, z] = False
            point = np.array([x, y, z])
            min_distance = float('inf')
            closest_segment_idx = None
            for dtw_points in dtw_points_sets:
                for i in range(num_segments):
                    distance_to_start = np.linalg.norm(point - dtw_points[i])
                    if distance_to_start < min_distance:
                        min_distance = distance_to_start
                        closest_segment_idx = i
            if closest_segment_idx is not None:
                segments[closest_segment_idx][x, y, z] = True
    return segments

def create_mask_from_trk(streams, shape):
    # Load TRK file
    # streams, header = trackvis.read(trk_file)

    # Create an empty mask
    transformed_streamlines = transform_streamlines(streams.streamlines, np.linalg.inv(streams.affine))
    mask = np.zeros(shape, dtype=np.uint8)

    # Iterate through each stream in the tractography data
    for stream in transformed_streamlines:
        # Convert stream coordinates to integer indices
        for point in stream:
            x, y, z = np.round(point).astype(int)
            # Check if the point is within the mask dimensions
            if 0 <= x < shape[0] and 0 <= y < shape[1] and 0 <= z < shape[2]:
                mask[x, y, z] = 1  # Mark this voxel in the mask
    return mask

def bundle_density(streams, ref_shape, ref_affine):
    """
    Create a binary density map from streamlines.
    
    Parameters:
    -----------
    streams : object
        Object containing streamlines attribute
    ref_shape : tuple
        Shape of the reference volume (x, y, z)
    ref_affine : numpy.ndarray
        4x4 affine transformation matrix
    
    Returns:
    --------
    nibabel.Nifti1Image
        Binary density map as a NIFTI image
    """
    # Get streamlines
    streamlines = streams.streamlines
    
    # Convert streamlines to list if they're not already
    if not isinstance(streamlines, list):
        streamlines = list(streamlines)
    
    # Calculate upsampling step size based on voxel size
    voxel_size = np.abs(ref_affine[0:3, 0:3].diagonal())
    max_seq_len = np.min(voxel_size) / 2  # Use half the smallest voxel dimension
    
    try:
        # Upsample streamlines
        streamlines = list(utils.subsegment(streamlines, max_seq_len))
        
        # Ensure streamlines are within volume bounds
        # Get the corner coordinates of the volume
        corner_coords = np.array(ref_shape) - 1
        corner_points = np.array([[0, 0, 0], 
                                [corner_coords[0], 0, 0],
                                [0, corner_coords[1], 0],
                                [0, 0, corner_coords[2]],
                                corner_coords])
        
        # Transform corner points to world coordinates
        world_corners = nib.affines.apply_affine(ref_affine, corner_points)
        
        # Get bounds
        bounds_min = world_corners.min(axis=0)
        bounds_max = world_corners.max(axis=0)
        
        # Clip streamlines to volume bounds
        clipped_streamlines = []
        for sl in streamlines:
            sl_array = np.array(sl)
            # Clip coordinates to bounds
            sl_clipped = np.clip(sl_array, bounds_min, bounds_max)
            clipped_streamlines.append(sl_clipped)
        
        # Create density map
        dm = utils.density_map(clipped_streamlines, 
                             vol_dims=ref_shape, 
                             affine=ref_affine)
        
        # Create binary map
        dm_binary = dm > 0
        
        # Create NIFTI image
        dm_binary_img = nib.Nifti1Image(dm_binary.astype("uint8"), ref_affine)
        
        return dm_binary_img
        
    except Exception as e:
        print(f"Error processing streamlines: {str(e)}")
        print(f"Reference shape: {ref_shape}")
        print(f"Number of streamlines: {len(streamlines)}")
        raise



def set_number_of_points(streamline,shape_streamline, n_points):
    """
    Resample a streamline to have a specified number of points.
    
    Parameters
    ----------
    streamline : ndarray (N, D)
        Array representing the streamline points in D dimensions
    n_points : int
        Number of points desired in the resampled streamline
        
    Returns
    -------
    ndarray (n_points, D)
        The resampled streamline
    """
    # Get shape information
    N = shape_streamline[0]  # number of points
    D = shape_streamline[1]  # number of dimensions
    
    # Initialize output array
    out = np.zeros((n_points, D))
    
    # Calculate arclengths
    def calculate_arclengths(points):
        distances = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
        arclengths = np.zeros(len(points))
        arclengths[1:] = np.cumsum(distances)
        return arclengths
    
    arclengths = calculate_arclengths(streamline)
    
    # Calculate step size for even spacing
    step = arclengths[-1] / (n_points - 1)
    
    # Initialize variables for resampling
    next_point = 0.0
    i = 0  # index for output points
    j = 0  # index for input points
    k = 0  # index for arclength checking
    
    # Main resampling loop
    while next_point < arclengths[-1]:
        if np.isclose(next_point, arclengths[k]):
            # If we're exactly at a point, use it
            out[i] = streamline[j]
            next_point += step
            i += 1
            j += 1
            k += 1
        elif next_point < arclengths[k]:
            # Interpolate between points
            ratio = 1 - ((arclengths[k] - next_point) /
                        (arclengths[k] - arclengths[k-1]))
            delta = streamline[j] - streamline[j-1]
            out[i] = streamline[j-1] + ratio * delta
            next_point += step
            i += 1
        else:
            j += 1
            k += 1
    
    # Ensure last point matches original streamline
    out[-1] = streamline[-1]
    
    return out