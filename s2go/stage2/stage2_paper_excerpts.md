# Stage2 excerpts from the paper
3.4 STAGE 2: 3D SEMANTIC OCCUPANCY ESTIMATION
## 3.4.1 OCCUPANCY ESTIMATION FRAMEWORK
Equipped with the pretraining prior, S2GO is then trained for 3D semantic occupancy estimation.
The model processes image observations, predicts a set of Gaussians Gt at each timestep, which now
also include semantic class predictions, and “splats” Gaussians to nearby voxels as in Section 3.1.
Notably, unlike the pre-training phase, query positions are in
itialized at learnable 3D locations. As
such, S2GO only uses RGB images during inference. The “splatted” voxel predictions are trained
using ground truth semantic occupancy, and we additionally supervise neighboring frames similar
to Stage 1. In this section, we propose crucial improvements to further strengthen this pipeline.

## 3.4.2 OPACITY-WEIGHTED GEOMETRY ESTIMATION
The Gaussian-to-voxel splatting framework in GaussianFormer-2 handles foreground classes as a
mixture of Gaussians, and opacity is only used for weighting Gaussians inside the mixture. As such,
opacity has no bearing on determining binary occupancy of a location, in contrast to Gaussians in
rendering (Kerbl et al., 2023) where opacity acts as a proxy for density. This leads to unexpected
behavior: Gaussians in unoccupied regions decrease their scale s and position themselves between
voxel centers to minimize their occupancy contribution (Eq. 2), all while maintaining significant
opacity. This unnatural representation for Gaussians conflicts with the rendering initialization and
hurts performance. To address this issue, we additionally weight the occupancy probability α(x; G)
with the opacity estimation a, yielding:
1
α(x; G) = aexp−
2(x−m)TΣ−1(x−m) (9)
Our change significantly improves performance by encouraging Gaussians in unoccupied regions to
simply predict lower opacity and stabilizing scale supervision to be more consistent.
## 3.4.3 EFFICIENT GAUSSIAN-TO-VOXEL SPLATTING
To implement Gaussian-to-voxel splatting, GaussianFormer (Huang et al., 2024) first determines
pairs of interacting Gaussians and voxels, then parallelizes over voxels in the forward pass and
over Gaussians in the backward pass. However, this formulation does not account for the inherent
locality of the splatting operation: neighboring voxels process a similar set of Gaussians and vice-
versa. Such voxels and Gaussians should be processed together in a CUDA block for optimized L1
cache usage. This is especially a problem for the backward pass since naively parallelizing over
Gaussians incurs random access costs on a large number of voxels (640k).
To address this problem in the forward pass, we block voxels into 4x4x4 grids and have threads tied
to each voxel collaboratively load nearby Gaussians onto memory before splatting them, similar
to 3DGS (Kerbl et al., 2023). In the backward pass, we adopt a similar approach but addition-
ally take care to tie threads to individual Gaussians to avoid atomic operations on the gradients
(Mallick et al., 2024). Our efficient Gaussian-to-voxel splatting implementation, with 9k Gaussians
and 640k voxels on an A100, speeds up the forward pass by 1.5×(1.29ms to 0.87ms) and the
backward pass by 20.4×(116ms to 5.7ms), substantially reducing the wall-clock time required for
training. Furthermore, it decreases the GPU memory cost of the forward/backwards pass by 3.3×
from 2013MB/2079MB to 631MB/633MB.
## 3.4.4 QUERY PROPAGATION
A key point in our streaming 3D occupancy pipeline is query propagation. More specifically, we
need to determine the optimal subset of current queries to push onto the queue for future timesteps.
While a straightforward selection of top-k largest query opacities works, propagating the most confi-
dently occupied regions of the scene, we find that queries end up highly overlapping over time, with
insufficient coverage over the entire scene. To mitigate this, we choose the highest opacity queries
that are pairwise separated by a distance δ, where δis a hyperparameter. This maintains an effective
balance between maintaining high-opacity regions and distributing queries across the scene.

## Evaluation metrics:

Table 1: 3D occupancy estimation results on the nuScenes-SurroundOcc validation set

S2GO-Base (ours) C 40.8 15.1 22.7 1.3 1.7 15.9 5.1 2.1 53.8 13.3 33.4 3.8 35.3 7.2 31.2 21.1 6.4 6.5 6.0 4.2

## Better to first run the gaussianFOrmer2 on these metrics and check for validity? 

The temporal transformer closely follows the design from PETR (Liu et al., 2022) and StreamPETR (Wang
et al., 2023), with a 4-frame (2 second) queue. Each transformer block consists of a self-attention layer across
all the queries, followed by a cross-attention layer to refine queries based on image features, and a feedforward
layer. The self attention layer includes keys and queries derived from past queries as in StreamPETR, and we
use deformable attention for the cross attention layer for efficiency. All models are trained with a 4e-4 learning
rate with a batch size of 16, with the cosine annealing schedule and the AdamW optimizer with weight decay
0.01. We use gradient clipping with max norm 35, and scale the learning rate for the backbone by a factor 0.25.
All of our models are trained with mixed precision. On nuScenes-SurroundOcc, the LiDAR nosing factor ϵ
is set to 1 meter. During training, the pairwise query distance δ for query propagation is randomly sampled
between 0 to 3 meters, and during inference, it is set to 1.6m. For nuScenes-Occ3D and KITTI, all distances are
scaled according to the smaller extent of the 3D scene. The embedding dimension of the temporal transformer
is 768, and we leverage Flash Attention (Dao et al., 2022) for efficient self-attention between queries. Queries
interact with the image through Deformable Attention (Zhu et al., 2020; Lin et al., 2022; Wang et al., 2023)