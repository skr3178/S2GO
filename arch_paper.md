On nuScenes, S2GO uses a 256x704 resolution image and is pre-trained on denoising and rendering for 12
epochs without semantic annotations, and then trained for 24 epochs for 3D semantic occupancy estimation.
S2GO-Small uses an ImageNet1k backbone, while S2GO-Base leverages nuImages pre-training. On KITTI,
we use a 256x1408 resolution image and an ImageNet1k backbone. The model is pre-trained for 12 epochs,
then trained for occupancy for another 12 epoch.

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
interact with the image through Deformable Attention (Zhu et al., 2020; Lin et al., 2022; Wang et al., 2023).

In this section, we verify the effectiveness of our proposed components. By default, models are
trained for 12 epochs during both stages. All ablations are on the SurroundOcc-nuScenes dataset.
Pretraining. In Table 3, we ablate the impact of pretraining. First, directly training occupancy
estimation for 12 epochs (a) or 24 epochs (a)† yields poor results due to ambiguous supervision.

Next, we pretrain with depth and RGB supervision and ablate query initialization. Learnable ini-
tialization during pretraining – which is what S2GO uses in the second stage – is worse than not
pretraining. This occurs because such queries are randomly distributed over 3D space, with most
queries far from occupied geometry and hence unable to get adequate supervision. On the other
hand, initializing query locations precisely at LiDAR points is only slightly better than not pretrain-
ing – this baseline supervises Gaussians to capture local geometry, but the queries themselves are
not supervised to move. Finally, adding noise to LiDAR before initializing achieves remarkable
performance, providing meaningful supervision to both queries and Gaussians. We emphasize that
this is the only initialization method that substantially improves over not pretraining with the same
compute budget (24 epochs of occupancy in (a)† vs 12+12 epochs with pretraining).
Next, we ablate each pretraining loss function. Depth supervision alone is enough to achieve good
performance. Adding RGB loss slightly boosts results as RGB supervises finer details, and denoising
supervision gives a substantial final boost. This confirms that our proposed pretraining is essential
for our streaming, sparse-query framework to reach its potential and achieve state-of-the-art.

While S2GO can directly be trained for occupancy estimation, the resulting performance is subopti-
mal. The queries and their Gaussians are unable to move effectively to occupied locations to capture
fine details – they instead coarsely model nearby regions as shown in Figure 2. This stems from the
weak and ambiguous supervision that queries and Gaussians receive from occupancy labels.
This limitation arises from two interconnected factors: First, unlike in GaussianFormer where each
Gaussian is refined individually, in our sparse query-based framework, each query moves J Gaus-
sians as a group before individual Gaussians locally branch out. As any perturbations to query
location propagate to its constituent Gaussians, aligning the query precisely with scene geometry
before predicting Gaussian offsets is critical. However, 3D occupancy estimation lacks a clear as-
signment between parts of the scene and individual queries – with multiple nearby scene elements,
the lack of clear-cut supervision causes query refinements to be noisy. Second, this ambiguity is
exacerbated by the inherent locality of the Gaussian-to-voxel splatting operation in Section 3.1. As