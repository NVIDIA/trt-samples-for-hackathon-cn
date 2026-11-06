# Default Plugins

+ What ships in the TensorRT plugin registry, and what each one is for.

**Documentation only — there is nothing to run here.** The unit test for this directory is disabled
on purpose; the list below is a snapshot, and the script that produces it is quoted at the bottom.

## Measured on TensorRT 11.3.0.99

**45 distinct plugins, 70 creators** (a plugin may register several versions), of which **20
creators use the newer `IPluginCreatorV3One`** interface and the rest the older `IPluginCreator`.

On 11.0.0.114 these were 44 / 69 / 19. The one addition is **`FFTPlugin`** (version `1`, default
namespace, `IPluginCreatorV3One`, fields `inverse` / `onesided` / `ndims`) - so every creator added
since 11.0 arrived on the new interface, which is the direction the whole plugin API is moving.

These numbers change between releases — plugins are added, retired, and migrated to new interfaces —
so re-run the script below after an upgrade rather than trusting this page.

## You rarely create these by hand

Most of them are inserted for you. The ONNX parser recognises the corresponding operator or subgraph
and substitutes the plugin: `EfficientNMS_TRT`, `InstanceNormalization_TRT`, `ScatterND`,
`ROIAlign_TRT` all arrive that way. Building one explicitly with `add_plugin_v3` is for your *own*
operators, which is what the rest of `05-Plugin/` is about.

## Internal use only — do not call these (7)

```txt
MyelinWeightsTransformPlugin
CaskConvolutionRunnerWeightsTransformerPlugin
CaskConvActPoolWeightsTransformerPlugin
CaskDeconvV1RunnerWeightsTransformerPlugin
CaskDeconvV2RunnerWeightsTransformerPlugin
CaskDepSepConvWeightsTransformerPlugin
CaskFlattenConvolutionRunnerWeightsTransformerPlugin
```

They are weight-layout transformers used by the parser at build time to rearrange weights into the
form a kernel wants. Asking one for its field names says so plainly:

```txt
Error Code 1: Internal Error (Not implemented. This shall be used only in parser.)
```

So of the 45, **38 are yours to use**.

## Transformer / BERT (4)

| Plugin | Purpose | Key fields |
| :----- | :------ | :--------- |
| `CustomQKVToContextPluginDynamic` | fused multi-head attention — QKV to context in one kernel | `hidden_size`, `num_heads`, `dq_probs`, `use_int8_scale_max`, `use_explicit_int8` |
| `CustomEmbLayerNormPluginDynamic` | BERT embeddings: word + token-type + position lookup followed by LayerNorm, fused | `bert_embeddings_word_embeddings`, `..._position_embeddings`, `output_fp16`, `full_mask` |
| `CustomSkipLayerNormPluginDynamic` | residual add + LayerNorm, fused | `beta`, `gamma` |
| `CustomFCPluginDynamic` | fully-connected layer with the weights embedded, which is what makes INT8 practical | `out_dims`, `type_id`, `W` |

`CustomQKVToContextPluginDynamic` carries six versions and `CustomSkipLayerNormPluginDynamic` eight.
That version count is a decent proxy for how often something has been re-optimised.

## Detection: NMS, proposals, ROI (12)

`EfficientNMS_TRT` is the one you will actually meet, because the parser folds an ONNX NMS subgraph
into it: `score_threshold`, `iou_threshold`, `max_output_boxes`, `background_class`,
`score_activation`, `class_agnostic`, `box_coding`. `EfficientNMS_Explicit_TF_TRT` and
`EfficientNMS_Implicit_TF_TRT` are the two TensorFlow combined-NMS output conventions.

The Mask/Faster R-CNN pipeline, in the order it runs:

```txt
ProposalLayer_TRT           RPN proposals            prenms_topk, keep_topk, iou_threshold
MultilevelProposeROI_TRT    the FPN version of it    + fg_threshold
PyramidROIAlign_TRT         crop from the pyramid    fpn_scale, pooled_size, sampling_ratio
MultilevelCropAndResize_TRT the same idea            pooled_size, image_size
DetectionLayer_TRT          final boxes and classes  num_classes, keep_topk, score_threshold
GenerateDetection_TRT       ditto, + image_size
```

`RPROI_TRT` is the older all-in-one fusion of RPN and ROI pooling.

SSD-style prior boxes: `GridAnchor_TRT`, `GridAnchorRect_TRT`, `PriorBox_TRT`
(`minSize`, `maxSize`, `aspectRatios`, `featureMapShapes`, `variance`), with `FlattenConcat_TRT` to
join the per-level outputs. `ROIAlign_TRT` implements the standard ONNX `RoiAlign`.

## Normalization (2)

`InstanceNormalization_TRT` (`epsilon`, `scales`, `bias`, `relu`, `alpha` — note the fused
ReLU/LeakyReLU) and `GroupNormalizationPlugin` (`eps`, `num_groups`). Both arrive through the parser.

## Point cloud / 3D detection (3)

The PointPillars-style trio, in pipeline order:

```txt
VoxelGeneratorPlugin   point cloud -> voxels   point_cloud_range, voxel_size, max_num_points_per_voxel
PillarScatterPlugin    voxels -> pseudo-image  dense_shape
DecodeBbox3DPlugin     3D box decode           anchors, anchor_bottom_height, num_dir_bins, dir_offset
```

## Deformable convolution and sparse attention (3)

`ModulatedDeformConv2d` (DCNv2: `stride`, `padding`, `dilation`, `group`, `deformable_group`),
`MultiscaleDeformableAttnPlugin_TRT` (the core operator of Deformable DETR), and
`DisentangledAttention_TRT` (DeBERTa's disentangled attention: `span`, `factor`).

## Filling gaps in operator coverage (4)

`ScatterND`, `ScatterElements` (`reduction`, `axis`), `TopkLastDim` (`k`, `is_largest`, `axis`),
`ResizeNearest_TRT` (`scale`), `CropAndResizeDynamic` (`crop_width`, `crop_height`).

## YOLO and model-specific fusions (4)

`Region_TRT` and `Reorg_TRT` are the YOLOv2/v3 region and passthrough layers (`num`, `coords`,
`classes`, `smTree`; `stride`). `RnRes2FullFusion_TRT` fuses a whole ResNet res2 block — its field
list spells out every branch, `c_res2a_br1_w`, `s_res2a_br1_s` and so on. `DLRM_BOTTOM_MLP_TRT` is
the bottom MLP of the DLRM recommender, weights and dynamic ranges embedded.
`SmallTileGEMM_TRT` is a small-tile GEMM for MLP-shaped work.

## Old and new creator interfaces coexist

20 of the 70 creators implement `IPluginCreatorV3One`; the rest are the older `IPluginCreator`. Where
one plugin has both, the higher version number is usually the new interface — `ROIAlign_TRT` v1 is
`IPluginCreator` and v2 is `IPluginCreatorV3One`. That matters when pinning a version:
see [`../MultiVersion/`](../MultiVersion/README.md).

## Regenerating this list

```python
import tensorrt as trt
from collections import OrderedDict

trt.init_libnvinfer_plugins(trt.Logger(trt.Logger.ERROR), "")
registry = trt.get_plugin_registry()

version_map = OrderedDict()
for creator in registry.all_creators:
    version_map.setdefault(creator.name, []).append(str(creator.plugin_version))
    # `creator.field_names` gives the parameters; the internal ones raise instead of answering
print(f"{trt.__version__}: {len(version_map)} plugins, {sum(len(v) for v in version_map.values())} creators")
```

## Related

+ [`../MultiVersion/`](../MultiVersion/README.md) — selecting among several versions of one plugin.
+ [`../ONNXParserWithPlugin/`](../ONNXParserWithPlugin/README.md) — how the parser substitutes a
  plugin for an operator it does not implement natively.
+ [`../BasicExample/`](../BasicExample/README.md) — writing one of your own.
