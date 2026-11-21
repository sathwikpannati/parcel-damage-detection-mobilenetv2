# Data

The image dataset is not included in this repository. The notebook expects this folder layout next to it:

```
damaged-and-intact-packages/
├── damaged/   # 286 images of crushed, torn, wet or dented parcels
└── intact/    # 342 images of parcels in good condition
```

Images are resized to 224 × 224 RGB and split per class into 70% training, 15% validation and 15% test (`random_state=7`).
