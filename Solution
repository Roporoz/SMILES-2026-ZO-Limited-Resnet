# Solution

## Main idea

The project budget limits how many training samples can be used during the zero-order fine-tuning stage. Therefore, my initial strategy was to split the available resources into two parts:

1. use part of the budget to build a strong initialization of the CIFAR-100 classification head;
2. reserve the remaining budget for zero-order fine-tuning.

The key observation is that the pretrained ImageNet ResNet-18 backbone already extracts useful visual features. The main weakness of the original model is the final ImageNet classification head, which maps these features to 1000 ImageNet classes rather than to the 100 CIFAR-100 classes.

Therefore, instead of starting from a random CIFAR-100 head, I first construct a supervised linear classifier on top of frozen ResNet features.

The pipeline is:

1) train a linear classifier on a balanced CIFAR-100 subset
2) copy the learned weights into model.fc
3) optional zero-order fine-tuning


## Choosing the number of samples for head initialization

As a first step, I studied how many CIFAR-100 training images are needed to obtain a useful initialization of the final classification head. For this experiment, only the final linear head was initialized from class prototypes. No zero-order fine-tuning was applied at this stage.

I varied the number of training images per CIFAR-100 class and measured the resulting Top-1 accuracy of the initialized head.

| Images per class | Total images | Top-1 accuracy |
|---:|---:|---:|
| 5 | 500 | 37.4% |
| 10 | 1000 | 44.1% |
| 20 | 2000 | 47.2% |
| 40 | 4000 | 50.3% |
| 60 | 6000 | 50.8% |
| 80 | 8000 | 51.3% |

The results show that the performance grows quickly at first but then starts to saturate. In particular, using 40 images per class already gives a result close to using 80 images per class. 

This observation motivated the initial strategy of splitting the budget into two parts: using roughly half of the samples to build a strong head initialization and reserving the remaining half for zero-order fine-tuning.

## Head selection strategy


### Prototype head

The first approach was a prototype classifier. For each CIFAR-100 class, I computed the average frozen feature vector:

`prototype_c = mean(features of class c)`

Then each prototype was used as the corresponding row of the final linear head.

This already gave a large improvement over the original ImageNet head:

| Method | Images/class | Top-1 accuracy |
|---|---:|---:|
| ImageNet head | - | 0.37% |
| Prototype | 40 | ~50.3% |
| Prototype | 80 | ~51.3% |

This showed that the pretrained ResNet-18 backbone already contains useful information for CIFAR-100. However, the prototype head saturated quickly and was not sufficient to separate visually similar classes.

Typical confusions included:

`otter ↔ seal`  
`ray ↔ shark`  
`girl ↔ woman`  
`maple_tree ↔ oak_tree`  
`bus ↔ streetcar`

This motivated using discriminative linear classifiers.

---

### Ridge classifier

The next method was a Ridge classifier trained on the same frozen features.

| Method | Images/class | Top-1 accuracy |
|---|---:|---:|
| Ridge classifier | 40 | ~55.6% |

This improved significantly over prototypes, but it was still weaker than later methods.

---

### Linear Discriminant Analysis

Linear Discriminant Analysis was the first strong head initialization method. I used:

`LinearDiscriminantAnalysis(solver="lsqr", shrinkage=0.3)`

| Method | Images/class | Top-1 accuracy |
|---|---:|---:|
| LDA | 40 | ~58.0% |

This became the first strong baseline.

---

### LDA + scaled Logistic Regression blend

The best head was obtained by blending LDA and scaled Logistic Regression:

`W_final = alpha_LDA W_LDA + (1 - alpha_LDA) W_LogReg`  
`b_final = alpha_LDA b_LDA + (1 - alpha_LDA) b_LogReg`

The best blend used mostly Logistic Regression with a small LDA contribution:

`alpha_LDA = 0.10`  
`alpha_LogReg = 0.90`

The intuition is that Logistic Regression provides a strong discriminative classifier, while LDA adds a small stabilizing contribution based on class means and covariance structure.

This was the strongest head initialization method.

| Method | Images/class | Top-1 accuracy |
|---|---:|---:|
| Prototype | 80 | ~51.3% |
| Ridge classifier | 40 | ~55.6% |
| LDA | 40 | ~58.0% |
| LDA + scaled LogReg blend | 40 | 58.78% |
| LDA + scaled LogReg blend | 80 | 61.72% |

Therefore, the final solution uses the LDA + scaled Logistic Regression blend as the head initialization method.

# Zero-order fine-tuning attempts

After obtaining a strong initialized head, I tried to improve it further with zero-order fine-tuning. The main target was the final classification layer initialized by the LDA + scaled Logistic Regression blend.

The motivation was the following: although the blended head already performed well, the remaining errors were not uniformly distributed across all CIFAR-100 classes. Only a relatively small subset of classes had low per-class accuracy. Therefore, instead of optimizing the whole final layer, I tried to use ZO updates only for the most difficult classes.

The full final layer has:

`W: 100 × 512`

which corresponds to 51200 weight parameters. This is too many parameters for a small ZO budget. To reduce the dimensionality of the optimization problem, I used a low-rank correction:

`W_new = W_base + A B`

where:

`W_base` is the initialized head,  
`A` is optimized by ZO,  
`B` is a fixed low-rank basis.

The first idea was to use a random low-rank basis. However, random directions are not necessarily aligned with the actual CIFAR-100 feature distribution. Therefore, I also tested a PCA-based basis.

The PCA basis was computed from frozen ResNet-18 feature vectors. This allowed the ZO optimizer to move the head weights along directions that correspond to real variations in the data:

`W_bad_new = W_bad_base + scale · A_bad B_PCA`

Here, only the rows corresponding to difficult classes were updated. This reduced the number of ZO parameters from the full `100 × 512` matrix to approximately:

`number_of_bad_classes × rank`

For example, if around 30 difficult classes are updated with `rank = 32`, the optimizer only controls about:

`30 × 32 = 960 parameters`

This is much more suitable for zero-order optimization than the full head.

I also modified the ZO training batches to focus more on hard classes and their main confusion partners. The idea was to spend the ZO budget on the classes where the initialized head still made most of its mistakes.

However, this strategy did not improve the final validation score. In practice, the ZO updates tended to optimize the noisy mini-batch loss but did not generalize to the full validation set. The hard-class PCA low-rank ZO variant slightly degraded the result:

| Method | Top-1 accuracy |
|---|---:|
| Blend40 initialized head | 58.78% |
| Blend40 + hard-class PCA low-rank ZO | 58.45% |

This suggests that the supervised head initialization was already well calibrated, while the ZO updates were too noisy and too local to improve the global validation accuracy reliably.

As a result, I did not use ZO fine-tuning in the final submission. The final solution uses the stronger `80 images/class` blended head and a no-op ZO optimizer.

## Error analysis and possible extensions

The remaining errors were mostly concentrated in visually and semantically similar CIFAR-100 classes. This was already visible for the prototype head and remained a useful diagnostic for later experiments.

Typical confusion pairs included:

| True class | Common wrong prediction |
|---|---|
| `otter` | `seal` |
| `seal` | `otter` |
| `ray` | `shark` |
| `turtle` | `shark` |
| `flatfish` | `ray` |
| `girl` | `woman` |
| `boy` | `man` |
| `maple_tree` | `oak_tree` |
| `pine_tree` | `oak_tree` |
| `willow_tree` | `oak_tree` |
| `bus` | `streetcar` |
| `train` | `streetcar` |
| `beetle` | `cockroach` |
| `butterfly` | `caterpillar` |
| `crab` | `lobster` |
| `lizard` | `crocodile` |
| `table` | `bed` |

These errors suggest that the frozen ResNet-18 feature space already separates broad semantic categories reasonably well, but still struggles with fine-grained distinctions inside the same visual group, such as:

`aquatic animals`, `trees`, `vehicles`, `people`, `insects`, and `furniture`.

I also considered using data augmentation during head initialization, for example random crops, horizontal flips, or multi-scale feature extraction. However, I did not include augmentation in the final solution.


---

## Final conclusion

The final solution uses a strong supervised initialization of the last linear layer.

The method included in the final solution is:

`LDA + scaled Logistic Regression blend`

with the following configuration:

| Component | Value |
|---|---:|
| Total images used for head initialization | 8000 |
| LDA shrinkage | 0.3 |
| Logistic Regression C | 0.020 |
| LDA weight in blend | 0.10 |
| Logistic Regression weight in blend | 0.90 |

The final head is:

`W_final = 0.10 W_LDA + 0.90 W_LogReg`

`b_final = 0.10 b_LDA + 0.90 b_LogReg`

The zero-order optimizer is kept as a no-op in the final submission, because the tested ZO variants did not reliably improve the validation score.

Final official validation result:

| Checkpoint | Top-1 accuracy |
|---|---:|
| Baseline ImageNet head | 0.37% |
| Initialized CIFAR-100 head | 61.72% |
| Fine-tuned head | 61.72% |

Thus, the best use of the budget was to build a strong linear probe on frozen ResNet-18 features rather than to spend samples on noisy zero-order fine-tuning.
