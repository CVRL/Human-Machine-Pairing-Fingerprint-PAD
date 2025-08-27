# Saliency-Guided Training for Fingerprint Presentation Attack Detection

Official GitHub repository for the paper: Samuel Webster and Adam Czajka, "Saliency-Guided Training for Fingerprint Presentation Attack Detection," IEEE/IAPR International Joint Conference on Biometrics, Osaka, Japan, September 8-11, 2025 **([ArXiv](https://www.arxiv.org/abs/2505.02176) | [IEEEXplore](https://ieeexplore.ieee.org/)**

![Paper teaser graphic](teaser.jpg)

## Table of contents
* [Abstract](#abstract)
* [Datasets](#datasets)
* [Model weights](#weights)
* [Citation](#citation)
* [Acknowledgment](#acknowledgment)

<a name="abstract"/></a>
## Abstract

<a name="datasets"/></a>
## Datasets

### Human Fingerprint Annotation Dataset

In our acquisition of human-annotative saliency, we conducted a 50-participant fingerprint annotation collection. Participants annotated samples from the 2015, 2017, 2019, and 2021 editions of the LivDet-Fingerprint competition. Each participant hand-annotated 16 bonafide and 16 spoof samples, producing 800 doubly-annotated fingerprints. Additionally, individual annotation stroke times (mouse down/up) are recorded, participants predicted the liveness (bonafide/spoof) for each sample, and, participants could textually describe their annotated region. 

We offer human annotations with the paper. However, we are not allowed (according to the dataset sharing license) to re-release the official LivDet-Fingerprint competition datasets. To access original files, please follow instructions provided by the competition organizers at [LivDet datasets page](https://livdet.org/registration.php).

### Algorithmically-sourced Pseudosaliency

Our experiments use pseudosaliency produced via various algorithmic means: minutiae-based regions (Neurotechnology VeriFinger SDK), low-quality maps (NIST Biometric Image Software), and human-mimicking autoencoder annotations. We offer these pseudosaliency maps with the paper.

### Obtaining Copies of the Datasets

Instructions on how to request a copy of the synthetic iris dataset used in this paper can be found at [the CVRL webpage](https://cvrl.nd.edu/projects/data/) (look for ND-FINGER-IJCB-2025 Dataset).

<a name="weights"/></a>
## Trained Fingerprint PAD models

Across all experiments, we train 720 individual models with varying configurations based on our five outlined scenarios. The models are available in this [box folder]().

<a name="citation"/></a>
### Citation

If you find this work useful in your research, please cite the following paper:
```
@inproceedings{mitcheff2024privacysafeirispresentationattack,
      title={Privacy-Safe Iris Presentation Attack Detection}, 
      author={Mahsa Mitcheff, Patrick Tinsley and Adam Czajka},
      year={2024},
      booktitle={IEEE International Joint Conference on Biometrics},
}
```

<a name="acknowledgment"/></a>
### Acknowledgment
This material is based upon work partially supported by the National Science Foundation under Grant No. 2237880. Any opinions, findings, and conclusions
or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation.
