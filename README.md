# Saliency-Guided Training for Fingerprint Presentation Attack Detection

![Paper teaser graphic](teaser.jpg)

Official repository for "Saliency-Guided Training for Fingerprint Presentation Attack Detection."

Accepted to the 2025 IEEE International Joint Conference on Biometrics (IJCB) conference.

Read the paper at: [[IEEEXplore (once available)]](https://ieeexplore.ieee.org/) | [[ArXiv pre-print]](https://www.arxiv.org/abs/2505.02176).

## Contents

### Human Fingerprint Annotation Dataset

In our acquisition of human-annotative saliency, we conducted a 50-participant fingerprint annotation collection. Each participant hand-annotated 16 bonafide and 16 spoof samples, producing 800 doubly-annotated fingerprints. Additionally, individual annotation stroke times (mouse down/up) are recorded, participants predicted the liveness (bonafide/spoof) for each sample, and, participants could textually describe their annotated region. 

As of 7/25/2025, the dataset is not yet publicly available. Once available, to access our dataset, please visit [the Notre Dame Computer Vision Research Lab datasets page](https://cvrl.nd.edu/projects/data/) and follow the page's licensing instructions. 

In our study, participants annotated samples from the 2015, 2017, 2019, and 2021 editions of the LivDet-Fingerprint competition. We are not able to re-release the official competition datasets. To access them, please follow instructions provided by the competition organizers at [LivDet datasets page](https://livdet.org/registration.php).

### Algorithmically-sourced Pseudosaliency

Our experiments use pseudosaliency produced via various algorithmic means: minutiae-based regions (Neurotechnology VeriFinger SDK), low-quality maps (NIST Biometric Image Software), and human-mimicking autoencoder annotations. 

Details on accessing pseudosaliency will be available soon.

### Trained Fingerprint PAD models

Across all experiments, we train 720 individual models with varying configurations based on our five outlined scenarios. 

Details on accessing trained models will be available soon.