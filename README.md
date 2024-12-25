# FLPCS-MRF

# 1. Overview
As the aging population and chronic diseases increase, fall detection plays a crucial role in elderly healthcare monitoring. In particular, efficiently and accurately detecting falls while ensuring data privacy has become a significant challenge in complex healthcare environments. To address this challenge, this paper presents an innovative federated learning-based fall detection framework with multimodal residual fusion and Pareto optimized client selection (PLPCS-MRF). The framework first designs a deep neural network with a residual mechanism to effectively fuse multimodal data, capturing complementary information across modalities to enhance fall detection accuracy. Then, the federated learning framework enables the task initiator to distribute the model to multiple clients for local training while preserving data privacy. For global model aggregation, this work focuses on local model convergence and generalization ability, defining multiple evaluation metrics to assess client models comprehensively. Pareto optimization is then applied to select clients that achieve a balanced performance across these factors, ensuring the quality and generalization of the aggregated global model. Experimental results demonstrate that the proposed method significantly outperforms existing approaches and exhibits strong robustness in complex data environments, such as incomplete modalities and uneven data distributions.

# 2. Data Preparation
Follow the tutorial [here](https://github.com/jpnm561/HAR-UP/tree/master/DataBaseDownload/) to download the FALL-UP dataset and place the files into the "dataset" folder.

# 3. File Descriptions
- **requirements.txt**: Dependency environment setup.
- **data-preprocessingToSub.py**: Data preprocessing, removing subjects with missing data, and keeping data from 12 subjects.
- **testResModel.py**: Defines the structure and hyperparameters of the local model.
- **tmodel.py**: Fall detection using time-series data.
- **c1model.py**: Fall detection using visual data from Camera 1.
- **c2model.py**: Fall detection using visual data from Camera 2.
- **tc1model.py**: Fall detection using a combination of time-series data and visual data from Camera 1.
- **tc2model.py**: Fall detection using a combination of time-series data and visual data from Camera 2.
- **tc1c2ResmodelDataV1.py**: Federated learning with data distribution across 12 clients from 12 subjects. Trials 1 and 2 of each subject are used for training, and Trial 3 from all subjects is used as the validation set for comparative experiments.
- **tc1c2ResmodelDataV2.py**: Federated learning with data distribution across 12 clients, using data from 11 subjects for training, with the data from the last subject used as the validation set. This script also handles data distribution for robustness testing.
