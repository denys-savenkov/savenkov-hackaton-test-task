# savenkov-hackaton-test-task
This github project is Savenkov's implementation of a hackathon test task (Machine Learning, Semantic Segmentation).

# Data
This project uses data from [Kaggle 2018 Data Science Bowl](https://www.kaggle.com/c/data-science-bowl-2018/data) competition.
From all the folders and files enlisted you will need to download only `stage1_train.zip` and unpack it in the `data` folder.
The reason to use only this file is that we need masks that aren’t anywhere else.

# Architecture
The main task of the project was to implement [UNet](https://arxiv.org/abs/1505.04597) architecture for Semantic Segmentation task:
![UNet Architecture](https://miro.medium.com/max/1412/1*f7YOaE4TWubwaFF7Z1fzNw.png)

# Initialization
* Create Python 3.6+ environment and install all requirements from `requirements.txt`
* Download data (see **Data** section
* Run `train.py` file to train the model (it will be saved to `model` folder)
* Run analysis.ipynb

# Results
Model achieves dice score equal to ~0.86 in 25-27 epochs
