# Machine Learning using UFC data
==============================

Using Machine Learning to analyze mixed martial arts fights in the UFC
==============================

In this project, we use machine learning to accurately predict the outcome of a mixed martial arts fight in the UFC.  The data is first cleaned, then a baseline model is produced and finally a neural network is used to try to achieve the highest accuracy possible for predicting the outcome of a fight.  The slide deck describing the results can also be found in this repository


## Background
### Mixed Martial Arts
Throughout history numerous martial art forms have been developed, from boxing and muay thai to wrestling and jiu jitsu.  Mixed Martial Arts (MMA) is a sport which pits the best fighters of each discipline against one another.  It is a simple structure of one versus one, man vs man or woman vs woman. 
The Ultimate Fighting Championship (UFC) is the largest MMA promotion company in the world, and features the highest ranked fighters.  
### Project
Machine Learning is being used more and more throughout the sporting world, offering new insights into strategies for overcoming oppositions.  In this project I have decided to analyse the sport of MMA, using data from fights in the UFC.  By using several classification machine learning models including 2 neural networks, my goal is to predict the outcome of fights to the highest accuracy possible.  
Further work is then done to place bets on several fights, when my algorithm predicts a fight to a certain threshold of accuracy.

## Project Outcome
eXtreme Gradient Boosting (XG Boost) outperformed all of the other models, with an accuracy of 73.7%
![Model Comparison](/reports/figures/accuracy_of_machine_learning_models.png)

The model was improved with 

Project Organization
------------
The directory structure for this projects is below. Brief descriptions follow the diagram.

```
machine_learning_ufc_data
├── LICENSE
│
├── Makefile  : Makefile with commands to perform selected tasks, e.g. `make clean_data`
│
├── README.md : Project README file containing description provided for project.
│
├── .env      : file to hold environment variables (see below for instructions)
│
├── test_environment.py
│
├── data
│   ├── processed : directory to hold interim and final versions of processed data.
│   └── raw : holds original data. This data should be read only.
│
├── models  : holds binary, json (among other formats) of initial, trained models.
│
├── notebooks : holds notebooks for eda, model, and evaluation. Use naming convention yourInitials_projectName_useCase.ipynb
│
├── references : data dictionaries, user notes project, external references.
│
├── reports : interim and final reports, slides for presentations, accompanying visualizations.
│   └── figures
│
├── requirements.txt
│
├── setup.py
│
├── src : local python scripts. Pre-supplied code stubs include clean_data, model and visualize.
    ├── __make_data__.py
    ├── __settings__.py
    ├── clean_data.py
    ├── custom.py
    ├── model.py
    └── visualize.py

```

## Next steps
---------------
### Use with github
As part of the project creation process a local git repository is initialized and committed. If you want to store the repo on github perform the following steps:

1. Create a an empty repository (no License or README) on github with the name machine_learning_ufc_data.git.
2. Push the local repo to github. From within the root directory of your local project execute the following:

```
  git remote add origin https://github.com/(Your Github UserName Here)/machine_learning_ufc_data.git
  git push -u origin master
```

3. Create a branch with (replace ```branch_name``` with whatever you want to call your branch):
```
  git branch branch_name
```
4. Checkout the branch:
```
  git checkout branch_name
```

If you are working with a group do not share jupyter notebooks. The other members of the group should pull from the master repository, create and checkout their own branch, then create separate notebooks within the notebook directories (e.g., copy and rename the original files). Be sure to follow the naming convention. All subsequent work done on the project should be done in the respective branches.


### Environment Variables
-------------------
The template includes a file ```.env``` which is used to hold values that shouldn't be shared on github, for example an apikey to be used with an online api or other client credentials. The notebooks make these items accessible locally, but will not retain them in the online github repository. You must install ```python-dotenv``` to access this functionality. You can install it stand alone as follows:

```
  pip install -U python-dotenv
```
Or you can install all required packages with the instructions given in the next section.

#### Accessing Environment Variables in Jupyter Notebook
-------------
Notebook access to the constants and variables stored in the ```.env``` file is described here. In a code cell the line (e.g. assume you have a variable named ```APIKEY = 'Your_APIKEY'``` in the  ```.env``` file)
```
  mykey = %env APIKEY`  
```
will place the value ```'Your_APIKEY'``` into ```mykey```

### Installing development requirements
------------
If your current environment does not meet requirements for using this template execute the following:
```
  pip install -r requirements.txt
```
