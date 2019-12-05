# AL
Simple Active Learning library that I created for my master thesis project.

# Introduction 
* The main script is the **run.py** file. Here you need to load the data en specify the different run options for AL experiments. 
* In the **al_cycle.py** file, different classes and functions for the conducting of the experiment are stored.
* The **models.py** file contains an example of a ML model object (in this case the Sklearn Naive Bayes Classifier) that is used in the experiments. Here you can specify your own ML model, using the desired external libraries. 
* The **acquisition_functions.py** file contains the different acquistion functions that are used in the experiment. As a benchmark, it is recommended to always include the RandomSelection function in your experiment. As with the ML models in models.py, you can easily create your own acquisition functions. 

### Credits: 
Main structure of this project was inspired by the following tutorial https://towardsdatascience.com/active-learning-tutorial-57c3398e34d by Ori Cohen.
