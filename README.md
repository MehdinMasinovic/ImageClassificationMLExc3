# ImageClassificationMLExc3
This project folder / repository stores all codes and data used for the third exercise of ML in the WS 2021/2022 for Group 1. Overall, there are two main implementations, one for the traditional methods and one for the deep learning approach. 

You can also visit our project repository on Github: https://github.com/MehdinMasinovic/ImageClassificationMLExc3

# Dependencies
Before running any scripts, please make sure that you have installed the required packages and versions from the "requirements.txt" file in this repository using:

```{sh}
pip3 install -r /path/to/requirements.txt
```

In case the command above raises an error regarding permissions, please try the command below:
```{sh}
pip3 install -r requirements.txt --user
```

# Running the whole analysis
To run the full analysis, traditional and deep learning, please use the main.sh file, which will execute all necessary commands. 

# Notes on the traditional approach
Please note that for the traditional approach, you have several command line options to vary e.g. the number of words in the clustering, or the number of threads that can be used to run the program in parallel (6 is set in the shell scripts).

In the "configs/" folder, there is a list of JSON files that store the configurations for individual models, which will be loaded into the pipeline once the shell script is executed. 

If you would like to execute the script for a single model, you can do so by varying the "--path-to-config" command line option and define the list of models or settings of a model that you would like to test. You can find examples of how to execute the pipeline in the "main_traditional_clothes.sh" and "main_traditional_faces.sh" files.

The pipeline loads and processes the data, and then trains the model. The model metrics and results are stored in individual folders stored in the "results/" folder. In every folder, you will find prediction results, confusion matrices, plots, and runtime information for each model setting. Note that if you run the same configurations for a single model multiple times, the old results will be overwritten. 

Generally the code is structured in a way that the "main_ml_exc3.py" is the main wrapper script which imports other modules and functions from the "utils/" folder, depending on the step in the analysis pipeline.

# Notes on the deep learning approach
For the deep approach you have two Python files located in the cnn directory, namely cnn_mgooglenet.py and cnn_mvggnet.py. Each of them runs all the setups that were used for this paper, for the MiniGoogLeNet and the MiniVGGNet architecture. This makes up 16 training different models with different setups in each of the scripts. Output images will be saved in a subdirectory named cnn_images/. The text results will be printed in the command line.

For more flexibility and user friendliness, the directory provides two Jupyter notebooks that the Python files were based on which is more descriptive. In their default state, the provided notebooks also provide the whole overview of results and images that were generated by running each of the setups.

It's possible to vary the parameters in the code blocks featuring calls to the wrapper_f() function to vary dataset fetching function, epoch count, learning rate, batch size and augment options. Per default, all the results get directly printed in the output block.
