## Check That Checker

This application can be used to train and test a machine learning algorithm to predict the "check worthiness" of statements of political speeches and debates.<br/>
Users can select feature groups, training datasets and testing datasets from a fixed selection and can also use their own dataset, as long as it complies with the required format.<br/>
They can then select which of the provided algorithms they want to use for the predictions.

### Getting Started

Unpack the application.zip and keep all files as they are in order to enable the application to access them correctly.<br/>
You will need to have Python version 3.7 or newer installed and need to set the Python working directory to the folder that you are running the application in.<br/>
Finally, you will just have to run the application using Python directly or a Python IDE of your choice.

### Using the application

The application is supposed to be rather self explanatory.<br/>
To select the fixed datasets included in the application, you can simply mark the corresponding checkboxes.
You need to pick at least one Feature Group, one Training Dataset and one Test Dataset.<br/>
The feature groups are shown as a short description of what the features in the group are focused on.
The datasets are shown as a short description of what kind of speech or debate the values were derived from.<br/>
All datasets are taken from the 2016 US elections.

#### Using custom datasets

Check That Checker also includes a simple option to use your own dataset in your predictions:<br/>
* custom datasets have to comply to the same structure as the fixed datasets
* custom datasets have to be provided as a "features" set and a "labels set".
* they have to be moved into the same folder as the fixed datasets
* to use them, you have to enter the exact name of the file into the corresponding field of the UI (including the .csv ending)
* only one pair of custom training sets and one pair of custom test sets can be used per run