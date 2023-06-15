# FakeNewsDetectionAIML
# Fake News Classification

This project implements a machine learning model to classify news articles as either fake or not fake (real). The model is trained using the Logistic Regression and Decision Tree algorithms. The implementation includes a GUI application built with Tkinter for real-time classification of news articles.

## Dataset

The dataset used for training and testing the model consists of a csv file

`News.csv`: It contains around 44000 news articles classifies as real or fake.

The dataset is preprocessed to remove unnecessary columns and clean the text data before training the model.

## Dependencies

The following dependencies are required to run the project:

- Python (3.7+)
- pandas
- seaborn
- matplotlib
- scikit-learn
- tkinter

You can install the required dependencies using pip:

pip install pandas seaborn matplotlib scikit-learn tkinter


## Usage

1. Ensure that the dataset file (`News.csv`) are present in the same directory as the `app.py` file.

2. The dataset can be downloaded from `https://drive.google.com/file/d/1q5jpI5M1EA9x3YPrLupmiu3gffkmGlHj/view` 

3. Run the `app.py` file using Python:

4. The GUI window will open. Enter the news article text to be classified in the provided input field.

5. Click the "Classify" button to classify the news article as fake or not fake.

6. A message box will display the classification result.

## Pre-trained Models

The trained models are saved in the following files:

- `finalLR_model.sav`: Pre-trained Logistic Regression model.
- `finalDT_model.sav`: Pre-trained Decision Tree model.
- `vectorizer.sav`: The fitted dataset in the models.

The models are loaded at runtime if the corresponding model files exist. If the model files are not found, the models are trained and saved for future use.


Feel free to modify and use the code according to your requirements.





