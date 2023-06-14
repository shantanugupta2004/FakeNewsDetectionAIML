import joblib
import pandas as pd
import re
import string
import tkinter as tk
from tkinter import messagebox

LR_model = joblib.load('finalLR_model.sav')
DT_model = joblib.load('finalDT_model.sav')
vectorization = joblib.load('vectorizer.sav')

def word_clean(text):
  text = text.lower()
  text = re.sub('\[.*?\]', '', text)
  text = re.sub("\\W", " ", text)
  text = re.sub('<.*?>+', '', text)
  text = re.sub('[%s]'%re.escape(string.punctuation),'', text)
  text = re.sub('\n', '', text)
  text = re.sub('\w*\d\w*', '', text)
  return text

def output_label(n):
    if n==0:
        return "Fake News"
    elif n==1:
        return "True News"

def manual_testing(news):
  testing_news = {"text": [news]}
  new_def_test = pd.DataFrame(testing_news)
  new_def_test["text"] = new_def_test["text"].apply(word_clean)
  new_x_test = new_def_test["text"]
  new_xv_test = vectorization.transform(new_x_test)
  pred_LR = LR_model.predict(new_xv_test)
  pred_DT = DT_model.predict(new_xv_test)

  return "\n\nLR Prediction: {} \nDT Prediction: {}".format(output_label(pred_LR[0]), output_label(pred_DT[0]))

def test_news():
    news = user_entry.get("1.0", "end-1c")
    result = manual_testing(news)
    messagebox.showinfo("News Classification Result", result)



window = tk.Tk()
window.title("Fake News Classification")
window.geometry("500x300")


label = tk.Label(window, text="Enter the news to be tested:")
label.pack()

user_entry = tk.Text(window, height=5, width=40)
user_entry.pack()

classify_button = tk.Button(window, text="Classify", command=test_news)
classify_button.pack()


window.mainloop()