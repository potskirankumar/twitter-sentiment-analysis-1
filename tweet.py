import pandas as pd
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

df= pd.read_csv("sentiment_tweets3.csv",index_col=None)

df=df.drop('Index',axis=1)

def remove_urls(text):
    pattern=re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub(r'',text)
    

def remove_htmls(text):
    pattern=re.compile('<.*?>')
    return pattern.sub(r'',text

wlm=WordNetLemmatizer()
def clean_text(text):
    text=text.lower()
    text= remove_urls(text)
    text=remove_htmls(text)
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        y.append(wlm.lemmatize(i))
    return " ".join(y)
    

df.rename(columns = {"message to examine": "message", "label (depression result)":"label"},inplace= True)

df['message']=df['message'].apply(clean_text)

tfidf=TfidfVectorizer()

X_tfidf=tfidf.fit_transform(df['message'])

Y=df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    Y,
    test_size=0.2,
    stratify=Y
)

classifier1=MultinomialNB()
classifier1.fit(X_train, y_train)
y_pred1=classifier1.predict(X_test)
print("Accuracy:",accuracy_score(y_test,y_pred1))


classifier2=BernoulliNB()
classifier2.fit(X_train, y_train)
y_pred2= classifier2.predict(X_test)

print("Accuracy:", accuracy_score(y_test,y_pred2))

# here BernoulliNB gives the highest accuracy because it is used for binary classification, this project is binary classification
#let's write for new predictions

input_text=input("Enter your text")

input_text1=clean_text(input_text)
tfidf_input=tfidf.transform([input_text1])
print(input_text1)
result=classifier2.predict(tfidf_input)[0]

if result==0:
    print("No Depression")
elif result==1:
    print("Depression")
