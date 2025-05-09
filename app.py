import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize Porter Stemmer
ps = PorterStemmer()

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    # Remove non-alphanumeric characters
    text = [word for word in text if word.isalnum()]

    # Remove stopwords and punctuation
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]

    # Apply stemming
    text = [ps.stem(word) for word in text]

    return " ".join(text)


# Load vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit UI Configuration
st.set_page_config(page_title="Spam Detector", layout="centered")

# Main Title
st.markdown("<h1 style='text-align: center;'>📩 Email Spam Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Using NLP & Machine Learning to detect spam messages</p>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar Info
with st.sidebar:
    st.header("ℹ️ About")
    st.write("This app uses a trained machine learning model to determine whether a message is **Spam** or **Not Spam**.")
    st.markdown("🔍 Powered by: `TF-IDF`, `Naive Bayes`, and `NLTK`")

# Input Area
with st.container():
    st.subheader("️📝 Enter your message:")
    input_sms = st.text_area("", height=150, placeholder="Type or paste your message here...")

# Prediction
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.button("🔍 Predict"):
        if input_sms.strip():
            # Preprocess text
            transformed_sms = transform_text(input_sms)

            # Vectorize and predict
            vector_input = tfidf.transform([transformed_sms])
            result = model.predict(vector_input)[0]

            # Show result
            st.markdown("### 🧾 Prediction Result:")
            if result == 1:
                st.error("🚨 This message is **Spam**!")
            else:
                st.success("✅ This message is **Not Spam**.")
        else:
            st.warning("⚠️ Please enter a message first!")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 0.9em;'>Made with Streamlit</p>", unsafe_allow_html=True)
