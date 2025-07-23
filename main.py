import streamlit as st
import pickle
import re
import nltk
import PyPDF2

nltk.download('punkt')
nltk.download('stopwords')

# Load models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

def cleanResume(txt):
    txt = txt.lower()
    cleanTxt = re.sub(r'http\S+', '', txt)
    cleanTxt = re.sub(r'\bRT\b|\bCC\b', '', cleanTxt)
    cleanTxt = re.sub(r'#\w+', '', cleanTxt)
    cleanTxt = re.sub(r'\S*@\S+', '', cleanTxt)
    cleanTxt = re.sub(r'[%s]' % re.escape("""!#$%^&*()_+={}[]|\:;'"<>,.?/~`-"""), '', cleanTxt)
    cleanTxt = re.sub(r'[^\x00-\x7f]', '', cleanTxt)
    cleanTxt = re.sub(r'\s+', ' ', cleanTxt).strip()
    return cleanTxt

def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
        return text.strip()
    except Exception as e:
        return ""

def extract_text_from_txt(file):
    try:
        resume_bytes = file.read()
        return resume_bytes.decode('utf-8')
    except UnicodeDecodeError:
        return resume_bytes.decode('latin-1')

def main():
    st.title("Resume Screening App")
    uploaded_file = st.file_uploader('Upload Resume', type=['txt', 'pdf'])

    if uploaded_file is not None:
        file_type = uploaded_file.name.split('.')[-1].lower()
        if file_type == 'pdf':
            txt = extract_text_from_pdf(uploaded_file)
        else:
            txt = extract_text_from_txt(uploaded_file)

        cleaned_resume = cleanResume(txt)
        input_features = tfidf.transform([cleaned_resume])
        prediction_id = clf.predict(input_features)[0]

        category_mapping = {
            6: 'Data Science',
            12: 'HR',
            0: 'Advocate',
            1: 'Arts',
            24: 'Web Designing',
            16: 'Mechanical Engineer',
            22: 'Sales',
            14: 'Health and fitness',
            5: 'Civil Engineer',
            15: 'Java Developer',
            4: 'Business Analyst',
            21: 'SAP Developer',
            2: 'Automation Testing',
            11: 'Electrical Engineering',
            18: 'Operations Manager',
            20: 'Python Developer',
            8: 'DevOps Engineer',
            17: 'Network Security Engineer',
            19: 'PMO',
            7: 'Database',
            13: 'Hadoop',
            10: 'ETL Developer',
            9: 'DotNet Developer',
            3: 'Blockchain',
            23: 'Testing',
        }

        category_name = category_mapping.get(prediction_id, "unknown")
        st.write("Predicted category:", category_name)

if __name__ == "__main__":
    main()