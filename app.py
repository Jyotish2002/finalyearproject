###### Packages Used ######
import streamlit as st # core package used in this project
import pandas as pd
import base64, random
import time,datetime
import pymysql
import os
import socket
import platform
import geocoder
import secrets
import io,random
import plotly.express as px # to create visualisations at the admin session
import plotly.graph_objects as go
from geopy.geocoders import Nominatim
# libraries used to parse the pdf files
from pyresparser import ResumeParser
from pdfminer3.layout import LAParams, LTTextBox
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager
from pdfminer3.pdfinterp import PDFPageInterpreter
from pdfminer3.converter import TextConverter
import re
from streamlit_tags import st_tags
from PIL import Image
# pre stored data for prediction purposes
from Courses import ds_course,web_course,android_course,ios_course,uiux_course,resume_videos,interview_videos
# Extended streams
try:
    from Courses import genai_course, cloud_course, devops_course, cyber_course, data_eng_course, qa_course, product_course, blockchain_course, career_streams
except Exception:
    genai_course = cloud_course = devops_course = cyber_course = data_eng_course = qa_course = product_course = blockchain_course = []
    career_streams = {}

import nltk
from nltk.corpus import stopwords
import re

# Download NLTK data at startup (for Streamlit Cloud)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    try:
        nltk.download('stopwords', quiet=True)
    except Exception as e:
        pass  # Will use fallback stopwords

DEFAULT_STOPWORDS = {
    'a','about','above','after','again','against','all','am','an','and','any','are',
    'as','at','be','because','been','before','being','below','between','both','but',
    'by','could','did','do','does','doing','down','during','each','few','for','from',
    'further','had','has','have','having','he','her','here','hers','herself','him',
    'himself','his','how','i','if','in','into','is','it','its','itself','just','me',
    'more','most','my','myself','no','nor','not','of','off','on','once','only','or',
    'other','our','ours','ourselves','out','over','own','same','she','should','so',
    'some','such','than','that','the','their','theirs','them','themselves','then','there',
    'these','they','this','those','through','to','too','under','until','up','very','was',
    'we','were','what','when','where','which','while','who','whom','why','with','you','your',
    'yours','yourself','yourselves'
}
_STOPWORDS_CACHE = None

def get_stopwords():
    """Get English stopwords with fallback to built-in list"""
    global _STOPWORDS_CACHE
    if _STOPWORDS_CACHE is not None:
        return _STOPWORDS_CACHE
    try:
        words = set(stopwords.words('english'))
        if words:
            _STOPWORDS_CACHE = words
            return words
    except:
        pass
    # Use fallback
    _STOPWORDS_CACHE = DEFAULT_STOPWORDS
    return DEFAULT_STOPWORDS


###### Preprocessing functions ######


# Generates a link allowing the data in a given panda dataframe to be downloaded in csv format 
def get_csv_download_link(df,filename,text):
    csv = df.to_csv(index=False)
    ## bytes conversions
    b64 = base64.b64encode(csv.encode()).decode()      
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href


# Reads Pdf file and check_extractable
def pdf_reader(file):
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    with open(file, 'rb') as fh:
        for page in PDFPage.get_pages(fh,
                                      caching=True,
                                      check_extractable=True):
            page_interpreter.process_page(page)
            print(page)
        text = fake_file_handle.getvalue()

    ## close open handles
    converter.close()
    fake_file_handle.close()
    return text


# Fallback resume parser when pyresparser fails
def fallback_resume_parser(resume_text, pdf_name):
    """
    Extract basic information from resume text when pyresparser fails
    """
    resume_data = {
        'name': '',
        'email': '',
        'mobile_number': '',
        'skills': [],
        'degree': '',
        'no_of_pages': 1
    }
    
    # Extract email
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, resume_text)
    if emails:
        resume_data['email'] = emails[0]
    
    # Extract phone number
    phone_pattern = r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    phones = re.findall(phone_pattern, resume_text)
    if phones:
        resume_data['mobile_number'] = ''.join(phones[0]) if isinstance(phones[0], tuple) else phones[0]
    
    # Extract name (usually in the first few lines)
    lines = resume_text.split('\n')
    for line in lines[:10]:  # Check first 10 lines
        line = line.strip()
        if len(line) > 2 and len(line) < 50 and not any(char.isdigit() for char in line):
            # Skip lines with email or phone
            if '@' not in line and not re.search(r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}', line):
                # Skip common headers
                if line.lower() not in ['resume', 'cv', 'curriculum vitae', 'profile', 'objective']:
                    resume_data['name'] = line
                    break
    
    # Extract skills (basic keyword matching)
    skill_keywords = [
        'python', 'java', 'javascript', 'html', 'css', 'react', 'angular', 'vue',
        'node.js', 'django', 'flask', 'sql', 'mysql', 'postgresql', 'mongodb',
        'machine learning', 'data science', 'tensorflow', 'keras', 'pandas',
        'numpy', 'scikit-learn', 'git', 'docker', 'aws', 'azure', 'kubernetes',
        'android', 'ios', 'swift', 'kotlin', 'flutter', 'php', 'laravel',
        'wordpress', 'photoshop', 'illustrator', 'figma', 'ui/ux', 'design'
    ]
    
    resume_text_lower = resume_text.lower()
    found_skills = []
    for skill in skill_keywords:
        if skill in resume_text_lower:
            found_skills.append(skill.title())
    
    resume_data['skills'] = found_skills[:10]  # Limit to 10 skills
    
    # Extract degree (basic pattern matching)
    degree_patterns = [
        r'bachelor.*?(?:computer science|engineering|technology|science)',
        r'master.*?(?:computer science|engineering|technology|science)',
        r'phd.*?(?:computer science|engineering|technology|science)',
        r'b\.?tech|m\.?tech|b\.?sc|m\.?sc|b\.?com|m\.?com|mba|phd'
    ]
    
    for pattern in degree_patterns:
        match = re.search(pattern, resume_text_lower)
        if match:
            resume_data['degree'] = match.group().title()
            break
    
    return resume_data


# show uploaded file path to view pdf_display
def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


# course recommendations which has data already loaded from Courses.py
def course_recommender(course_list):
    st.subheader("**Courses & Certificates Recommendations 👨‍🎓**")
    c = 0
    rec_course = []
    ## slider to choose from range 1-10
    no_of_reco = st.slider('Choose Number of Course Recommendations:', 1, 10, 5)
    random.shuffle(course_list)
    for c_name, c_link in course_list:
        c += 1
        st.markdown(f"({c}) [{c_name}]({c_link})")
        rec_course.append(c_name)
        if c == no_of_reco:
            break
    return rec_course


###### Database Stuffs ######


# sql connector
try:
    connection = pymysql.connect(host='localhost',user='root',password='pass123',db='cv')
    cursor = connection.cursor()
    DB_AVAILABLE = True
except Exception as e:
    # Database not available - continue without it
    DB_AVAILABLE = False
    connection = None
    cursor = None
    # Only show warning in development, not on Streamlit Cloud
    if os.getenv('STREAMLIT_SHARING_MODE') is None:
        pass  # Silent fail for cloud deployment


# inserting miscellaneous data, fetched results, prediction and recommendation into user_data table
def insert_data(sec_token,ip_add,host_name,dev_user,os_name_ver,latlong,city,state,country,act_name,act_mail,act_mob,name,email,res_score,timestamp,no_of_pages,reco_field,cand_level,skills,recommended_skills,courses,pdf_name):
    if not DB_AVAILABLE or cursor is None:
        # Skip persistence when DB is not available
        return True
    try:
        DB_table_name = 'user_data'
        insert_sql = "insert into " + DB_table_name + """
        values (0,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
        rec_values = (str(sec_token),str(ip_add),host_name,dev_user,os_name_ver,str(latlong),city,state,country,act_name,act_mail,act_mob,name,email,str(res_score),timestamp,str(no_of_pages),reco_field,cand_level,skills,recommended_skills,courses,pdf_name)
        cursor.execute(insert_sql, rec_values)
        connection.commit()
    except pymysql.Error as e:
        st.error(f"Error inserting data into database: {e}")
        return False
    return True


# inserting feedback data into user_feedback table
def insertf_data(feed_name,feed_email,feed_score,comments,Timestamp):
    if not DB_AVAILABLE or cursor is None:
        return True
    try:
        DBf_table_name = 'user_feedback'
        insertfeed_sql = "insert into " + DBf_table_name + """
        values (0,%s,%s,%s,%s,%s)"""
        rec_values = (feed_name, feed_email, feed_score, comments, Timestamp)
        cursor.execute(insertfeed_sql, rec_values)
        connection.commit()
    except pymysql.Error as e:
        st.error(f"Error inserting feedback data: {e}")
        return False
    return True


###### Setting Page Configuration (favicon, Logo, Title) ######

import os

# Get the current directory path
current_dir = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
   page_title="AI Resume Analyzer",
   page_icon=os.path.join(current_dir, 'Logo', 'recommend.png'),
   layout="wide",
   initial_sidebar_state="expanded"
)

# Custom CSS for attractive styling
def load_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Main page styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Poppins', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom header styling */
    .custom-header {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .custom-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .custom-header p {
        color: white;
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }
    
    .css-1d391kg .css-1v0mbdj {
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    /* Custom cards */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 5px solid #4facfe;
    }
    
    .success-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
        text-align: center;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        color: #333;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    /* File uploader styling */
    .stFileUploader > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        border: none;
        color: white;
        padding: 2rem;
        text-align: center;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    
    /* Input field styling */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 0.75rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #4facfe;
        box-shadow: 0 0 10px rgba(79, 172, 254, 0.3);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #ff6b35 0%, #f7931e 50%, #4facfe 100%);
        border-radius: 10px;
    }
    
    /* Metrics styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    
    /* Tags styling */
    .stTags {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Chart container */
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    /* Animated elements */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in {
        animation: fadeInUp 0.6s ease-out;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .custom-header h1 {
            font-size: 2rem;
        }
        .custom-header p {
            font-size: 1rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)


###### Main function run() ######


def run():
    
    # Load custom CSS
    load_css()
    
    # Create attractive header
    st.markdown("""
    <div class="custom-header fade-in">
        <h1>🤖 AI Resume Analyzer</h1>
        <p>Unlock Your Career Potential with Smart Resume Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    # Enhanced sidebar with icons
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem; margin-bottom: 2rem;">
        <h2 style="color: #4facfe; font-weight: 700; margin: 0;">🚀 Navigation</h2>
        <p style="color: #888; font-size: 0.9rem; margin: 0.5rem 0 0 0;">Choose your journey</p>
    </div>
    """, unsafe_allow_html=True)
    
    activities = ["👤 User", "💭 Feedback", "ℹ️ About", "👨‍💼 Admin"]
    choice = st.sidebar.selectbox("Choose Section", activities, help="Select the section you want to visit", label_visibility="collapsed")
    
    # Clean up choice for processing
    choice = choice.split(" ", 1)[1] if " " in choice else choice
    
    st.sidebar.markdown("---")
    
    # Enhanced creator credits
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin: 1rem 0;">
        <p style="color: white; font-weight: 600; margin: 0; font-size: 0.9rem;">
            ✨ Built with passion by<br>
            <a href="https://jyotishyadav.netlify.app/" style="text-decoration: none; color: #ffd700; font-weight: 700;">Jyotish Yadav</a>
        </p>
    </div>
    """, unsafe_allow_html=True)
    # Enhanced visitor counter
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); border-radius: 10px; margin: 1rem 0;">
        <p style="color: white; font-weight: 600; margin: 0; font-size: 0.9rem;">
            👥 Site Visitors<br>
            <img src="https://counter9.stat.ovh/private/freecounterstat.php?c=t2xghr8ak6lfqt3kgru233378jya38dy" title="Visitor Counter" alt="visitor count" width="60px" style="margin-top: 0.5rem; border-radius: 5px;" />
        </p>
    </div>
    """, unsafe_allow_html=True)

    ###### Creating Database and Table ######


    # Create the DB
    try:
        db_sql = """CREATE DATABASE IF NOT EXISTS CV;"""
        cursor.execute(db_sql)

        # Create table user_data and user_feedback
        DB_table_name = 'user_data'
        table_sql = "CREATE TABLE IF NOT EXISTS " + DB_table_name + """
                        (ID INT NOT NULL AUTO_INCREMENT,
                        sec_token varchar(20) NOT NULL,
                        ip_add varchar(50) NULL,
                        host_name varchar(50) NULL,
                        dev_user varchar(50) NULL,
                        os_name_ver varchar(50) NULL,
                        latlong varchar(50) NULL,
                        city varchar(50) NULL,
                        state varchar(50) NULL,
                        country varchar(50) NULL,
                        act_name varchar(50) NOT NULL,
                        act_mail varchar(50) NOT NULL,
                        act_mob varchar(20) NOT NULL,
                        Name varchar(500) NOT NULL,
                        Email_ID VARCHAR(500) NOT NULL,
                        resume_score VARCHAR(8) NOT NULL,
                        Timestamp VARCHAR(50) NOT NULL,
                        Page_no VARCHAR(5) NOT NULL,
                        Predicted_Field BLOB NOT NULL,
                        User_level BLOB NOT NULL,
                        Actual_skills BLOB NOT NULL,
                        Recommended_skills BLOB NOT NULL,
                        Recommended_courses BLOB NOT NULL,
                        pdf_name varchar(50) NOT NULL,
                        PRIMARY KEY (ID)
                        );
                    """
        cursor.execute(table_sql)

        DBf_table_name = 'user_feedback'
        tablef_sql = "CREATE TABLE IF NOT EXISTS " + DBf_table_name + """
                        (ID INT NOT NULL AUTO_INCREMENT,
                            feed_name varchar(50) NOT NULL,
                            feed_email VARCHAR(50) NOT NULL,
                            feed_score VARCHAR(5) NOT NULL,
                            comments VARCHAR(100) NULL,
                            Timestamp VARCHAR(50) NOT NULL,
                            PRIMARY KEY (ID)
                        );
                    """
        cursor.execute(tablef_sql)
    except pymysql.Error as e:
        st.error(f"Error creating database tables: {e}")
        st.stop()


    ###### CODE FOR CLIENT SIDE (USER) ######

    if choice == 'User':
        
        # Enhanced user information section
        st.markdown("""
        <div class="info-card">
            <h3 style="color: #333; margin-top: 0;">📝 Personal Information</h3>
            <p style="color: #666; margin-bottom: 1rem;">Please fill in your details to get started with the analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            act_name = st.text_input('👤 Full Name *', placeholder="Enter your full name")
            act_mail = st.text_input('📧 Email Address *', placeholder="your.email@example.com")
        
        with col2:
            act_mob = st.text_input('📱 Mobile Number *', placeholder="+1 (555) 123-4567")
            st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        sec_token = secrets.token_urlsafe(12)
        host_name = socket.gethostname()
        ip_add = socket.gethostbyname(host_name)
        dev_user = os.getlogin()
        os_name_ver = platform.system() + " " + platform.release()
        
        # Geocoding with error handling
        try:
            g = geocoder.ip('me')
            latlong = g.latlng
            geolocator = Nominatim(user_agent="http")
            location = geolocator.reverse(latlong, language='en')
            address = location.raw['address']
            cityy = address.get('city', 'Unknown')
            statee = address.get('state', 'Unknown')
            countryy = address.get('country', 'Unknown')
        except Exception as e:
            st.warning(f"Geocoding failed: {e}. Using default location values.")
            latlong = [0.0, 0.0]
            cityy = 'Unknown'
            statee = 'Unknown' 
            countryy = 'Unknown'
            
        city = cityy
        state = statee
        country = countryy


        # Enhanced file upload section
        st.markdown("""
        <div class="info-card">
            <h3 style="color: #333; margin-top: 0;">📄 Resume & Job Description</h3>
            <p style="color: #666; margin-bottom: 1rem;">Upload your resume and paste the job description for a detailed analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 3])

        with col1:
            ## Enhanced file upload
            pdf_file = st.file_uploader(
                "🚀 Choose your Resume (PDF only)", 
                type=["pdf"],
                help="Upload a PDF file containing your resume for analysis"
            )
        
        with col2:
            job_description = st.text_area("📝 Paste Job Description Here", height=200, placeholder="Paste the full job description to compare your resume against...")

        if pdf_file is not None:
            # Validate required fields
            if not act_name or not act_mail or not act_mob:
                st.error("Please fill in all required fields (Name, Mail, Mobile Number) before uploading your resume.")
                st.stop()
            with st.spinner('Hang On While We Cook Magic For You...'):
                time.sleep(4)
        
            ### saving the uploaded resume to folder
            upload_dir = os.path.join(current_dir, 'Uploaded_Resumes')
            
            # Create directory if it doesn't exist
            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir)
            
            save_image_path = os.path.join(upload_dir, pdf_file.name)
            pdf_name = pdf_file.name
            with open(save_image_path, "wb") as f:
                f.write(pdf_file.getbuffer())
            show_pdf(save_image_path)

            ### parsing and extracting whole resume 
            # Parse resume with fallback handling (silent for better UX)
            resume_data = None
            try:
                resume_data = ResumeParser(save_image_path).get_extracted_data()
                if not resume_data:
                    raise Exception("Primary parser returned empty data")
            except Exception:
                # Silently try fallback method without showing technical errors
                try:
                    resume_text = pdf_reader(save_image_path)
                    if resume_text.strip():
                        resume_data = fallback_resume_parser(resume_text, pdf_file.name)
                    else:
                        resume_data = None
                except Exception:
                    resume_data = None
            
            # Show user-friendly message based on result
            if not resume_data:
                st.error("Unable to parse your resume. Please ensure your PDF contains readable text and try again.")
                st.info("💡 **Tips for better results:**")
                st.info("• Use a PDF with selectable text (not scanned images)")
                st.info("• Ensure your resume has clear sections and formatting")
                st.info("• Try converting your resume to a new PDF if issues persist")
                
            if resume_data:
                
                ## Get the whole resume data into resume_text (silently handle errors)
                try:
                    resume_text = pdf_reader(save_image_path)
                except Exception:
                    # Silently handle PDF reading errors
                    resume_text = ""

                ## Enhanced resume analysis display
                st.markdown("---")
                
                # Safely get name
                name = resume_data.get('name', 'User')
                if not name or not name.strip():
                    name = 'User'
                
                st.markdown(f"""
                <div class="success-card fade-in">
                    <h2 style="margin: 0; font-size: 2rem;">🎉 Resume Analysis Complete!</h2>
                    <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">Hello {name}, welcome to your personalized analysis</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="info-card">
                    <h3 style="color: #333; margin-top: 0;">👤 Personal Information</h3>
                </div>
                """, unsafe_allow_html=True)
                try:
                    # Display information with fallbacks in an attractive layout
                    name_display = resume_data.get('name', 'Not detected')
                    email_display = resume_data.get('email', 'Not detected')
                    mobile_display = resume_data.get('mobile_number', 'Not detected')
                    degree_display = resume_data.get('degree', 'Not detected')
                    pages_display = resume_data.get('no_of_pages', 1)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4 style="margin: 0; color: white;">👤 Name</h4>
                            <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">{name_display}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4 style="margin: 0; color: white;">📧 Email</h4>
                            <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">{email_display}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4 style="margin: 0; color: white;">📱 Contact</h4>
                            <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">{mobile_display}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4 style="margin: 0; color: white;">🎓 Degree</h4>
                            <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">{degree_display}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="info-card" style="text-align: center;">
                        <h4 style="color: #333; margin: 0;">📄 Resume Pages: <span style="color: #4facfe; font-weight: 700;">{pages_display}</span></h4>
                    </div>
                    """, unsafe_allow_html=True)

                except Exception:
                    # Silently handle display errors and show fallback
                    st.info("📄 Some resume details could not be extracted, but analysis will continue.")
                ## Predicting Candidate Experience Level 

                ### Trying with different possibilities
                cand_level = ''
                no_of_pages = resume_data.get('no_of_pages', 1)
                if no_of_pages < 1:                
                    cand_level = "NA"
                    st.markdown("""
                    <div class="warning-card">
                        <h3 style="margin: 0; color: #333;">🌱 Career Level: Fresher</h3>
                        <p style="margin: 0.5rem 0 0 0;">You're just starting your professional journey!</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                #### if internship then intermediate level
                elif any(word in resume_text.upper() for word in ['INTERNSHIP', 'INTERNSHIPS']):
                    cand_level = "Intermediate"
                    st.markdown("""
                    <div class="success-card">
                        <h3 style="margin: 0; color: white;">🚀 Career Level: Intermediate</h3>
                        <p style="margin: 0.5rem 0 0 0;">Great! You have internship experience that sets you apart!</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                #### if Work Experience/Experience then Experience level
                elif any(exp in resume_text.upper() for exp in ['EXPERIENCE', 'WORK EXPERIENCE']):
                    cand_level = "Experienced"
                    st.markdown("""
                    <div class="info-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white;">
                        <h3 style="margin: 0; color: white;">💼 Career Level: Experienced</h3>
                        <p style="margin: 0.5rem 0 0 0;">Excellent! Your work experience makes you a valuable candidate!</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    cand_level = "Fresher"
                    st.markdown("""
                    <div class="warning-card">
                        <h3 style="margin: 0; color: #333;">🌱 Career Level: Fresher</h3>
                        <p style="margin: 0.5rem 0 0 0;">You're at the beginning of your career journey. Time to shine!</p>
                    </div>
                    """, unsafe_allow_html=True)


                ## Enhanced Skills Analyzing and Recommendation
                st.markdown("---")
                st.markdown("""
                <div class="info-card">
                    <h3 style="color: #333; margin-top: 0;">💡 Skills Analysis & Recommendations</h3>
                    <p style="color: #666; margin-bottom: 1rem;">Based on your resume, here are your current skills and our AI-powered recommendations</p>
                </div>
                """, unsafe_allow_html=True)
                
                ### Current Analyzed Skills
                current_skills = resume_data.get('skills', [])
                if not current_skills:
                    current_skills = ['No skills detected']
                
                keywords = st_tags(label='### Your Current Skills',
                text='See our skills recommendation below',value=current_skills,key = '1  ')

                ### Keywords for Recommendations
                ds_keyword = ['tensorflow','keras','pytorch','machine learning','deep Learning','flask','streamlit']
                web_keyword = ['react', 'django', 'node jS', 'react js', 'php', 'laravel', 'magento', 'wordpress','javascript', 'angular js', 'C#', 'Asp.net', 'flask']
                android_keyword = ['android','android development','flutter','kotlin','xml','kivy']
                ios_keyword = ['ios','ios development','swift','cocoa','cocoa touch','xcode']
                uiux_keyword = ['ux','adobe xd','figma','zeplin','balsamiq','ui','prototyping','wireframes','storyframes','adobe photoshop','photoshop','editing','adobe illustrator','illustrator','adobe after effects','after effects','adobe premier pro','premier pro','adobe indesign','indesign','wireframe','solid','grasp','user research','user experience']
                n_any = ['english','communication','writing', 'microsoft office', 'leadership','customer management', 'social media']
                ### Skill Recommendations Starts                
                recommended_skills = []
                reco_field = ''
                rec_course = ''

                ### condition starts to check skills from keywords and predict field
                skills_to_check = resume_data.get('skills', [])
                if not skills_to_check or skills_to_check == ['No skills detected']:
                    # If no skills detected, try to analyze from the resume text
                    skills_to_check = []
                    if 'resume_text' in locals():
                        # Extract skills from text analysis
                        text_lower = resume_text.lower()
                        for category_skills in [ds_keyword, web_keyword, android_keyword, ios_keyword, uiux_keyword]:
                            for skill in category_skills:
                                if skill.lower() in text_lower:
                                    skills_to_check.append(skill)
                        
                        if not skills_to_check:
                            skills_to_check = ['general']  # Default to general category
                
                for i in skills_to_check:
                
                    #### Data science recommendation
                    if i.lower() in ds_keyword:
                        print(i.lower())
                        reco_field = 'Data Science'
                        st.success("** Our analysis says you are looking for Data Science Jobs.**")
                        recommended_skills = ['Data Visualization','Predictive Analysis','Statistical Modeling','Data Mining','Clustering & Classification','Data Analytics','Quantitative Analysis','Web Scraping','ML Algorithms','Keras','Pytorch','Probability','Scikit-learn','Tensorflow',"Flask",'Streamlit']
                        recommended_keywords = st_tags(label='### Recommended skills for you.',
                        text='Recommended skills generated from System',value=recommended_skills,key = '2')
                        st.markdown('''<h5 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job</h5>''',unsafe_allow_html=True)
                        # course recommendation
                        rec_course = course_recommender(ds_course)
                        break

                    #### Web development recommendation
                    elif i.lower() in web_keyword:
                        print(i.lower())
                        reco_field = 'Web Development'
                        st.success("** Our analysis says you are looking for Web Development Jobs **")
                        recommended_skills = ['React','Django','Node JS','React JS','php','laravel','Magento','wordpress','Javascript','Angular JS','c#','Flask','SDK']
                        recommended_keywords = st_tags(label='### Recommended skills for you.',
                        text='Recommended skills generated from System',value=recommended_skills,key = '3')
                        st.markdown('''<h5 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h5>''',unsafe_allow_html=True)
                        # course recommendation
                        rec_course = course_recommender(web_course)
                        break

                    #### Android App Development
                    elif i.lower() in android_keyword:
                        print(i.lower())
                        reco_field = 'Android Development'
                        st.success("** Our analysis says you are looking for Android App Development Jobs **")
                        recommended_skills = ['Android','Android development','Flutter','Kotlin','XML','Java','Kivy','GIT','SDK','SQLite']
                        recommended_keywords = st_tags(label='### Recommended skills for you.',
                        text='Recommended skills generated from System',value=recommended_skills,key = '4')
                        st.markdown('''<h5 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h5>''',unsafe_allow_html=True)
                        # course recommendation
                        rec_course = course_recommender(android_course)
                        break

                    #### IOS App Development
                    elif i.lower() in ios_keyword:
                        print(i.lower())
                        reco_field = 'IOS Development'
                        st.success("** Our analysis says you are looking for IOS App Development Jobs **")
                        recommended_skills = ['IOS','IOS Development','Swift','Cocoa','Cocoa Touch','Xcode','Objective-C','SQLite','Plist','StoreKit',"UI-Kit",'AV Foundation','Auto-Layout']
                        recommended_keywords = st_tags(label='### Recommended skills for you.',
                        text='Recommended skills generated from System',value=recommended_skills,key = '5')
                        st.markdown('''<h5 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h5>''',unsafe_allow_html=True)
                        # course recommendation
                        rec_course = course_recommender(ios_course)
                        break

                    #### Ui-UX Recommendation
                    elif i.lower() in uiux_keyword:
                        print(i.lower())
                        reco_field = 'UI-UX Development'
                        st.success("** Our analysis says you are looking for UI-UX Development Jobs **")
                        recommended_skills = ['UI','User Experience','Adobe XD','Figma','Zeplin','Balsamiq','Prototyping','Wireframes','Storyframes','Adobe Photoshop','Editing','Illustrator','After Effects','Premier Pro','Indesign','Wireframe','Solid','Grasp','User Research']
                        recommended_keywords = st_tags(label='### Recommended skills for you.',
                        text='Recommended skills generated from System',value=recommended_skills,key = '6')
                        st.markdown('''<h5 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h5>''',unsafe_allow_html=True)
                        # course recommendation
                        rec_course = course_recommender(uiux_course)
                        break

                    #### For Not Any Recommendations
                    elif i.lower() in n_any:
                        print(i.lower())
                        reco_field = 'NA'
                        st.warning("** Currently our tool only predicts and recommends for Data Science, Web, Android, IOS and UI/UX Development**")
                        recommended_skills = ['No Recommendations']
                        recommended_keywords = st_tags(label='### Recommended skills for you.',
                        text='Currently No Recommendations',value=recommended_skills,key = '6')
                        st.markdown('''<h5 style='text-align: left; color: #092851;'>Maybe Available in Future Updates</h5>''',unsafe_allow_html=True)
                        # course recommendation
                        rec_course = "Sorry! Not Available for this Field"
                        break


                ## Resume Scorer & Resume Writing Tips
                st.markdown("---")
                st.subheader("**Resume Analysis & Scoring 🥂**")

                if job_description:
                    # Ensure stopwords are available
                    stop_words = get_stopwords()

                    def calculate_scores(resume_text, job_description_text, resume_data, stop_words):
                        scores = {
                            'skills_match': 0,
                            'keyword_alignment': 0,
                            'overall_score': 0,
                            'missing_skills': []
                        }

                        # 1. Skills Match Score
                        resume_skills = set([skill.lower() for skill in resume_data.get('skills', [])])
                        
                        # Extract skills from job description using the pre-defined keyword lists
                        all_skill_keywords = ds_keyword + web_keyword + android_keyword + ios_keyword + uiux_keyword + n_any
                        job_text_lower = job_description_text.lower()
                        job_skills = set()
                        for skill in all_skill_keywords:
                            if re.search(r'\b' + re.escape(skill.lower()) + r'\b', job_text_lower):
                                job_skills.add(skill.lower())

                        if not job_skills:
                            st.warning("Could not identify key technical skills in the job description. The skills score may be inaccurate.")
                        
                        matching_skills = resume_skills.intersection(job_skills)
                        scores['missing_skills'] = list(job_skills - resume_skills)
                        
                        skills_match_score = (len(matching_skills) / len(job_skills)) * 100 if job_skills else 0
                        scores['skills_match'] = min(int(skills_match_score), 100)

                        # 2. Keyword Alignment Score
                        resume_words = set(re.findall(r'\w+', resume_text.lower()))
                        job_words = set(re.findall(r'\w+', job_text_lower))
                        
                        # Remove stopwords from both sets
                        resume_words = resume_words - stop_words
                        job_words = job_words - stop_words
                        
                        common_words = resume_words.intersection(job_words)
                        
                        keyword_score = (len(common_words) / len(job_words)) * 100 if job_words else 0
                        scores['keyword_alignment'] = min(int(keyword_score), 100)

                        # 3. Overall Score (weighted average)
                        scores['overall_score'] = int((scores['skills_match'] * 0.6) + (scores['keyword_alignment'] * 0.4))
                        
                        return scores

                    analysis_scores = calculate_scores(resume_text, job_description, resume_data, stop_words)
                    
                    st.markdown("""
                    <div class="chart-container">
                        <h4 style="text-align: center; color: #333;">Resume to Job Description Match Score</h4>
                    </div>
                    """, unsafe_allow_html=True)

                    # Display scores with gauges
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = analysis_scores['skills_match'],
                            title = {'text': "Skills Match"},
                            gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': "#4facfe"}}))
                        fig.update_layout(height=250)
                        st.plotly_chart(fig, use_container_width=True)
                    with col2:
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = analysis_scores['keyword_alignment'],
                            title = {'text': "Keyword Alignment"},
                            gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': "#667eea"}}))
                        fig.update_layout(height=250)
                        st.plotly_chart(fig, use_container_width=True)
                    with col3:
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = analysis_scores['overall_score'],
                            title = {'text': "Overall ATS Score"},
                            gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': "#11998e"}}))
                        fig.update_layout(height=250)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Display missing skills
                    if analysis_scores['missing_skills']:
                        st.markdown("---")
                        st.subheader("🎯 **Skills to Target**")
                        st.markdown("Consider adding these skills found in the job description to your resume:")
                        st.info(', '.join(analysis_scores['missing_skills']))


                else:
                    st.markdown("""
                    <div class="warning-card">
                        <h4 style="margin: 0; color: #333;">💡 Pro Tip</h4>
                        <p style="margin: 0.5rem 0 0 0;">Paste a job description in the field above to get a detailed match score and keyword analysis!</p>
                    </div>
                    """, unsafe_allow_html=True)


                st.markdown("---")
                st.subheader("**Resume Content Checklist & Tips 🥂**")
                resume_score = 0
                
                ### Predicting Whether these key points are added to the resume
                if 'Objective' in resume_text or 'Summary' in resume_text:
                    resume_score = resume_score+6
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Objective/Summary</h4>''',unsafe_allow_html=True)                
                else:
                    st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] Please add your career objective, it will give your career intension to the Recruiters.</h4>''',unsafe_allow_html=True)

                if 'Education' in resume_text or 'School' in resume_text or 'College' in resume_text:
                    resume_score = resume_score + 12
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Education Details</h4>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] Please add Education. It will give Your Qualification level to the recruiter</h4>''',unsafe_allow_html=True)

                if 'EXPERIENCE' in resume_text or 'Experience' in resume_text:
                    resume_score = resume_score + 16
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Experience</h4>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] Please add Experience. It will help you to stand out from crowd</h4>''',unsafe_allow_html=True)

                if 'INTERNSHIPS' in resume_text or 'INTERNSHIP' in resume_text or 'Internships' in resume_text or 'Internship' in resume_text:
                    resume_score = resume_score + 6
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Internships</h4>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] Please add Internships. It will help you to stand out from crowd</h4>''',unsafe_allow_html=True)

                if 'SKILLS' in resume_text or 'SKILL' in resume_text or 'Skills' in resume_text or 'Skill' in resume_text:
                    resume_score = resume_score + 7
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Skills</h4>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] Please add Skills. It will help you a lot</h4>''',unsafe_allow_html=True)

                if 'HOBBIES' in resume_text or 'Hobbies' in resume_text:
                    resume_score = resume_score + 4
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Hobbies</h4>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] Please add Hobbies. It will show your personality to the Recruiters and give the assurance that you are fit for this role or not.</h4>''',unsafe_allow_html=True)

                if 'INTERESTS' in resume_text or 'Interests' in resume_text:
                    resume_score = resume_score + 5
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Interest</h4>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] Please add Interest. It will show your interest other that job.</h4>''',unsafe_allow_html=True)

                if 'ACHIEVEMENTS' in resume_text or 'Achievements' in resume_text:
                    resume_score = resume_score + 13
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Achievements </h4>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] Please add Achievements. It will show that you are capable for the required position.</h4>''',unsafe_allow_html=True)

                if 'CERTIFICATIONS' in resume_text or 'Certifications' in resume_text or 'Certification' in resume_text:
                    resume_score = resume_score + 12
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Certifications </h4>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] Please add Certifications. It will show that you have done some specialization for the required position.</h4>''',unsafe_allow_html=True)

                if 'PROJECTS' in resume_text or 'PROJECT' in resume_text or 'Projects' in resume_text:
                    resume_score = resume_score + 19
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Projects</h4>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] Please add Projects. It will show that you have done some work related to the required position.</h4>''',unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Display overall resume score
                st.markdown(f"""
                <div class="chart-container">
                    <h3 style="text-align: center; color: #333;">Resume Strength Score</h3>
                    <p style="text-align: center; color: #666;">This score reflects the completeness of your resume's sections.</p>
                </div>
                """, unsafe_allow_html=True)
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = resume_score,
                    title = {'text': "Completeness Score"},
                    gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': "#f5576c"}}))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
                
                # Insert data into database
                ts = time.time()
                act_timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                
                # Safely get data for insertion
                db_name = resume_data.get('name', 'N/A')
                db_email = resume_data.get('email', 'N/A')
                db_skills = ', '.join(resume_data.get('skills', []))
                db_pages = resume_data.get('no_of_pages', 'N/A')

                insert_data(sec_token,ip_add,host_name,dev_user,os_name_ver,latlong,city,state,country,act_name,act_mail,act_mob,db_name,db_email,str(resume_score),act_timestamp,str(db_pages),reco_field,cand_level,db_skills,', '.join(recommended_skills),', '.join(rec_course),pdf_name)
                
                ## Recommending Resume Writing Videos
                st.markdown("---")
                st.header("🎬 Recommended Videos for Resume Enhancement")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("**Resume Writing Tips**")
                    resume_vid = random.choice(resume_videos)
                    st.video(resume_vid)
                with col2:
                    st.subheader("**Interview Preparation**")
                    interview_vid = random.choice(interview_videos)
                    st.video(interview_vid)

                # Clean up uploaded file
                os.remove(save_image_path)
    
    ###### CODE FOR FEEDBACK SIDE ######
    elif choice == 'Feedback':
        
        # Enhanced feedback form
        st.markdown("""
        <div class="info-card">
            <h3 style="color: #333; margin-top: 0;">🗣️ We Value Your Feedback!</h3>
            <p style="color: #666; margin-bottom: 1rem;">Help us improve by sharing your experience.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            feed_name = st.text_input('👤 Your Name', placeholder="Enter your name")
            feed_email = st.text_input('📧 Your Email', placeholder="your.email@example.com")
        
        with col2:
            feed_score = st.slider('⭐ Rate our Application (1-5)', 1, 5, 3)
            st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        
        comments = st.text_area('💬 Your Comments or Suggestions', placeholder="Tell us what you think...")
        
        if st.button("Submit Feedback"):
            if not feed_name or not feed_email:
                st.error("Please provide your name and email.")
            else:
                ts = time.time()
                feed_timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                if insertf_data(feed_name, feed_email, str(feed_score), comments, feed_timestamp):
                    st.success("Thank you for your valuable feedback! We appreciate you helping us improve.")
                else:
                    st.error("Sorry, there was an issue submitting your feedback. Please try again later.")

    ###### CODE FOR ABOUT PAGE ######
    elif choice == 'About':
        
        # Enhanced About section
        st.markdown("""
        <div class="info-card fade-in">
            <h2 style="color: #333; margin-top: 0;">🌟 About AI Resume Analyzer</h2>
            <p style="color: #666; line-height: 1.6;">
                This application is a powerful tool designed to help job seekers optimize their resumes and increase their chances of landing their dream job. By leveraging the power of AI and Natural Language Processing (NLP), our analyzer provides intelligent feedback, skill recommendations, and insights that are crucial in today's competitive job market.
            </p>
            <h4 style="color: #4facfe; margin-top: 2rem;">🚀 Key Features:</h4>
            <ul style="color: #666; list-style-type: '✅ '; padding-left: 1.5rem;">
                <li><b>Intelligent Resume Parsing:</b> Extracts key information like contact details, skills, and education from your PDF resume.</li>
                <li><b>AI-Powered Skill Recommendations:</b> Suggests relevant skills to add to your resume based on your predicted career field.</li>
                <li><b>Resume Scoring:</b> Provides a score based on the completeness of your resume's sections.</li>
                <li><b>Course Recommendations:</b> Recommends online courses to help you bridge skill gaps.</li>
                <li><b>Video Resources:</b> Offers curated videos on resume writing and interview preparation.</li>
            </ul>
            <h4 style="color: #4facfe; margin-top: 2rem;">🛠️ Technologies Used:</h4>
            <p style="color: #666;">
                <b>Frontend:</b> Streamlit<br>
                <b>Backend & NLP:</b> Python, Pyresparser, NLTK, Spacy, PDFMiner<br>
                <b>Database:</b> MySQL<br>
                <b>Visualization:</b> Plotly
            </p>
        </div>
        """, unsafe_allow_html=True)

    ###### CODE FOR ADMIN SIDE (ADMIN) ######
    elif choice == 'Admin':
        
        # Enhanced admin login
        st.markdown("""
        <div class="info-card">
            <h3 style="color: #333; margin-top: 0;">🔐 Admin Login</h3>
            <p style="color: #666; margin-bottom: 1rem;">This section is for authorized personnel only.</p>
        </div>
        """, unsafe_allow_html=True)
        
        ad_user = st.text_input("👤 Username", type="default")
        ad_password = st.text_input("🔑 Password", type="password")
        
        if st.button("Login"):
            if ad_user == 'admin' and ad_password == 'admin123':
                st.success(f'Welcome {ad_user}')
                
                # Display Data
                try:
                    cursor.execute('''SELECT * FROM user_data''')
                    data = cursor.fetchall()
                    st.header("**User's Data**")
                    df = pd.DataFrame(data, columns=['ID', 'Token', 'IP Address', 'Host Name', 'Device User',
                                                    'OS', 'Lat/Long', 'City', 'State', 'Country', 'User Name',
                                                    'User Mail', 'User Mobile', 'Resume Name', 'Resume Mail',
                                                    'Resume Score', 'Timestamp', 'Total Pages', 'Predicted Field',
                                                    'User Level', 'Skills', 'Recommended Skills', 'Recommended Courses',
                                                    'PDF Name'])
                    st.dataframe(df)
                    st.markdown(get_csv_download_link(df, 'User_Data.csv', 'Download Report'), unsafe_allow_html=True)
                    
                    # Display Feedbacks
                    cursor.execute('''SELECT * FROM user_feedback''')
                    data = cursor.fetchall()
                    st.header("**User's Feedback Data**")
                    df = pd.DataFrame(data, columns=['ID', 'Name', 'Email', 'Score', 'Comments', 'Timestamp'])
                    st.dataframe(df)
                    st.markdown(get_csv_download_link(df, 'Feedback_Data.csv', 'Download Report'), unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Failed to fetch data from the database: {e}")

            else:
                st.error("Wrong ID & Password Provided")

# main function call
if __name__ == '__main__':
    run()