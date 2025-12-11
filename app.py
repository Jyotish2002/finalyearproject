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
# Extended tech streams
try:
    from Courses import genai_course, cloud_course, devops_course, cyber_course, data_eng_course, qa_course, product_course, blockchain_course, career_streams
except Exception:
    genai_course = cloud_course = devops_course = cyber_course = data_eng_course = qa_course = product_course = blockchain_course = []
    career_streams = {}
# Non-tech streams
try:
    from Courses import (marketing_course, sales_course, finance_course, hr_course, operations_course, 
                         supply_chain_course, project_mgmt_course, business_analysis_course, entrepreneurship_course,
                         admin_course, education_course, psychology_course, law_course, healthcare_course,
                         nursing_course, medical_coding_course, pharmacy_course, graphic_design_course,
                         journalism_course, content_writing_course, soft_skills_course)
except Exception:
    marketing_course = sales_course = finance_course = hr_course = operations_course = []
    supply_chain_course = project_mgmt_course = business_analysis_course = entrepreneurship_course = []
    admin_course = education_course = psychology_course = law_course = healthcare_course = []
    nursing_course = medical_coding_course = pharmacy_course = graphic_design_course = []
    journalism_course = content_writing_course = soft_skills_course = []

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
            ✨ Built with passion <br>
            <a href= style="text-decoration: none; color: #ffd700; font-weight: 700;"></a>
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
                # Tech Fields
                ds_keyword = ['tensorflow','keras','pytorch','machine learning','deep Learning','flask','streamlit','data science','pandas','numpy','scikit-learn','jupyter','matplotlib','seaborn','statistics','regression','classification','neural network','nlp','computer vision']
                web_keyword = ['react', 'django', 'node js', 'react js', 'php', 'laravel', 'magento', 'wordpress','javascript', 'angular js', 'c#', 'asp.net', 'flask', 'html', 'css', 'vue', 'bootstrap', 'tailwind', 'typescript', 'next.js', 'express', 'mongodb', 'mysql', 'frontend', 'backend', 'fullstack']
                android_keyword = ['android','android development','flutter','kotlin','xml','kivy','android studio','java android','mobile development','firebase android']
                ios_keyword = ['ios','ios development','swift','cocoa','cocoa touch','xcode','objective-c','swiftui','uikit','iphone development']
                uiux_keyword = ['ux','adobe xd','figma','zeplin','balsamiq','ui','prototyping','wireframes','storyframes','adobe photoshop','photoshop','editing','adobe illustrator','illustrator','adobe after effects','after effects','adobe premier pro','premier pro','adobe indesign','indesign','wireframe','solid','grasp','user research','user experience','sketch','invision','usability']
                
                # Extended Tech Fields
                genai_keyword = ['generative ai','llm','large language model','chatgpt','gpt','openai','langchain','prompt engineering','rag','retrieval augmented','transformers','hugging face','bert','claude','gemini','ai agents','vector database','embeddings','fine-tuning','llama']
                cloud_keyword = ['aws','azure','gcp','google cloud','amazon web services','cloud computing','ec2','s3','lambda','cloudformation','terraform','cloud architecture','iaas','paas','saas','docker','kubernetes','k8s','microservices','serverless']
                devops_keyword = ['devops','ci/cd','jenkins','github actions','gitlab ci','docker','kubernetes','ansible','puppet','chef','terraform','monitoring','prometheus','grafana','elk','linux','bash','shell scripting','infrastructure','deployment','containerization']
                cyber_keyword = ['cybersecurity','security','penetration testing','ethical hacking','vulnerability','firewall','siem','soc','network security','information security','cryptography','malware','incident response','compliance','gdpr','iso 27001','cissp','ceh','security audit','threat analysis']
                data_eng_keyword = ['data engineering','etl','data pipeline','airflow','spark','hadoop','kafka','data warehouse','snowflake','databricks','sql','big data','data lake','dbt','data modeling','redshift','bigquery','data integration','batch processing','stream processing']
                qa_keyword = ['qa','quality assurance','testing','test automation','selenium','cypress','jest','junit','testng','api testing','postman','jmeter','load testing','performance testing','regression testing','manual testing','bug tracking','jira','test cases','agile testing']
                product_keyword = ['product management','product manager','roadmap','agile','scrum','jira','user stories','sprint','backlog','stakeholder','mvp','product strategy','product analytics','a/b testing','feature prioritization','go-to-market','product lifecycle','user feedback','competitive analysis']
                blockchain_keyword = ['blockchain','solidity','ethereum','smart contracts','web3','cryptocurrency','defi','nft','consensus','distributed ledger','hyperledger','bitcoin','crypto','token','dapp','decentralized','metamask','truffle','hardhat']
                
                # Non-Tech Fields
                marketing_keyword = ['marketing','digital marketing','seo','sem','google ads','facebook ads','social media marketing','content marketing','email marketing','marketing automation','hubspot','mailchimp','brand management','campaign','analytics','google analytics','influencer marketing','ppc','cpc','conversion rate']
                sales_keyword = ['sales','salesforce','crm','lead generation','b2b','b2c','account management','cold calling','sales strategy','negotiation','closing','quota','pipeline','customer acquisition','upselling','cross-selling','sales funnel','business development','client relationship']
                finance_keyword = ['finance','accounting','financial analysis','excel','financial modeling','budgeting','forecasting','investment','banking','audit','taxation','sap','quickbooks','tally','gst','balance sheet','profit loss','cash flow','equity','valuation','cpa','cfa','risk management']
                hr_keyword = ['hr','human resources','recruitment','talent acquisition','onboarding','payroll','employee engagement','performance management','hris','workday','succession planning','compensation','benefits','training','development','labor law','hr policies','workforce planning','diversity','inclusion']
                operations_keyword = ['operations','operations management','process improvement','lean','six sigma','supply chain','logistics','inventory','procurement','vendor management','quality control','manufacturing','production','efficiency','kpi','sop','erp','sap','oracle','operational excellence']
                supply_chain_keyword = ['supply chain','logistics','inventory management','procurement','warehousing','distribution','demand planning','supply planning','freight','shipping','transportation','vendor management','sourcing','purchase order','material planning','mrp','erp supply chain','cost optimization']
                project_mgmt_keyword = ['project management','pmp','prince2','agile','scrum','kanban','ms project','asana','monday.com','trello','project planning','resource allocation','timeline','milestone','risk management','stakeholder management','budget management','scope management','gantt chart','project delivery']
                business_analysis_keyword = ['business analysis','business analyst','requirements gathering','brd','frd','use cases','process mapping','gap analysis','stakeholder analysis','data analysis','sql','tableau','power bi','business process','system analysis','functional requirements','uml','swimlane','as-is to-be']
                entrepreneurship_keyword = ['entrepreneurship','startup','business plan','venture capital','angel investor','pitch deck','bootstrapping','business model','market research','competitive analysis','growth hacking','scaling','funding','seed round','series a','incubator','accelerator','lean startup','mvp','customer validation']
                admin_keyword = ['administration','administrative','office management','scheduling','calendar management','data entry','filing','records management','correspondence','travel arrangement','meeting coordination','office supplies','reception','executive assistant','personal assistant','office administration','clerical','organizational skills','business administration','business operations','strategic planning','department coordination','workflow','smooth workflow','coordinating departments','daily operations','strategic goals','interpersonal skills','multitasking','time management','attention to detail','decision-making','problem-solving','ms office','microsoft office','word excel','excel powerpoint','communication skills','organizational','work independently','work in a team','team player','basic understanding of business','knowledge of ms office','good organizational']
                education_keyword = ['education','teaching','curriculum','lesson planning','classroom management','pedagogy','e-learning','lms','moodle','instructional design','student assessment','educational technology','tutoring','training','course development','academic','k-12','higher education','special education']
                psychology_keyword = ['psychology','counseling','therapy','mental health','behavioral analysis','cognitive psychology','clinical psychology','psychotherapy','assessment','diagnosis','intervention','case management','client assessment','psychological testing','trauma','anxiety','depression','cbt','dbt']
                law_keyword = ['law','legal','attorney','lawyer','litigation','contracts','compliance','corporate law','intellectual property','patent','trademark','legal research','case law','paralegal','legal writing','due diligence','regulatory','mergers acquisitions','dispute resolution','arbitration']
                healthcare_keyword = ['healthcare','clinical','patient care','medical records','emr','ehr','hipaa','healthcare management','hospital','clinic','diagnosis','treatment','medical terminology','icd-10','cpt','billing','healthcare it','telemedicine','public health','epidemiology']
                nursing_keyword = ['nursing','registered nurse','rn','patient care','medication administration','vital signs','clinical skills','nursing assessment','care plan','wound care','iv therapy','emergency nursing','icu','or','pediatric nursing','geriatric nursing','nurse practitioner','lpn','bsn']
                medical_coding_keyword = ['medical coding','icd-10','cpt','hcpcs','medical billing','claims processing','revenue cycle','coding certification','cpc','ccs','drg','healthcare billing','insurance claims','reimbursement','coding accuracy','medical terminology','anatomy','health information']
                pharmacy_keyword = ['pharmacy','pharmacist','pharmaceutical','drug dispensing','prescription','medication therapy','clinical pharmacy','compounding','pharmacy technician','drug interactions','formulary','pharmaceutical care','medication review','patient counseling','pharmacy management','inventory control','hospital pharmacy','retail pharmacy']
                graphic_design_keyword = ['graphic design','photoshop','illustrator','indesign','canva','coreldraw','visual design','typography','branding','logo design','print design','digital design','layout','composition','color theory','vector graphics','raster','creative design','visual communication','motion graphics']
                journalism_keyword = ['journalism','news writing','reporting','investigative journalism','press release','editorial','media','broadcasting','print media','digital media','news gathering','interviewing','fact-checking','ap style','beat reporting','feature writing','news production','multimedia journalism']
                content_writing_keyword = ['content writing','copywriting','blog writing','seo writing','technical writing','creative writing','article writing','web content','social media content','content strategy','content creation','storytelling','editing','proofreading','ghostwriting','content marketing','website copy','product description']
                
                n_any = ['english','communication','writing', 'microsoft office', 'leadership','customer management', 'social media']
                ### Skill Recommendations Starts                
                recommended_skills = []
                reco_field = ''
                rec_course = ''

                ### condition starts to check skills from keywords and predict field
                skills_to_check = resume_data.get('skills', [])
                
                # Also analyze resume text for better matching
                resume_text_for_matching = resume_text.lower() if resume_text else ""
                
                # Also analyze job description for matching (prioritize job description)
                job_desc_for_matching = job_description.lower() if job_description else ""
                
                # Create a dictionary to count matches for each category
                category_matches = {
                    'Data Science': {'keywords': ds_keyword, 'count': 0, 'course': ds_course},
                    'Web Development': {'keywords': web_keyword, 'count': 0, 'course': web_course},
                    'Android Development': {'keywords': android_keyword, 'count': 0, 'course': android_course},
                    'IOS Development': {'keywords': ios_keyword, 'count': 0, 'course': ios_course},
                    'UI-UX Development': {'keywords': uiux_keyword, 'count': 0, 'course': uiux_course},
                    'Generative AI / LLM': {'keywords': genai_keyword, 'count': 0, 'course': genai_course},
                    'Cloud Computing': {'keywords': cloud_keyword, 'count': 0, 'course': cloud_course},
                    'DevOps Engineering': {'keywords': devops_keyword, 'count': 0, 'course': devops_course},
                    'Cybersecurity': {'keywords': cyber_keyword, 'count': 0, 'course': cyber_course},
                    'Data Engineering': {'keywords': data_eng_keyword, 'count': 0, 'course': data_eng_course},
                    'QA / Software Testing': {'keywords': qa_keyword, 'count': 0, 'course': qa_course},
                    'Product Management': {'keywords': product_keyword, 'count': 0, 'course': product_course},
                    'Blockchain Development': {'keywords': blockchain_keyword, 'count': 0, 'course': blockchain_course},
                    'Marketing': {'keywords': marketing_keyword, 'count': 0, 'course': marketing_course},
                    'Sales': {'keywords': sales_keyword, 'count': 0, 'course': sales_course},
                    'Finance & Accounting': {'keywords': finance_keyword, 'count': 0, 'course': finance_course},
                    'Human Resources': {'keywords': hr_keyword, 'count': 0, 'course': hr_course},
                    'Operations Management': {'keywords': operations_keyword, 'count': 0, 'course': operations_course},
                    'Supply Chain Management': {'keywords': supply_chain_keyword, 'count': 0, 'course': supply_chain_course},
                    'Project Management': {'keywords': project_mgmt_keyword, 'count': 0, 'course': project_mgmt_course},
                    'Business Analysis': {'keywords': business_analysis_keyword, 'count': 0, 'course': business_analysis_course},
                    'Entrepreneurship': {'keywords': entrepreneurship_keyword, 'count': 0, 'course': entrepreneurship_course},
                    'Administration': {'keywords': admin_keyword, 'count': 0, 'course': admin_course},
                    'Education & Teaching': {'keywords': education_keyword, 'count': 0, 'course': education_course},
                    'Psychology & Counseling': {'keywords': psychology_keyword, 'count': 0, 'course': psychology_course},
                    'Law & Legal': {'keywords': law_keyword, 'count': 0, 'course': law_course},
                    'Healthcare Management': {'keywords': healthcare_keyword, 'count': 0, 'course': healthcare_course},
                    'Nursing': {'keywords': nursing_keyword, 'count': 0, 'course': nursing_course},
                    'Medical Coding & Billing': {'keywords': medical_coding_keyword, 'count': 0, 'course': medical_coding_course},
                    'Pharmacy': {'keywords': pharmacy_keyword, 'count': 0, 'course': pharmacy_course},
                    'Graphic Design': {'keywords': graphic_design_keyword, 'count': 0, 'course': graphic_design_course},
                    'Journalism & Media': {'keywords': journalism_keyword, 'count': 0, 'course': journalism_course},
                    'Content Writing': {'keywords': content_writing_keyword, 'count': 0, 'course': content_writing_course},
                }
                
                # Count keyword matches - PRIORITIZE JOB DESCRIPTION
                # If job description is provided, match ONLY against it for recommendations
                # Otherwise fall back to resume analysis
                import re
                
                def word_match(keyword, text):
                    """Check if keyword exists as a whole word/phrase in text"""
                    # Escape special regex characters in keyword
                    escaped_keyword = re.escape(keyword.lower())
                    # Use word boundaries for single words, or just check presence for phrases
                    if ' ' in keyword:
                        # Multi-word phrase - check if it exists in text
                        return keyword.lower() in text.lower()
                    else:
                        # Single word - use word boundary matching
                        pattern = r'\b' + escaped_keyword + r'\b'
                        return bool(re.search(pattern, text.lower()))
                
                for category, data in category_matches.items():
                    for keyword in data['keywords']:
                        keyword_lower = keyword.lower()
                        
                        # If job description is provided, ONLY match against it
                        if job_desc_for_matching:
                            # Match ONLY against job description when it's provided
                            if word_match(keyword, job_desc_for_matching):
                                data['count'] += 5
                        else:
                            # No job description - fall back to resume analysis
                            # Check in skills list
                            if skills_to_check:
                                for skill in skills_to_check:
                                    if keyword_lower in skill.lower() or skill.lower() in keyword_lower:
                                        data['count'] += 2
                            
                            # Check in resume text
                            if word_match(keyword, resume_text_for_matching):
                                data['count'] += 1
                
                # Find the category with maximum matches
                best_category = None
                max_count = 0
                for category, data in category_matches.items():
                    if data['count'] > max_count:
                        max_count = data['count']
                        best_category = category
                
                # Define recommended skills for each category
                category_recommended_skills = {
                    'Data Science': ['Data Visualization','Predictive Analysis','Statistical Modeling','Data Mining','Clustering & Classification','Data Analytics','Quantitative Analysis','Web Scraping','ML Algorithms','Keras','Pytorch','Probability','Scikit-learn','Tensorflow','Flask','Streamlit'],
                    'Web Development': ['React','Django','Node JS','React JS','php','laravel','Magento','wordpress','Javascript','Angular JS','c#','Flask','SDK'],
                    'Android Development': ['Android','Android development','Flutter','Kotlin','XML','Java','Kivy','GIT','SDK','SQLite'],
                    'IOS Development': ['IOS','IOS Development','Swift','Cocoa','Cocoa Touch','Xcode','Objective-C','SQLite','Plist','StoreKit','UI-Kit','AV Foundation','Auto-Layout'],
                    'UI-UX Development': ['UI','User Experience','Adobe XD','Figma','Zeplin','Balsamiq','Prototyping','Wireframes','Storyframes','Adobe Photoshop','Editing','Illustrator','After Effects','Premier Pro','Indesign','Wireframe','Solid','Grasp','User Research'],
                    'Generative AI / LLM': ['LLM','Prompt Engineering','LangChain','RAG','Vector Databases','Hugging Face','Transformers','Fine-tuning','OpenAI API','Claude','GPT','Embeddings','AI Agents','Python','PyTorch'],
                    'Cloud Computing': ['AWS','Azure','GCP','Docker','Kubernetes','Terraform','CloudFormation','Lambda','EC2','S3','IAM','VPC','Serverless','Microservices','CI/CD'],
                    'DevOps Engineering': ['CI/CD','Jenkins','GitHub Actions','Docker','Kubernetes','Ansible','Terraform','Linux','Shell Scripting','Prometheus','Grafana','ELK Stack','Git','Monitoring','Infrastructure as Code'],
                    'Cybersecurity': ['Penetration Testing','Network Security','SIEM','SOC','Vulnerability Assessment','Firewall','Incident Response','Cryptography','Ethical Hacking','CISSP','CEH','Compliance','Risk Assessment','Security Audit'],
                    'Data Engineering': ['ETL','Apache Spark','Airflow','Kafka','SQL','Data Warehouse','Snowflake','Databricks','Python','Big Data','Data Pipeline','Hadoop','Data Modeling','dbt','AWS/GCP/Azure Data Services'],
                    'QA / Software Testing': ['Test Automation','Selenium','Cypress','API Testing','Postman','JMeter','JIRA','Test Cases','Manual Testing','Performance Testing','Regression Testing','Agile Testing','Bug Tracking','Load Testing'],
                    'Product Management': ['Product Strategy','Agile/Scrum','JIRA','Roadmapping','User Stories','Stakeholder Management','Data Analytics','A/B Testing','MVP','Go-to-Market','Competitive Analysis','User Research','Prioritization'],
                    'Blockchain Development': ['Solidity','Ethereum','Smart Contracts','Web3.js','DeFi','NFT','Hyperledger','Consensus Mechanisms','Cryptography','dApps','Truffle','Hardhat','MetaMask','Token Standards'],
                    'Marketing': ['Digital Marketing','SEO','SEM','Google Ads','Social Media Marketing','Content Marketing','Email Marketing','Marketing Automation','Google Analytics','HubSpot','Brand Management','PPC','Influencer Marketing'],
                    'Sales': ['Salesforce','CRM','Lead Generation','B2B Sales','Negotiation','Account Management','Pipeline Management','Cold Calling','Sales Strategy','Client Relationship','Business Development','Closing Deals'],
                    'Finance & Accounting': ['Financial Analysis','Excel','Financial Modeling','Budgeting','Forecasting','Accounting','Taxation','Audit','SAP','Tally','QuickBooks','GST','Balance Sheet','Cash Flow','Valuation'],
                    'Human Resources': ['Recruitment','Talent Acquisition','HRIS','Workday','Payroll','Employee Engagement','Performance Management','Onboarding','Training & Development','Labor Law','Compensation & Benefits','Succession Planning'],
                    'Operations Management': ['Process Improvement','Lean','Six Sigma','Operations Planning','Quality Control','KPI Management','ERP','SAP','Vendor Management','Inventory Management','Production Planning','SOP Development'],
                    'Supply Chain Management': ['Logistics','Inventory Management','Procurement','Warehousing','Demand Planning','Supply Planning','Vendor Management','ERP','SAP SCM','Cost Optimization','Distribution','Transportation Management'],
                    'Project Management': ['PMP','PRINCE2','Agile','Scrum','JIRA','MS Project','Risk Management','Stakeholder Management','Budget Management','Resource Planning','Gantt Charts','Sprint Planning','Project Delivery'],
                    'Business Analysis': ['Requirements Gathering','BRD','FRD','Use Cases','Process Mapping','Gap Analysis','SQL','Tableau','Power BI','Stakeholder Analysis','UML','Data Analysis','Business Process Improvement'],
                    'Entrepreneurship': ['Business Planning','Pitch Deck','Market Research','Fundraising','Growth Hacking','Business Model Canvas','Lean Startup','MVP Development','Customer Validation','Venture Capital','Financial Projections'],
                    'Administration': ['Office Management','MS Office','Calendar Management','Data Entry','Filing','Records Management','Travel Arrangements','Meeting Coordination','Executive Support','Communication','Organizational Skills','Business Operations','Strategic Planning','Department Coordination'],
                    'Education & Teaching': ['Curriculum Development','Lesson Planning','Classroom Management','E-Learning','LMS','Instructional Design','Student Assessment','Pedagogy','Educational Technology','Course Development','Training'],
                    'Psychology & Counseling': ['Counseling','Psychotherapy','CBT','DBT','Assessment','Clinical Skills','Case Management','Mental Health','Behavioral Analysis','Client Assessment','Psychological Testing','Trauma-Informed Care'],
                    'Law & Legal': ['Legal Research','Contract Law','Litigation','Compliance','Corporate Law','Legal Writing','Due Diligence','Case Analysis','Paralegal Skills','Regulatory Compliance','Intellectual Property','Dispute Resolution'],
                    'Healthcare Management': ['Healthcare Administration','EMR/EHR','HIPAA','Medical Terminology','Healthcare IT','Hospital Management','Patient Care Coordination','ICD-10','Revenue Cycle','Quality Improvement','Telemedicine'],
                    'Nursing': ['Patient Care','Clinical Skills','Medication Administration','Vital Signs','Nursing Assessment','Care Planning','IV Therapy','Wound Care','Emergency Care','ICU','Patient Education','Electronic Health Records'],
                    'Medical Coding & Billing': ['ICD-10','CPT','HCPCS','Medical Billing','Claims Processing','Revenue Cycle','CPC Certification','Medical Terminology','Healthcare Billing','Insurance Claims','DRG','Coding Accuracy'],
                    'Pharmacy': ['Drug Dispensing','Medication Therapy','Clinical Pharmacy','Pharmaceutical Care','Patient Counseling','Drug Interactions','Formulary Management','Pharmacy Operations','Compounding','Inventory Management'],
                    'Graphic Design': ['Photoshop','Illustrator','InDesign','Canva','Typography','Branding','Logo Design','Visual Design','Color Theory','Print Design','Digital Design','Motion Graphics','Layout Design'],
                    'Journalism & Media': ['News Writing','Reporting','Investigative Journalism','Press Releases','Editorial','Broadcasting','Multimedia','Interviewing','Fact-Checking','Feature Writing','News Production','AP Style'],
                    'Content Writing': ['Content Writing','Copywriting','SEO Writing','Blog Writing','Technical Writing','Creative Writing','Content Strategy','Editing','Proofreading','Storytelling','Content Marketing','Web Content'],
                }
                
                # Display recommendation based on best match
                if best_category and max_count > 0:
                    reco_field = best_category
                    # Show different message based on whether job description was provided
                    if job_description:
                        st.success(f"** Based on the Job Description, recommended field: {best_category} **")
                        label_text = '### Skills Required for this Job'
                        help_text = 'Skills extracted from Job Description'
                        tip_message = '''<h5 style='text-align: left; color: #1ed760;'>These skills match the job requirements. Add them to your resume to boost🚀 your chances!</h5>'''
                    else:
                        st.success(f"** Based on your Resume, you seem suited for {best_category} roles **")
                        label_text = '### Recommended skills for you'
                        help_text = 'Recommended skills generated from System'
                        tip_message = '''<h5 style='text-align: left; color: #1ed760;'>Adding these skills to your resume will boost🚀 the chances of getting a Job💼</h5>'''
                    
                    recommended_skills = category_recommended_skills.get(best_category, [])
                    recommended_keywords = st_tags(label=label_text,
                        text=help_text,value=recommended_skills,key = 'reco_skills')
                    st.markdown(tip_message,unsafe_allow_html=True)
                    
                    # Get course for the category
                    course_data = category_matches[best_category]['course']
                    if course_data:
                        rec_course = course_recommender(course_data)
                    else:
                        rec_course = f"Check online platforms for {best_category} courses"
                else:
                    # No strong match found - show general skills
                    reco_field = 'General Skills'
                    st.info("** Your resume shows general professional skills. Consider specializing in a specific field for better career opportunities. **")
                    recommended_skills = ['Communication','Leadership','Problem Solving','Time Management','Teamwork','Critical Thinking','Adaptability','Organization','Project Management']
                    recommended_keywords = st_tags(label='### Recommended skills for you.',
                        text='Recommended skills generated from System',value=recommended_skills,key = 'general')
                    st.markdown('''<h5 style='text-align: left; color: #092851;'>Consider exploring specific career paths to stand out</h5>''',unsafe_allow_html=True)
                    rec_course = course_recommender(soft_skills_course) if soft_skills_course else "Explore courses in your area of interest"


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
                        
                        # Extract skills from job description using ALL pre-defined keyword lists
                        all_skill_keywords = (ds_keyword + web_keyword + android_keyword + ios_keyword + uiux_keyword + 
                                            genai_keyword + cloud_keyword + devops_keyword + cyber_keyword + data_eng_keyword +
                                            qa_keyword + product_keyword + blockchain_keyword + marketing_keyword + sales_keyword +
                                            finance_keyword + hr_keyword + operations_keyword + supply_chain_keyword + 
                                            project_mgmt_keyword + business_analysis_keyword + entrepreneurship_keyword +
                                            admin_keyword + education_keyword + psychology_keyword + law_keyword +
                                            healthcare_keyword + nursing_keyword + medical_coding_keyword + pharmacy_keyword +
                                            graphic_design_keyword + journalism_keyword + content_writing_keyword + n_any)
                        
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
                
                # Convert resume text to lowercase for case-insensitive matching
                resume_text_lower = resume_text.lower()
                
                # Check for experience first (needed for internship logic)
                has_experience = 'experience' in resume_text_lower or 'work history' in resume_text_lower or 'employment' in resume_text_lower or 'worked at' in resume_text_lower or 'working at' in resume_text_lower
                has_internship = 'internship' in resume_text_lower or 'intern ' in resume_text_lower or 'trainee' in resume_text_lower
                
                ### Essential Sections ###
                st.markdown('''<h6 style='text-align: left; color: #4facfe; margin-top: 1rem;'>📋 Essential Sections</h6>''',unsafe_allow_html=True)
                
                # 1. Contact Information
                if resume_data.get('email') or resume_data.get('mobile_number') or '@' in resume_text_lower or 'phone' in resume_text_lower or 'contact' in resume_text_lower:
                    resume_score = resume_score + 8
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Contact information detected</h5>''',unsafe_allow_html=True)                
                else:
                    st.markdown('''<h5 style='text-align: left; color: #e74c3c;'>[-] Add contact info (email, phone) - recruiters need to reach you!</h5>''',unsafe_allow_html=True)
                
                # 2. Objective/Summary
                if 'objective' in resume_text_lower or 'summary' in resume_text_lower or 'profile' in resume_text_lower or 'about me' in resume_text_lower:
                    resume_score = resume_score + 6
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Career objective/summary included</h5>''',unsafe_allow_html=True)                
                else:
                    st.markdown('''<h5 style='text-align: left; color: #e74c3c;'>[-] Add a career objective or professional summary - gives recruiters quick insight</h5>''',unsafe_allow_html=True)

                # 3. Education
                if 'education' in resume_text_lower or 'school' in resume_text_lower or 'college' in resume_text_lower or 'university' in resume_text_lower or 'degree' in resume_text_lower or 'bachelor' in resume_text_lower or 'master' in resume_text_lower or 'b.tech' in resume_text_lower or 'b.e' in resume_text_lower or 'm.tech' in resume_text_lower:
                    resume_score = resume_score + 12
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Education details included</h5>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left; color: #e74c3c;'>[-] Add education details - shows your qualification level</h5>''',unsafe_allow_html=True)

                # 4. Experience OR Internship (smart check)
                if has_experience:
                    resume_score = resume_score + 16
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Work experience included - great for showcasing your professional background!</h5>''',unsafe_allow_html=True)
                    # If experienced, internship is optional bonus
                    if has_internship:
                        resume_score = resume_score + 4
                        st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Internship experience also included - shows your journey!</h5>''',unsafe_allow_html=True)
                elif has_internship:
                    # Fresher with internship
                    resume_score = resume_score + 12
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Internship experience included - great start for your career!</h5>''',unsafe_allow_html=True)
                    st.markdown('''<h5 style='text-align: left; color: #f39c12;'>[~] Consider gaining more work experience as you progress</h5>''',unsafe_allow_html=True)
                else:
                    # Neither experience nor internship
                    st.markdown('''<h5 style='text-align: left; color: #e74c3c;'>[-] Add work experience or internships - crucial for recruiters</h5>''',unsafe_allow_html=True)

                # 5. Skills
                if 'skill' in resume_text_lower or 'proficient' in resume_text_lower or 'expertise' in resume_text_lower or 'competenc' in resume_text_lower or 'technical' in resume_text_lower or 'technologies' in resume_text_lower:
                    resume_score = resume_score + 8
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Skills section included</h5>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left; color: #e74c3c;'>[-] Add a skills section - helps ATS and recruiters find you</h5>''',unsafe_allow_html=True)

                ### Value-Adding Sections ###
                st.markdown('''<h6 style='text-align: left; color: #4facfe; margin-top: 1rem;'>⭐ Value-Adding Sections</h6>''',unsafe_allow_html=True)

                # 6. Projects
                if 'project' in resume_text_lower or 'developed' in resume_text_lower or 'built' in resume_text_lower or 'created' in resume_text_lower or 'implemented' in resume_text_lower:
                    resume_score = resume_score + 12
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Projects section included - shows hands-on experience!</h5>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left; color: #e74c3c;'>[-] Add projects - demonstrates practical skills and initiative</h5>''',unsafe_allow_html=True)

                # 7. Achievements/Awards
                if 'achievement' in resume_text_lower or 'award' in resume_text_lower or 'accomplish' in resume_text_lower or 'recognized' in resume_text_lower or 'honor' in resume_text_lower or 'won' in resume_text_lower or 'rank' in resume_text_lower:
                    resume_score = resume_score + 10
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Achievements/Awards included - sets you apart!</h5>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left; color: #f39c12;'>[~] Consider adding achievements or awards if you have any</h5>''',unsafe_allow_html=True)

                # 8. Certifications
                if 'certification' in resume_text_lower or 'certified' in resume_text_lower or 'certificate' in resume_text_lower or 'license' in resume_text_lower or 'course' in resume_text_lower:
                    resume_score = resume_score + 8
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Certifications included - shows continuous learning!</h5>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left; color: #f39c12;'>[~] Consider adding relevant certifications to boost credibility</h5>''',unsafe_allow_html=True)

                ### Professional Polish ###
                st.markdown('''<h6 style='text-align: left; color: #4facfe; margin-top: 1rem;'>✨ Professional Polish</h6>''',unsafe_allow_html=True)

                # 9. LinkedIn Profile
                if 'linkedin' in resume_text_lower or 'linkedin.com' in resume_text_lower:
                    resume_score = resume_score + 5
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] LinkedIn profile included - great for networking!</h5>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left; color: #f39c12;'>[~] Consider adding your LinkedIn profile URL</h5>''',unsafe_allow_html=True)

                # 10. GitHub/Portfolio (for tech roles)
                if 'github' in resume_text_lower or 'portfolio' in resume_text_lower or 'github.com' in resume_text_lower or 'behance' in resume_text_lower or 'dribbble' in resume_text_lower:
                    resume_score = resume_score + 5
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Portfolio/GitHub included - showcases your work!</h5>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left; color: #f39c12;'>[~] Consider adding GitHub or portfolio link to showcase your work</h5>''',unsafe_allow_html=True)

                # 11. Languages
                if 'language' in resume_text_lower or 'english' in resume_text_lower or 'hindi' in resume_text_lower or 'fluent' in resume_text_lower or 'proficient in' in resume_text_lower:
                    resume_score = resume_score + 4
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Language proficiency mentioned</h5>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left; color: #f39c12;'>[~] Consider mentioning language proficiencies</h5>''',unsafe_allow_html=True)

                # 12. References
                if 'reference' in resume_text_lower or 'available upon request' in resume_text_lower:
                    resume_score = resume_score + 3
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] References section included</h5>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left; color: #f39c12;'>[~] You may add "References available upon request"</h5>''',unsafe_allow_html=True)

                # 13. Hobbies/Interests (optional)
                if 'hobbies' in resume_text_lower or 'hobby' in resume_text_lower or 'interests' in resume_text_lower or 'extracurricular' in resume_text_lower or 'activities' in resume_text_lower:
                    resume_score = resume_score + 3
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Hobbies/Interests included - shows personality!</h5>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left; color: #f39c12;'>[~] Hobbies/Interests are optional but can show personality</h5>''',unsafe_allow_html=True)

                # 14. Volunteer/Social Work
                if 'volunteer' in resume_text_lower or 'social work' in resume_text_lower or 'community' in resume_text_lower or 'ngo' in resume_text_lower or 'nss' in resume_text_lower:
                    resume_score = resume_score + 4
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Volunteer/Social work included - shows character!</h5>''',unsafe_allow_html=True)

                # Cap the score at 100
                resume_score = min(resume_score, 100)
                
                st.markdown("---")
                
                # Display overall resume score
                st.markdown(f"""
                <div class="chart-container">
                    <h3 style="text-align: center; color: #333;">Resume Section Score</h3>
                    <p style="text-align: center; color: #666;">This score shows how well your resume covers essential sections (Contact, Education, Skills, etc.) - NOT job match!</p>
                    <p style="text-align: center; color: #f39c12; font-weight: bold;">📌 For Job Match Score, check the "Overall ATS Score" above (only available when you enter a Job Description)</p>
                </div>
                """, unsafe_allow_html=True)
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = resume_score,
                    title = {'text': "Section Coverage"},
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
        
        # Initialize session state for admin login
        if 'admin_logged_in' not in st.session_state:
            st.session_state.admin_logged_in = False
        
        # Show login form if not logged in
        if not st.session_state.admin_logged_in:
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
                    st.session_state.admin_logged_in = True
                    st.rerun()
                else:
                    st.error("Wrong ID & Password Provided")
        
        else:
            # Admin is logged in - show dashboard
            st.success('Welcome Admin!')
            
            # Logout button
            if st.button("🚪 Logout"):
                st.session_state.admin_logged_in = False
                st.rerun()
            
            st.markdown("---")
            
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
                
                # Delete User Data Section
                st.markdown("---")
                st.subheader("🗑️ Delete User Data")
                col1, col2 = st.columns(2)
                with col1:
                    if len(df) > 0:
                        user_ids = df['ID'].tolist()
                        selected_id = st.selectbox("Select User ID to Delete", user_ids, key="delete_user")
                        if st.button("Delete Selected User", key="del_user_btn"):
                            cursor.execute(f"DELETE FROM user_data WHERE ID = {selected_id}")
                            connection.commit()
                            st.success(f"User with ID {selected_id} deleted successfully!")
                            st.rerun()
                    else:
                        st.info("No user data to delete")
                with col2:
                    if st.button("🗑️ Delete ALL User Data", key="del_all_users"):
                        cursor.execute("DELETE FROM user_data")
                        connection.commit()
                        st.success("All user data deleted successfully!")
                        st.rerun()
                
                # Display Feedbacks
                st.markdown("---")
                cursor.execute('''SELECT * FROM user_feedback''')
                data = cursor.fetchall()
                st.header("**User's Feedback Data**")
                df_feedback = pd.DataFrame(data, columns=['ID', 'Name', 'Email', 'Score', 'Comments', 'Timestamp'])
                st.dataframe(df_feedback)
                st.markdown(get_csv_download_link(df_feedback, 'Feedback_Data.csv', 'Download Report'), unsafe_allow_html=True)
                
                # Delete Feedback Section
                st.markdown("---")
                st.subheader("🗑️ Delete Feedback Data")
                col3, col4 = st.columns(2)
                with col3:
                    if len(df_feedback) > 0:
                        feedback_ids = df_feedback['ID'].tolist()
                        selected_feedback_id = st.selectbox("Select Feedback ID to Delete", feedback_ids, key="delete_feedback")
                        if st.button("Delete Selected Feedback", key="del_feedback_btn"):
                            cursor.execute(f"DELETE FROM user_feedback WHERE ID = {selected_feedback_id}")
                            connection.commit()
                            st.success(f"Feedback with ID {selected_feedback_id} deleted successfully!")
                            st.rerun()
                    else:
                        st.info("No feedback data to delete")
                with col4:
                    if st.button("🗑️ Delete ALL Feedback Data", key="del_all_feedback"):
                        cursor.execute("DELETE FROM user_feedback")
                        connection.commit()
                        st.success("All feedback data deleted successfully!")
                        st.rerun()
                        
            except Exception as e:
                st.error(f"Failed to fetch data from the database: {e}")

# main function call
if __name__ == '__main__':
    run()