# AI Resume Analyzer 🚀

An advanced AI-powered resume analyzer that helps job seekers optimize their resumes for better job opportunities.

## Features ✨

### Core Analysis
- **Resume Parsing**: Automatically extracts key information (contact details, skills, education)
- **Job Description Matching**: Compares resume against job descriptions
- **ATS Friendliness Check**: Identifies formatting issues that may cause ATS parsing problems
- **Skills Gap Analysis**: Shows missing skills from job descriptions

### Advanced Metrics
- **Skills Match Score**: Measures alignment between resume and job requirements
- **Keyword Alignment**: Analyzes keyword overlap with job descriptions
- **Experience Relevance**: Evaluates experience level and role alignment
- **Quantification Index**: Checks for quantified achievements
- **Readability Score**: Ensures resume is easy to scan
- **Bias Audit**: Flags potentially problematic terminology

### Smart Features
- **Interview Question Generator**: Creates tailored interview prep questions
- **Course Recommendations**: Suggests relevant courses across 30+ career streams
- **Resume Scoring**: Provides completeness score based on key sections
- **Video Resources**: Curated resume and interview preparation videos

## Career Streams Covered 🎯

### Tech
- Data Science / ML
- Web Development
- Mobile (Android/iOS)
- UI/UX Design
- Cloud & DevOps
- Cybersecurity
- Data Engineering
- QA / Test Automation
- Product Management
- Blockchain / Web3
- Generative AI / LLMs

### Non-Tech
- Marketing & Sales
- Finance & Accounting
- HR & Operations
- Supply Chain
- Project Management
- Business Analysis
- Entrepreneurship
- Healthcare & Nursing
- Education & Psychology
- Law & Compliance
- Creative (Design, Writing, Journalism)

## Deployment on Streamlit Cloud 🌐

### Prerequisites
1. GitHub account
2. Streamlit Cloud account (free at [streamlit.io/cloud](https://streamlit.io/cloud))

### Steps

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/Jyotish2002/finalyearproject.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io/)
   - Click "New app"
   - Select your repository and branch (main)
   - Set main file path: `app.py`
   - Click "Deploy"

3. **Environment Variables (Optional)**
   - If you want to use MySQL database, add secrets in Streamlit Cloud:
     - Go to app settings → Secrets
     - Add your database credentials

### Configuration Files Included
- `requirements.txt`: Python dependencies
- `packages.txt`: System-level dependencies
- `.streamlit/config.toml`: Streamlit configuration
- `.gitignore`: Excludes sensitive files

## Local Development 💻

### Installation

```bash
# Clone the repository
git clone https://github.com/Jyotish2002/finalyearproject.git
cd finalyear

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords')"
```

### Run Locally

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## Database Setup (Optional) 🗄️

The app works without a database, but for tracking analytics:

```sql
CREATE DATABASE cv;
USE cv;

CREATE TABLE user_data (
    ID INT AUTO_INCREMENT PRIMARY KEY,
    sec_token VARCHAR(255),
    ip_add VARCHAR(50),
    host_name VARCHAR(255),
    dev_user VARCHAR(255),
    os_name_ver VARCHAR(255),
    latlong VARCHAR(100),
    city VARCHAR(100),
    state VARCHAR(100),
    country VARCHAR(100),
    act_name VARCHAR(255),
    act_mail VARCHAR(255),
    act_mob VARCHAR(50),
    name VARCHAR(255),
    email VARCHAR(255),
    res_score VARCHAR(10),
    timestamp DATETIME,
    no_of_pages VARCHAR(10),
    reco_field VARCHAR(100),
    cand_level VARCHAR(50),
    skills TEXT,
    recommended_skills TEXT,
    courses TEXT,
    pdf_name VARCHAR(255)
);

CREATE TABLE user_feedback (
    ID INT AUTO_INCREMENT PRIMARY KEY,
    feed_name VARCHAR(255),
    feed_email VARCHAR(255),
    feed_score VARCHAR(10),
    comments TEXT,
    Timestamp DATETIME
);
```

## Technologies Used 🛠️

- **Frontend**: Streamlit
- **Backend**: Python
- **NLP**: NLTK, Spacy, Pyresparser
- **PDF Processing**: PDFMiner
- **Database**: MySQL (optional)
- **Visualization**: Plotly
- **Deployment**: Streamlit Cloud

## Privacy & Security 🔒

- No error details exposed in production
- Database credentials not hardcoded
- Uploaded resumes can be configured to auto-delete
- XSRF protection enabled
- CORS properly configured

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.

## License 📄

This project is licensed under the MIT License.

## Support 💬

For issues or questions:
- Open an issue on GitHub
- Contact via the feedback form in the app

---

Made with ❤️ for job seekers worldwide by Jyotish Yadav
