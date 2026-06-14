import os
import re
import spacy
from spacy.matcher import Matcher

class ResumeParser(object):
    def __init__(self, resume_path, skills_file=None, custom_regex=None):
        self.__resume_path = resume_path
        self.__skills_file = skills_file
        self.__custom_regex = custom_regex
        self.__details = {
            'name': None,
            'email': None,
            'mobile_number': None,
            'skills': [],
            'degree': None,
            'no_of_pages': 1,
        }
        self.__parse()

    def get_extracted_data(self):
        return self.__details

    def __parse(self):
        # 1. Read PDF text using pdfminer
        text = ""
        try:
            from pdfminer.high_level import extract_text
            text = extract_text(self.__resume_path)
        except Exception:
            # Fallback to simple read if pdfminer high_level fails
            pass

        if not text or not text.strip():
            # If no text could be extracted, return empty details
            return

        # 2. Get number of pages
        try:
            from pdfminer.pdfpage import PDFPage
            with open(self.__resume_path, 'rb') as f:
                self.__details['no_of_pages'] = len(list(PDFPage.get_pages(f)))
        except Exception:
            self.__details['no_of_pages'] = 1

        # Save clean text
        cleaned_text = ' '.join(text.split())

        # 3. Extract Email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, cleaned_text)
        if emails:
            self.__details['email'] = emails[0].split()[0].strip(';')

        # 4. Extract Mobile Number
        if self.__custom_regex:
            phones = re.findall(re.compile(self.__custom_regex), cleaned_text)
        else:
            phone_pattern = r'(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
            phones = re.findall(phone_pattern, cleaned_text)
        if phones:
            self.__details['mobile_number'] = phones[0]

        # Load Spacy Model
        try:
            nlp = spacy.load('en_core_web_sm')
        except Exception:
            nlp = None

        # 5. Extract Name
        # Try to find person entities using spaCy
        name = None
        if nlp:
            try:
                doc = nlp(cleaned_text[:1000]) # Name is usually at the top
                for ent in doc.ents:
                    if ent.label_ == 'PERSON':
                        potential_name = ent.text.strip()
                        # A name should typically be between 2 and 4 words, not contain numbers or emails
                        if 2 <= len(potential_name.split()) <= 4 and not any(char.isdigit() for char in potential_name) and '@' not in potential_name:
                            # Exclude common headers
                            if potential_name.lower() not in ['resume', 'cv', 'curriculum vitae', 'profile', 'summary']:
                                name = potential_name
                                break
            except Exception:
                pass
        
        # Fallback Name Extraction: check the first few lines of the text
        if not name:
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            for line in lines[:15]:  # scan first 15 non-empty lines
                if len(line) > 2 and len(line) < 40 and not any(char.isdigit() for char in line):
                    if '@' not in line and not re.search(r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}', line):
                        if line.lower() not in ['resume', 'cv', 'curriculum vitae', 'profile', 'objective', 'summary', 'contact', 'education', 'skills', 'experience']:
                            name = line
                            break
        
        self.__details['name'] = name if name else 'User'

        # 6. Extract Degree
        degree_patterns = [
            r'\bB\.?Tech\b', r'\bM\.?Tech\b', r'\bB\.?E\.?\b', r'\bM\.?E\.?\b',
            r'\bB\.?Sc\b', r'\bM\.?Sc\b', r'\bB\.?Com\b', r'\bM\.?Com\b',
            r'\bB\.?C\.?A\b', r'\bM\.?C\.?A\b', r'\bM\.?B\.?A\b', r'\bPh\.?D\b',
            r'\bbachelor\s+of\s+\w+', r'\bmaster\s+of\s+\w+',
            r'\bdiploma\b', r'\bdegree\b'
        ]
        degree = None
        text_lower = text.lower()
        for pattern in degree_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                degree = match.group().title()
                cleaned_degree = degree.replace('.', '').upper()
                if cleaned_degree in ['BTECH', 'MTECH', 'BE', 'ME', 'BSC', 'MSC', 'BCOM', 'MCOM', 'BCA', 'MCA', 'MBA', 'PHD']:
                    degree = cleaned_degree
                break
        self.__details['degree'] = degree if degree else 'Not detected'

        # 7. Extract Skills
        skills_set = {
            # Programming languages
            'python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'swift', 'kotlin', 'golang', 'rust', 'typescript', 'r', 'sql', 'html', 'css',
            # Databases
            'mysql', 'postgresql', 'mongodb', 'sqlite', 'oracle', 'redis', 'cassandra', 'dynamodb',
            # Web frameworks / libraries
            'react', 'angular', 'vue', 'django', 'flask', 'fastapi', 'spring boot', 'node.js', 'express', 'laravel', 'asp.net', 'jquery', 'bootstrap', 'tailwind', 'next.js',
            # Mobile
            'android', 'ios', 'flutter', 'react native',
            # Cloud / DevOps
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'ansible', 'terraform', 'git', 'github', 'gitlab', 'ci/cd', 'linux', 'bash',
            # Data Science / ML / AI
            'machine learning', 'data science', 'deep learning', 'nlp', 'computer vision', 'tensorflow', 'pytorch', 'keras', 'pandas', 'numpy', 'scikit-learn', 'scipy', 'matplotlib', 'seaborn', 'tableau', 'power bi', 'excel', 'generative ai', 'llm', 'langchain', 'prompt engineering', 'openai',
            # Business / Non-tech
            'marketing', 'sales', 'finance', 'accounting', 'hr', 'human resources', 'recruitment', 'agile', 'scrum', 'project management', 'product management', 'business analysis', 'operations', 'supply chain',
            # Design
            'ui/ux', 'figma', 'adobe xd', 'sketch', 'photoshop', 'illustrator', 'indesign',
            # Testing / QA
            'testing', 'qa', 'selenium', 'cypress', 'jest', 'junit', 'postman',
            # Blockchain
            'blockchain', 'solidity', 'ethereum', 'web3',
            # Soft Skills
            'communication', 'leadership', 'teamwork', 'problem solving', 'time management'
        }

        found_skills = set()
        cleaned_text_lower = cleaned_text.lower()
        
        for skill in skills_set:
            if ' ' in skill or '/' in skill or '.' in skill:
                if skill in cleaned_text_lower:
                    found_skills.add(skill.title())
            else:
                pattern = r'\b' + re.escape(skill) + r'\b'
                if re.search(pattern, cleaned_text_lower):
                    if skill in ['aws', 'gcp', 'sql', 'html', 'css', 'qa', 'hr', 'llm', 'nlp', 'api', 'cv']:
                        found_skills.add(skill.upper())
                    elif skill in ['ui/ux', 'node.js', 'ci/cd', 'next.js']:
                        found_skills.add(skill)
                    else:
                        found_skills.add(skill.title())

        self.__details['skills'] = list(found_skills)