"""
Mentorship CV Processor - Individual Item Embeddings (IMPROVED)
===============================================================
Generates individual embeddings for each skill, project, club, education, and experience.
Fixed regex patterns for better extraction.
"""

import PyPDF2
import re
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from sentence_transformers import SentenceTransformer
from pathlib import Path
from datetime import datetime

# Use SAME model as reference code
model = SentenceTransformer("all-MiniLM-L6-v2")


class CVProcessor:
    """CV processor that generates individual embeddings for each item."""
    
    def __init__(self):
        """Initialize with the reference code's model."""
        self.embedding_model = model
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract raw text from PDF."""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")
        return text
    
    def extract_email(self, text: str) -> List[str]:
        """Extract email addresses."""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        return list(set(emails))  # Remove duplicates
    
    def extract_phone(self, text: str) -> List[str]:
        """Extract phone numbers - IMPROVED to avoid year ranges."""
        # More specific phone pattern that requires phone-like formatting
        phone_patterns = [
            r'\+\d{1,3}\s*\(\d{3}\)\s*\d{3}[-\s]?\d{4}',  # +1 (555) 123-4567
            r'\+\d{1,3}[-\s]?\d{3}[-\s]?\d{3}[-\s]?\d{4}',  # +1-555-123-4567
            r'\(\d{3}\)\s*\d{3}[-\s]?\d{4}',  # (555) 123-4567
            r'\d{3}[-\s]\d{3}[-\s]\d{4}',  # 555-123-4567
        ]
        
        phones = []
        for pattern in phone_patterns:
            matches = re.findall(pattern, text)
            phones.extend(matches)
        
        # Filter out year ranges (e.g., "2016 - 2020")
        phones = [p for p in phones if not re.match(r'^\d{4}\s*-\s*\d{4}$', p)]
        
        return list(set(phones))  # Remove duplicates
    
    def extract_urls(self, text: str) -> List[str]:
        """Extract URLs."""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, text)
        
        # Also look for common domains without http
        domain_pattern = r'(?:linkedin\.com|github\.com|twitter\.com|facebook\.com)[/\w\-.]+'
        domains = re.findall(domain_pattern, text, re.IGNORECASE)
        urls.extend(['https://' + d for d in domains])
        
        return list(set(urls))
    
    def extract_skills(self, text: str) -> List[str]:
        """Extract skills from CV - IMPROVED."""
        skills = []
        text_lower = text.lower()
        
        # Look for skills sections
        skill_headers = [
            'technical skills', 'skills', 'core competencies', 
            'expertise', 'technologies', 'programming languages',
            'tools & technologies'
        ]
        
        for header in skill_headers:
            # Match section and get content until next major section
            pattern = rf'{header}\s*[:\-]?\s*(.*?)(?=\n(?:professional experience|experience|education|projects|certifications|awards|leadership|$))'
            matches = re.findall(pattern, text_lower, re.IGNORECASE | re.DOTALL)
            
            if matches:
                skill_text = matches[0]
                # Split by common delimiters
                extracted = re.split(r'[,;•\n|]', skill_text)
                
                for skill in extracted:
                    skill = skill.strip()
                    # Clean up skill text
                    skill = re.sub(r'^\s*[-•*]\s*', '', skill)  # Remove bullets
                    skill = re.sub(r'^\s*\d+\.\s*', '', skill)  # Remove numbers
                    
                    # Only add if it's meaningful
                    if skill and len(skill) > 2 and len(skill) < 100:
                        # Remove common words that aren't skills
                        if not skill.startswith(('the ', 'a ', 'an ', 'and ', 'or ')):
                            skills.append(skill)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_skills = []
        for skill in skills:
            if skill not in seen:
                seen.add(skill)
                unique_skills.append(skill)
        
        return unique_skills[:30]  # Limit to 30 skills
    
    def extract_projects(self, text: str) -> List[Dict[str, str]]:
        """Extract projects - IMPROVED."""
        projects = []
        
        project_headers = [
            'key projects', 'projects', 'personal projects', 
            'academic projects', 'notable projects'
        ]
        
        for header in project_headers:
            # Match section
            pattern = rf'{header}\s*[:\-]?\s*\n(.*?)(?=\n(?:professional experience|experience|education|technical skills|skills|leadership|certifications|$))'
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            
            if matches:
                project_section = matches[0]
                
                # Split by bullet points or numbered items
                project_items = re.split(r'\n(?=[•\-*]|\d+\.)', project_section)
                
                for item in project_items:
                    item = item.strip()
                    if len(item) > 30:  # Meaningful project entry
                        lines = item.split('\n')
                        
                        # First line is usually the project name
                        name = lines[0].strip('•\-* \t0123456789.')
                        name = re.sub(r'^\s*\d+\.\s*', '', name)  # Remove numbering
                        
                        # Rest is description
                        description = ' '.join(lines[1:]).strip() if len(lines) > 1 else ''
                        
                        # If first line is too long, it might contain description too
                        if len(name) > 100 and ':' not in name:
                            parts = name.split('.', 1)
                            if len(parts) == 2:
                                name = parts[0].strip()
                                description = parts[1].strip() + ' ' + description
                        
                        projects.append({
                            'name': name[:150],
                            'description': description[:600]
                        })
        
        # Remove duplicates based on name
        seen_names = set()
        unique_projects = []
        for proj in projects:
            if proj['name'] not in seen_names:
                seen_names.add(proj['name'])
                unique_projects.append(proj)
        
        return unique_projects[:15]
    
    def extract_clubs_activities(self, text: str) -> List[str]:
        """Extract clubs and activities - IMPROVED."""
        clubs = []
        
        club_headers = [
            'leadership', 'leadership & extracurricular',
            'extracurricular activities', 'extracurricular',
            'clubs', 'organizations', 'activities', 
            'volunteer', 'volunteering', 'involvement'
        ]
        
        for header in club_headers:
            # Match section
            pattern = rf'{header}(?:\s+(?:&|and)\s+\w+)?\s*[:\-]?\s*\n(.*?)(?=\n(?:publications|certifications|awards|education|professional experience|experience|$))'
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            
            if matches:
                club_section = matches[0]
                
                # Split by bullet points
                club_items = re.split(r'\n(?=[•\-*])', club_section)
                
                for item in club_items:
                    item = item.strip('•\-* \t\n')
                    # Remove numbering
                    item = re.sub(r'^\s*\d+\.\s*', '', item)
                    
                    if len(item) > 10 and len(item) < 300:
                        clubs.append(item.strip())
        
        # Remove duplicates
        seen = set()
        unique_clubs = []
        for club in clubs:
            if club not in seen:
                seen.add(club)
                unique_clubs.append(club)
        
        return unique_clubs[:20]
    
    def extract_education(self, text: str) -> List[Dict[str, Any]]:
        """Extract education - IMPROVED."""
        education = []
        
        # Degree keywords
        degrees = [
            'ph\\.?d', 'doctorate', 'doctor of philosophy',
            'master', 'm\\.?s\\.?', 'm\\.?a\\.?', 'mba', 'm\\.?tech', 'm\\.?sc',
            'bachelor', 'b\\.?s\\.?', 'b\\.?a\\.?', 'b\\.?tech', 'b\\.?e\\.?', 'b\\.?sc'
        ]
        
        # Find education section
        edu_pattern = r'education\s*[:\-]?\s*\n(.*?)(?=\n(?:professional experience|experience|technical skills|skills|projects|certifications|$))'
        matches = re.findall(edu_pattern, text, re.IGNORECASE | re.DOTALL)
        
        if not matches:
            return education
        
        edu_section = matches[0]
        
        # Split into individual education entries
        # Look for degree keywords as entry separators
        degree_pattern = r'(?=' + '|'.join([rf'\b{d}' for d in degrees]) + r')'
        entries = re.split(degree_pattern, edu_section, flags=re.IGNORECASE)
        
        for entry in entries:
            entry = entry.strip()
            if len(entry) < 20:
                continue
            
            entry_lower = entry.lower()
            
            # Check if this entry contains a degree
            has_degree = any(re.search(rf'\b{d}\b', entry_lower) for d in degrees)
            if not has_degree:
                continue
            
            edu_info = {
                'degree': entry.split('\n')[0].strip()[:200],
                'gpa': None,
                'gpa_scale': None,
                'gpa_percentage': None,
                'graduated': None,
                'field': None,
                'year': None,
                'institution': None
            }
            
            # Extract GPA
            gpa_patterns = [
                r'gpa[:\s]*([0-9]+\.?[0-9]*)\s*/\s*([0-9]+\.?[0-9]*)',
                r'gpa[:\s]*([0-9]+\.?[0-9]*)',
                r'cgpa[:\s]*([0-9]+\.?[0-9]*)\s*/\s*([0-9]+\.?[0-9]*)',
                r'cgpa[:\s]*([0-9]+\.?[0-9]*)',
            ]
            
            for pattern in gpa_patterns:
                gpa_match = re.search(pattern, entry_lower)
                if gpa_match:
                    groups = gpa_match.groups()
                    if len(groups) == 2 and groups[1]:
                        edu_info['gpa'] = float(groups[0])
                        edu_info['gpa_scale'] = float(groups[1])
                    else:
                        edu_info['gpa'] = float(groups[0])
                        # Infer scale
                        if edu_info['gpa'] <= 4.0:
                            edu_info['gpa_scale'] = 4.0
                        elif edu_info['gpa'] <= 5.0:
                            edu_info['gpa_scale'] = 5.0
                        elif edu_info['gpa'] <= 10.0:
                            edu_info['gpa_scale'] = 10.0
                        else:
                            edu_info['gpa_scale'] = 100.0
                    
                    if edu_info['gpa_scale']:
                        edu_info['gpa_percentage'] = (edu_info['gpa'] / edu_info['gpa_scale']) * 100
                    break
            
            # Extract graduation status
            if any(kw in entry_lower for kw in ['graduated', 'completed', 'earned', 'received']):
                edu_info['graduated'] = True
            elif any(kw in entry_lower for kw in ['pursuing', 'current', 'present', 'ongoing', 'expected', 'candidate']):
                edu_info['graduated'] = False
            
            # Extract field/major
            field_patterns = [
                r'in\s+([A-Z][a-zA-Z\s&,]+?)(?:\s*[\|\n]|,|\s+from|\s+at|$)',
                r'of\s+([A-Z][a-zA-Z\s&,]+?)(?:\s*[\|\n]|,|\s+from|\s+at|$)',
            ]
            for pattern in field_patterns:
                field_match = re.search(pattern, entry)
                if field_match:
                    field = field_match.group(1).strip()
                    # Clean up
                    field = re.sub(r'\s+', ' ', field)
                    if len(field) > 3 and len(field) < 100:
                        edu_info['field'] = field
                        break
            
            # Extract institution
            inst_patterns = [
                r'\|\s*([A-Z][a-zA-Z\s&,\.]+?)\s*\|',
                r'from\s+([A-Z][a-zA-Z\s&,\.]+?)(?:\s*[\|\n]|,|$)',
                r'at\s+([A-Z][a-zA-Z\s&,\.]+?)(?:\s*[\|\n]|,|$)',
            ]
            for pattern in inst_patterns:
                inst_match = re.search(pattern, entry)
                if inst_match:
                    inst = inst_match.group(1).strip()
                    if len(inst) > 5 and len(inst) < 100:
                        edu_info['institution'] = inst
                        break
            
            # Extract year
            year_pattern = r'\b(19|20)\d{2}\b'
            years = re.findall(year_pattern, entry)
            if years:
                edu_info['year'] = int(years[-1])  # Most recent year
            
            education.append(edu_info)
        
        return education[:10]
    
    def extract_experience(self, text: str) -> List[Dict[str, Any]]:
        """Extract work experience - IMPROVED."""
        experience = []
        
        # Find experience section
        exp_pattern = r'(?:professional\s+)?experience\s*[:\-]?\s*\n(.*?)(?=\n(?:education|technical skills|skills|projects|key projects|certifications|$))'
        matches = re.findall(exp_pattern, text, re.IGNORECASE | re.DOTALL)
        
        if not matches:
            return experience
        
        exp_section = matches[0]
        
        # Split by job entries (usually start with bold title or bullet)
        entries = re.split(r'\n(?=[•\-*]|[A-Z][a-z]+\s+[A-Z])', exp_section)
        
        for entry in entries:
            if len(entry.strip()) < 30:
                continue
            
            lines = [l.strip() for l in entry.split('\n') if l.strip()]
            if not lines:
                continue
            
            title_line = lines[0].strip('•\-* \t')
            
            exp_info = {
                'title': title_line[:200],
                'company': None,
                'location': None,
                'start_date': None,
                'end_date': None,
                'current': False,
                'description': ' '.join(lines[1:])[:800] if len(lines) > 1 else ""
            }
            
            # Parse title line (format: "Title | Company | Location | Dates")
            parts = [p.strip() for p in title_line.split('|')]
            
            if len(parts) >= 2:
                exp_info['company'] = parts[1]
            if len(parts) >= 3:
                exp_info['location'] = parts[2]
            
            # Check if current position
            entry_lower = entry.lower()
            if 'present' in entry_lower or 'current' in entry_lower:
                exp_info['current'] = True
            
            # Extract dates
            date_pattern = r'(\w+\s+\d{4})\s*-\s*(\w+\s+\d{4}|present)'
            date_match = re.search(date_pattern, entry, re.IGNORECASE)
            if date_match:
                exp_info['start_date'] = date_match.group(1)
                exp_info['end_date'] = date_match.group(2)
            
            experience.append(exp_info)
        
        # Remove duplicates
        seen_titles = set()
        unique_exp = []
        for exp in experience:
            key = exp['title'][:50]  # Use first 50 chars as key
            if key not in seen_titles:
                seen_titles.add(key)
                unique_exp.append(exp)
        
        return unique_exp[:15]
    
    def assess_academic_performance(self, education: List[Dict[str, Any]]) -> str:
        """
        Assess overall academic performance based on GPA.
        
        Compares all GPAs from education entries and calculates average percentage.
        
        Categories:
        - Excellent: >= 90%
        - Very Good: >= 85%
        - Good: >= 75%
        - Average: >= 65%
        - Below Average: < 65%
        """
        if not education:
            return "Unknown"
        
        # Extract GPA percentages from all education entries
        gpas = [e['gpa_percentage'] for e in education if e.get('gpa_percentage')]
        
        if not gpas:
            return "Unknown"
        
        # Calculate average GPA percentage across all degrees
        avg_gpa = sum(gpas) / len(gpas)
        
        # Categorize based on average
        if avg_gpa >= 90:
            return "Excellent"
        elif avg_gpa >= 85:
            return "Very Good"
        elif avg_gpa >= 75:
            return "Good"
        elif avg_gpa >= 65:
            return "Average"
        else:
            return "Below Average"
    
    def determine_domain_proficiency(self, skills: List[str], projects: List[Dict], 
                                     education: List[Dict]) -> Dict[str, str]:
        """Determine main domain and proficiency level."""
        
        # Helper function to safely get string value
        def safe_str(value):
            return str(value) if value is not None else ''
        
        # Domain keywords
        domains = {
            'Data Science': ['data science', 'data', 'analytics', 'machine learning', 'ml', 'ai', 
                            'statistics', 'statistical', 'data analytics', 'data engineer'],
            'Machine Learning': ['machine learning', 'deep learning', 'neural network', 'ml', 'ai',
                                'tensorflow', 'pytorch', 'keras', 'scikit', 'reinforcement learning'],
            'Computer Science': ['python', 'java', 'c++', 'programming', 'software', 'coding', 
                                'algorithm', 'computer science', 'software engineer'],
            'Web Development': ['react', 'angular', 'vue', 'html', 'css', 'javascript', 'web', 
                               'frontend', 'backend', 'full stack', 'node', 'express'],
            'Mobile Development': ['android', 'ios', 'swift', 'kotlin', 'mobile', 'app development',
                                  'react native', 'flutter'],
            'Cloud Computing': ['aws', 'azure', 'gcp', 'cloud', 'docker', 'kubernetes', 'devops',
                               'cloud computing', 'serverless'],
            'Cybersecurity': ['security', 'penetration', 'ethical hacking', 'cryptography',
                             'cybersecurity', 'infosec'],
        }
        
        # Count domain matches
        domain_scores = {domain: 0 for domain in domains}
        
        # Combine all text safely
        all_text = ' '.join(skills).lower()
        
        # Add project text
        project_texts = []
        for p in projects:
            text = safe_str(p.get('name', '')) + ' ' + safe_str(p.get('description', ''))
            project_texts.append(text)
        all_text += ' ' + ' '.join(project_texts).lower()
        
        # Add education text
        edu_texts = []
        for e in education:
            text = safe_str(e.get('field', '')) + ' ' + safe_str(e.get('degree', ''))
            edu_texts.append(text)
        all_text += ' ' + ' '.join(edu_texts).lower()
        
        for domain, keywords in domains.items():
            for keyword in keywords:
                domain_scores[domain] += all_text.count(keyword)
        
        # Find top domain
        top_domain = max(domain_scores, key=domain_scores.get)
        top_score = domain_scores[top_domain]
        
        if top_score == 0:
            top_domain = 'General'
            proficiency = 'Beginner'
        else:
            # Determine proficiency based on score
            if top_score >= 15:
                proficiency = 'Advanced'
            elif top_score >= 8:
                proficiency = 'Intermediate'
            else:
                proficiency = 'Beginner'
        
        return {
            'domain': top_domain,
            'proficiency_level': proficiency,
            'score': top_score
        }
    
    def create_individual_embeddings(self, items: List[Any], item_type: str) -> List[Dict[str, Any]]:
        """Create individual embeddings for each item in a list."""
        results = []
        
        # Helper to safely get string
        def safe_str(val):
            return str(val) if val is not None else ''
        
        for idx, item in enumerate(items):
            # Determine text to embed based on item type
            if item_type == 'skill':
                text = safe_str(item)  # Skills are strings
                item_data = {'skill': item}
            elif item_type == 'project':
                name = safe_str(item.get('name', ''))
                desc = safe_str(item.get('description', ''))
                text = f"{name}. {desc}"
                item_data = item
            elif item_type == 'club':
                text = safe_str(item)  # Clubs are strings
                item_data = {'club': item}
            elif item_type == 'education':
                degree = safe_str(item.get('degree', ''))
                field = safe_str(item.get('field', 'N/A'))
                inst = safe_str(item.get('institution', 'N/A'))
                text = f"{degree}. Field: {field}. Institution: {inst}"
                item_data = item
            elif item_type == 'experience':
                title = safe_str(item.get('title', ''))
                desc = safe_str(item.get('description', ''))[:200]
                text = f"{title}. {desc}"
                item_data = item
            else:
                continue
            
            # Skip if text is too short
            if len(text.strip()) < 3:
                continue
            
            # Generate embedding
            try:
                embedding = self.embedding_model.encode(
                    [text],
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    normalize_embeddings=True
                )[0]
                
                results.append({
                    'index': idx,
                    'type': item_type,
                    'text': text,
                    'embedding': embedding,
                    'data': item_data
                })
            except Exception as e:
                print(f"Warning: Could not create embedding for {item_type} {idx}: {e}")
                continue
        
        return results
    
    def process_cv(self, pdf_path: str) -> Dict[str, Any]:
        """Process CV and return data with individual embeddings for each item."""
        text = self.extract_text_from_pdf(pdf_path)
        
        print("Extracting information...")
        skills = self.extract_skills(text)
        projects = self.extract_projects(text)
        clubs = self.extract_clubs_activities(text)
        education = self.extract_education(text)
        experience = self.extract_experience(text)
        
        academic_performance = self.assess_academic_performance(education)
        domain_info = self.determine_domain_proficiency(skills, projects, education)
        
        graduation_status = "Unknown"
        if education:
            # Check all education entries
            graduated_count = sum(1 for e in education if e.get('graduated') is True)
            pursuing_count = sum(1 for e in education if e.get('graduated') is False)
            
            if pursuing_count > 0:
                # If any degree is being pursued, status is "Pursuing"
                graduation_status = "Pursuing"
            elif graduated_count > 0:
                # If all completed degrees show graduated, status is "Graduated"
                graduation_status = "Graduated"
            # else remains "Unknown" if no clear status found
        
        metadata = {
            'emails': self.extract_email(text),
            'phones': self.extract_phone(text),
            'urls': self.extract_urls(text),
            'academic_performance': academic_performance,
            'graduation_status': graduation_status,
            'main_domain': domain_info['domain'],
            'domain_proficiency': domain_info['proficiency_level']
        }
        
        print("Creating embeddings...")
        # Create individual embeddings for each category
        skill_embeddings = self.create_individual_embeddings(skills, 'skill')
        project_embeddings = self.create_individual_embeddings(projects, 'project')
        club_embeddings = self.create_individual_embeddings(clubs, 'club')
        education_embeddings = self.create_individual_embeddings(education, 'education')
        experience_embeddings = self.create_individual_embeddings(experience, 'experience')
        
        return {
            'raw_text': text,
            'metadata': metadata,
            'skills': {
                'raw': skills,
                'embeddings': skill_embeddings,
                'count': len(skill_embeddings)
            },
            'projects': {
                'raw': projects,
                'embeddings': project_embeddings,
                'count': len(project_embeddings)
            },
            'clubs': {
                'raw': clubs,
                'embeddings': club_embeddings,
                'count': len(club_embeddings)
            },
            'education': {
                'raw': education,
                'embeddings': education_embeddings,
                'count': len(education_embeddings)
            },
            'experience': {
                'raw': experience,
                'embeddings': experience_embeddings,
                'count': len(experience_embeddings)
            },
            'embedding_dimension': 384
        }


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def process_mentorship_cv(pdf_path: str) -> Dict[str, Any]:
    """Process CV and return individual embeddings for each item."""
    processor = CVProcessor()
    return processor.process_cv(pdf_path)


# ============================================================================
# TEST FUNCTION
# ============================================================================

def test_cv_processor():
    """Test CV processor with sample CV."""
    print("="*80)
    print("TESTING MENTORSHIP CV PROCESSOR")
    print("="*80)
    
    # Find sample CV
    possible_paths = [
        Path.home() / "Desktop" / "Cv" / "cvs" / "sample_cv.pdf",
        Path(__file__).parent.parent.parent / "datasets" / "cvs" / "sample_cv.pdf",
        Path(__file__).parent / "sample_cv.pdf",
        Path.cwd() / "mentorship-program" / "datasets" / "cvs" / "sample_cv.pdf",
        Path.cwd() / "datasets" / "cvs" / "sample_cv.pdf",
        Path.cwd() / "sample_cv.pdf",
        Path.cwd() / "comprehensive_sample_cv.pdf",
    ]
    
    cv_path = None
    for path in possible_paths:
        if path.exists():
            cv_path = str(path)
            break
    
    if not cv_path:
        print("\n✗ Error: sample_cv.pdf not found")
        print("Searched in:")
        for path in possible_paths:
            print(f"  - {path}")
        print(f"Current directory: {Path.cwd()}")
        print(f"Script location: {Path(__file__).parent}")
        return
    
    print(f"✓ Found CV: {cv_path}")
    print("Processing...")
    
    # Process CV
    result = process_mentorship_cv(cv_path)
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    # Display metadata
    print("\n[METADATA]")
    print(f"  Emails: {', '.join(result['metadata']['emails']) if result['metadata']['emails'] else 'None'}")
    print(f"  Phones: {', '.join(result['metadata']['phones']) if result['metadata']['phones'] else 'None'}")
    if result['metadata']['urls']:
        print(f"  URLs: {', '.join(result['metadata']['urls'][:3])}")
    print(f"  Academic Performance: {result['metadata']['academic_performance']}")
    print(f"  Graduation Status: {result['metadata']['graduation_status']}")
    print(f"  Main Domain: {result['metadata']['main_domain']}")
    print(f"  Domain Proficiency: {result['metadata']['domain_proficiency']}")
    
    print(f"\n[EDUCATION] ({len(result['education']['raw'])} entries)")
    for edu in result['education']['raw']:
        print(f"  • {edu['degree'][:70]}")
        if edu.get('institution'):
            print(f"    Institution: {edu['institution']}")
        if edu.get('field'):
            print(f"    Field: {edu['field']}")
        if edu.get('gpa'):
            print(f"    GPA: {edu['gpa']}/{edu['gpa_scale']} ({edu['gpa_percentage']:.1f}%)")
    
    print(f"\n[SKILLS] ({len(result['skills']['raw'])} found)")
    for skill in result['skills']['raw'][:5]:
        print(f"  • {skill}")
    if len(result['skills']['raw']) > 5:
        print(f"  ... and {len(result['skills']['raw']) - 5} more")
    
    print(f"\n[PROJECTS] ({len(result['projects']['raw'])} found)")
    for proj in result['projects']['raw']:
        print(f"  • {proj['name']}")
    
    print(f"\n[EXPERIENCE] ({len(result['experience']['raw'])} found)")
    for exp in result['experience']['raw']:
        print(f"  • {exp['title'][:70]}")
    
    print(f"\n[CLUBS & ACTIVITIES] ({len(result['clubs']['raw'])} found)")
    for club in result['clubs']['raw'][:5]:
        print(f"  • {club[:70]}")
    if len(result['clubs']['raw']) > 5:
        print(f"  ... and {len(result['clubs']['raw']) - 5} more")
    
    print("\n[INDIVIDUAL EMBEDDINGS]")
    print(f"  ✓ Skills: {result['skills']['count']} individual embeddings")
    print(f"  ✓ Projects: {result['projects']['count']} individual embeddings")
    print(f"  ✓ Clubs: {result['clubs']['count']} individual embeddings")
    print(f"  ✓ Education: {result['education']['count']} individual embeddings")
    print(f"  ✓ Experience: {result['experience']['count']} individual embeddings")
    print(f"  ✓ Total: {sum([result[k]['count'] for k in ['skills', 'projects', 'clubs', 'education', 'experience']])} embeddings (384 dimensions each)")


if __name__ == "__main__":
    test_cv_processor()