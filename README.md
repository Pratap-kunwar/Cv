# Cv
AI-powered CV processing for mentorship matching
# Mentorship CV Processor - Complete Guide

## Overview

This CV processor extracts information from PDF resumes and generates **individual embeddings** for each skill, project, club, education entry, and work experience. These embeddings can be used for semantic matching between mentees and mentors.

---

## Table of Contents

1. [What Are Embeddings?](#what-are-embeddings)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Understanding the Output](#understanding-the-output)
5. [Data Structure Explained](#data-structure-explained)
6. [Accessing Embeddings](#accessing-embeddings)
7. [Use Cases](#use-cases)
8. [Advanced Usage](#advanced-usage)
9. [Troubleshooting](#troubleshooting)

---

## What Are Embeddings?

**Embeddings** are numerical representations (vectors) of text that capture semantic meaning. 

- Each embedding is a **384-dimensional vector** (an array of 384 numbers)
- Similar concepts have similar embeddings
- Used for semantic search, similarity matching, and clustering

### Example:
```python
"Python programming" → [0.123, -0.456, 0.789, ..., 0.234]  # 384 numbers
"Java development"   → [0.145, -0.423, 0.812, ..., 0.267]  # Similar values!
"Cooking recipes"    → [-0.567, 0.891, -0.234, ..., -0.123] # Very different!
```

---

## Installation

### 1. Install Required Packages

```bash
pip install PyPDF2 sentence-transformers numpy pandas
```

### 2. Download the Processor

Save `mentorship_cv_processor.py` to your project directory.

---

## Quick Start

### Basic Usage

```python
from mentorship_cv_processor import process_mentorship_cv

# Process a CV
result = process_mentorship_cv("path/to/cv.pdf")

# Check what was extracted
print(f"Skills found: {result['skills']['count']}")
print(f"Projects found: {result['projects']['count']}")
print(f"Total embeddings: {result['skills']['count'] + result['projects']['count']}")
```

### Running the Test

```bash
python mentorship_cv_processor.py
```

**Expected Output:**
```
================================================================================
TESTING MENTORSHIP CV PROCESSOR
================================================================================
✓ Found CV: sample_cv.pdf
Processing...
Extracting information...
Creating embeddings...

================================================================================
RESULTS
================================================================================
[METADATA]
  Emails: sarah.chen@email.com
  Phones: +1 (555) 987-6543
  Academic Performance: Excellent
  Graduation Status: Graduated
  Main Domain: Machine Learning
  Domain Proficiency: Advanced

[EDUCATION] (3 entries)
  • Ph.D. in Computer Science - Machine Learning
    GPA: 3.92/4.0 (98.0%)
  • Master of Science in Data Science
    GPA: 3.88/4.0 (97.0%)
  • Bachelor of Science...
    GPA: 3.85/4.0 (96.2%)

[SKILLS] (20 found)
  • python
  • tensorflow
  • pytorch
  • react
  • docker
  ... and 15 more

[PROJECTS] (4 found)
  • AI-Powered Content Recommendation Engine
  • Real-Time Fraud Detection System
  • Natural Language Processing Platform
  • Computer Vision for Medical Image Analysis

[INDIVIDUAL EMBEDDINGS]
  ✓ Skills: 20 individual embeddings
  ✓ Projects: 4 individual embeddings
  ✓ Clubs: 7 individual embeddings
  ✓ Education: 3 individual embeddings
  ✓ Experience: 3 individual embeddings
  ✓ Total: 37 embeddings (384 dimensions each)
```

---

## Understanding the Output

### Console Output Shows **Counts**, Not Values

When you run the processor, you see:
```
[INDIVIDUAL EMBEDDINGS]
  ✓ Skills: 30 individual embeddings
  ✓ Projects: 2 individual embeddings
  ✓ Total: 32 embeddings (384 dimensions each)
```

**This is just a summary!** The actual embeddings ARE created and stored in memory.

### Why Don't We Print All Embeddings?

Printing 32 embeddings × 384 dimensions = **12,288 numbers** would flood your console!

Instead, the embeddings are stored in the `result` dictionary and can be accessed programmatically.

---

## Data Structure Explained

### Complete Structure

```python
result = {
    'raw_text': "Full CV text...",
    
    'metadata': {
        'emails': ['sarah.chen@email.com'],
        'phones': ['+1 (555) 987-6543'],
        'urls': ['https://github.com/sarahchen', ...],
        'academic_performance': 'Excellent',  # Based on GPA comparison
        'graduation_status': 'Graduated',
        'main_domain': 'Machine Learning',
        'domain_proficiency': 'Advanced'
    },
    
    'skills': {
        'raw': ['Python', 'TensorFlow', 'PyTorch', ...],  # List of skill strings
        'embeddings': [
            {
                'index': 0,
                'type': 'skill',
                'text': 'Python',
                'embedding': array([0.123, -0.456, ..., 0.789]),  # 384-dimensional vector
                'data': {'skill': 'Python'}
            },
            {
                'index': 1,
                'type': 'skill',
                'text': 'TensorFlow',
                'embedding': array([0.234, -0.567, ..., 0.891]),  # 384-dimensional vector
                'data': {'skill': 'TensorFlow'}
            },
            # ... more skills
        ],
        'count': 20  # Number of embeddings created
    },
    
    'projects': {
        'raw': [
            {'name': 'AI Recommendation Engine', 'description': '...'},
            {'name': 'Fraud Detection System', 'description': '...'}
        ],
        'embeddings': [
            {
                'index': 0,
                'type': 'project',
                'text': 'AI Recommendation Engine. Developed hybrid system...',
                'embedding': array([...]),  # 384-dimensional vector
                'data': {'name': '...', 'description': '...'}
            },
            # ... more projects
        ],
        'count': 4
    },
    
    'clubs': {
        'raw': ['President of Data Science Club', ...],
        'embeddings': [...],
        'count': 7
    },
    
    'education': {
        'raw': [
            {
                'degree': 'Ph.D. in Computer Science',
                'field': 'Machine Learning',
                'institution': 'Stanford University',
                'gpa': 3.92,
                'gpa_scale': 4.0,
                'gpa_percentage': 98.0,
                'graduated': True
            },
            # ... more degrees
        ],
        'embeddings': [...],
        'count': 3
    },
    
    'experience': {
        'raw': [
            {
                'title': 'Senior ML Engineer',
                'company': 'Tech Innovations Inc.',
                'description': '...',
                'current': True
            },
            # ... more experience
        ],
        'embeddings': [...],
        'count': 3
    },
    
    'embedding_dimension': 384
}
```

---

## Accessing Embeddings

### Example 1: Print First Skill Embedding

```python
import numpy as np
from mentorship_cv_processor import process_mentorship_cv

# Process CV
result = process_mentorship_cv("sample_cv.pdf")

# Access first skill embedding
if result['skills']['embeddings']:
    first_skill = result['skills']['embeddings'][0]
    
    print(f"Skill: {first_skill['text']}")
    print(f"Embedding shape: {first_skill['embedding'].shape}")
    print(f"Embedding type: {type(first_skill['embedding'])}")
    print(f"First 10 values: {first_skill['embedding'][:10]}")
    print(f"Norm (should be ~1.0): {np.linalg.norm(first_skill['embedding']):.4f}")
```

**Output:**
```
Skill: Python
Embedding shape: (384,)
Embedding type: <class 'numpy.ndarray'>
First 10 values: [ 0.123 -0.456  0.789 -0.234  0.567 -0.891  0.345 -0.678  0.912 -0.456]
Norm (should be ~1.0): 1.0000
```

### Example 2: Loop Through All Embeddings

```python
# Loop through all skill embeddings
for skill_item in result['skills']['embeddings']:
    skill_name = skill_item['text']
    skill_embedding = skill_item['embedding']
    
    print(f"{skill_name}: {skill_embedding.shape}")
    # Use skill_embedding for matching, similarity, etc.
```

### Example 3: Calculate Similarity Between Two Skills

```python
from numpy import dot
from numpy.linalg import norm

def cosine_similarity(emb1, emb2):
    """Calculate cosine similarity between two embeddings."""
    return dot(emb1, emb2) / (norm(emb1) * norm(emb2))

# Get two skill embeddings
skill1 = result['skills']['embeddings'][0]['embedding']  # Python
skill2 = result['skills']['embeddings'][1]['embedding']  # TensorFlow

similarity = cosine_similarity(skill1, skill2)
print(f"Similarity: {similarity:.4f}")  # Value between -1 and 1
```

---

## Use Cases

### 1. Mentee-Mentor Matching

```python
# Process mentee CV
mentee_result = process_mentorship_cv("mentee_cv.pdf")

# Process mentor CV
mentor_result = process_mentorship_cv("mentor_cv.pdf")

# Compare mentee's skills with mentor's skills
for mentee_skill in mentee_result['skills']['embeddings']:
    for mentor_skill in mentor_result['skills']['embeddings']:
        similarity = cosine_similarity(
            mentee_skill['embedding'],
            mentor_skill['embedding']
        )
        
        if similarity > 0.8:  # High similarity threshold
            print(f"Match found!")
            print(f"  Mentee: {mentee_skill['text']}")
            print(f"  Mentor: {mentor_skill['text']}")
            print(f"  Similarity: {similarity:.4f}")
```

### 2. Find Similar Projects

```python
def find_similar_projects(query_project_emb, all_projects, top_k=5):
    """Find top K most similar projects."""
    similarities = []
    
    for proj in all_projects['embeddings']:
        sim = cosine_similarity(query_project_emb, proj['embedding'])
        similarities.append((proj['text'], sim))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_k]

# Find similar projects
query_project = result['projects']['embeddings'][0]['embedding']
similar = find_similar_projects(query_project, result['projects'])

print("Similar projects:")
for project_text, similarity in similar:
    print(f"  {project_text[:50]}... (similarity: {similarity:.4f})")
```
#Projectcv
### 3. Export Embeddings to DataFrame

```python
import pandas as pd

# Flatten all embeddings into a DataFrame
rows = []

for skill_item in result['skills']['embeddings']:
    rows.append({
        'type': 'skill',
        'text': skill_item['text'],
        'embedding': skill_item['embedding'].tolist()  # Convert to list for CSV
    })

for proj_item in result['projects']['embeddings']:
    rows.append({
        'type': 'project',
        'text': proj_item['text'],
        'embedding': proj_item['embedding'].tolist()
    })

df = pd.DataFrame(rows)
print(df.head())

# Save to CSV
df.to_csv('cv_embeddings.csv', index=False)
```

---

## Advanced Usage

### Save Embeddings to File

```python
import json

# Save result to JSON
result_copy = result.copy()

# Convert numpy arrays to lists for JSON serialization
for category in ['skills', 'projects', 'clubs', 'education', 'experience']:
    for item in result_copy[category]['embeddings']:
        item['embedding'] = item['embedding'].tolist()

with open('cv_embeddings.json', 'w') as f:
    json.dump(result_copy, f, indent=2)

print("✓ Saved embeddings to cv_embeddings.json")
```

### Load Embeddings from File

```python
import json
import numpy as np

# Load from JSON
with open('cv_embeddings.json', 'r') as f:
    loaded_result = json.load(f)

# Convert lists back to numpy arrays
for category in ['skills', 'projects', 'clubs', 'education', 'experience']:
    for item in loaded_result[category]['embeddings']:
        item['embedding'] = np.array(item['embedding'])

# Now you can use loaded_result just like the original result
```

### Batch Process Multiple CVs

```python
from pathlib import Path

cv_folder = Path("cvs")
all_results = {}

for cv_file in cv_folder.glob("*.pdf"):
    print(f"Processing {cv_file.name}...")
    result = process_mentorship_cv(str(cv_file))
    all_results[cv_file.stem] = result

print(f"Processed {len(all_results)} CVs")
```

---

## Academic Performance Assessment

The processor automatically assesses academic performance by:

1. **Extracting GPAs** from all education entries
2. **Converting to percentage** (e.g., 3.92/4.0 = 98%)
3. **Calculating average** across all degrees
4. **Categorizing** based on thresholds:

```python
Average GPA ≥ 90%  → "Excellent"
Average GPA ≥ 85%  → "Very Good"
Average GPA ≥ 75%  → "Good"
Average GPA ≥ 65%  → "Average"
Average GPA < 65%  → "Below Average"
```

### Example:
```
Education:
  - Ph.D.: 3.92/4.0 (98%)
  - M.S.: 3.88/4.0 (97%)
  - B.S.: 3.85/4.0 (96.25%)

Average: (98 + 97 + 96.25) / 3 = 97.08%
Result: "Excellent"
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'PyPDF2'"

**Solution:**
```bash
pip install PyPDF2 sentence-transformers numpy pandas
```

### Issue: "FileNotFoundError: sample_cv.pdf not found"

**Solution:**
Place your CV in one of these locations:
- `C:\Users\ASUS\Desktop\Cv\cvs\sample_cv.pdf`
- `datasets/cvs/sample_cv.pdf`
- Current working directory

Or specify the path directly:
```python
result = process_mentorship_cv("C:/path/to/your/cv.pdf")
```

### Issue: "No education entries found"

**Possible causes:**
- CV doesn't have an "Education" section
- Education section format is unusual
- PDF text extraction failed

**Solution:**
Check if text is extracted correctly:
```python
from mentorship_cv_processor import CVProcessor

processor = CVProcessor()
text = processor.extract_text_from_pdf("cv.pdf")
print(text[:500])  # Print first 500 characters
```

### Issue: Embeddings seem incorrect

**Check embedding properties:**
```python
emb = result['skills']['embeddings'][0]['embedding']
print(f"Shape: {emb.shape}")  # Should be (384,)
print(f"Type: {type(emb)}")   # Should be numpy.ndarray
print(f"Norm: {np.linalg.norm(emb)}")  # Should be ~1.0
```

---

## Model Information

- **Model**: `all-MiniLM-L6-v2`
- **Embedding Dimension**: 384
- **Normalization**: Yes (all embeddings are unit vectors)
- **Use Case**: General-purpose sentence embeddings
- **Source**: Hugging Face Sentence Transformers

### Model Performance:
- Fast inference (~1ms per sentence)
- Good balance between speed and accuracy
- Works well for semantic similarity tasks
- Pretrained on large text corpus

---

## Summary

### What This Processor Does:
✅ Extracts structured data from PDF CVs  
✅ Creates individual 384-dimensional embeddings for each item  
✅ Normalizes embeddings to unit vectors  
✅ Assesses academic performance from GPAs  
✅ Determines domain and proficiency level  

### What You Get:
- **39 individual embeddings** (for a typical CV)
- Each embedding is ready for similarity matching
- Structured metadata for filtering
- Raw extracted text for reference

### Next Steps:
1. Process your CV collection
2. Build a similarity matching system
3. Create a recommendation engine
4. Match mentees with mentors

---

## Questions?

If you have questions or need help:
1. Check the [Data Structure Explained](#data-structure-explained) section
2. Review the [Use Cases](#use-cases) examples
3. Try the [Advanced Usage](#advanced-usage) examples

Happy matching! 🚀
