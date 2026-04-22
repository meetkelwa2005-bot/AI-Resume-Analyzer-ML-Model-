import PyPDF2
import spacy
import re
# Load the spacy English model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading language model for the spacy POS tagger")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
def extract_text_from_pdf(pdf_file):
    """Extracts text from an uploaded PDF file."""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text
COMMON_SKILLS = [
    'python', 'java', 'c++', 'c#', 'react', 'node.js', 'sql', 'mysql', 'postgresql', 
    'mongodb', 'aws', 'docker', 'kubernetes', 'machine learning', 'deep learning', 
    'backend', 'frontend', 'hibernate', 'jdbc', 'css', 'html', 'javascript', 
    'typescript', 'django', 'flask', 'spring boot', 'git', 'linux'
]
def clean_extracted_text(text):
    """Cleans noisy text strings."""
    return re.sub(r'\s+', ' ', text).strip()
def extract_skills(text):
    """Extracts common technical skills based on a predefined list."""
    found_skills = set()
    text_lower = text.lower()
    for skill in COMMON_SKILLS:
        # Use word boundaries to avoid partial matches (e.g., 'c' in 'cat')
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            # Capitalize properly for display
            found_skills.add(skill.title() if len(skill) > 3 else skill.upper())
    return list(found_skills)
def extract_entities(text):
    """Extracts named entities from text using Spacy, filtering noise."""
    doc = nlp(text)
    entities = {}
    
    # We only care about a few types and want to filter out long garbage strings
    allowed_labels = ['ORG', 'PERSON', 'GPE']
    
    for ent in doc.ents:
        if ent.label_ in allowed_labels:
            clean_text = clean_extracted_text(ent.text)
            # Filter out things that are too long, too short, or contain weird characters
            if 2 < len(clean_text) < 30 and not re.search(r'[^a-zA-Z0-9\s&.-]', clean_text):
                if ent.label_ not in entities:
                    entities[ent.label_] = []
                if clean_text not in entities[ent.label_]:
                    entities[ent.label_].append(clean_text)
                    
    # Add our custom skills extractor
    entities['Skills'] = extract_skills(text)
    
    return entities
def extract_emails_and_phones(text):
    """Extracts email addresses and phone numbers using Regex."""
    emails = re.findall(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', text)
    phones = re.findall(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text)
    return {
        "Emails": list(set(emails)),
        "Phone Numbers": list(set(phones))
    }
def process_resume(pdf_file):
    """Main function to process resume and return extracted data."""
    text = extract_text_from_pdf(pdf_file)
    entities = extract_entities(text)
    contact_info = extract_emails_and_phones(text)
    
    return {
        "text": text,
        "entities": entities,
        "contact_info": contact_info
    }
