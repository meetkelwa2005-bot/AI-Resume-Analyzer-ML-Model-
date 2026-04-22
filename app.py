from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import os
import uvicorn
from resume_parser import extract_text_from_pdf, process_resume
app = FastAPI(
    title="AI Resume Analyzer API",
    description="API for parsing resumes and predicting job categories using a trained ML model.",
    version="1.0.0"
)
# Global variables for the model and vectorizer
MODEL = None
VECTORIZER = None
class PredictionResponse(BaseModel):
    category: str
    confidence: float
    parsed_entities: dict
    contact_info: dict
    recommended_jobs: list[str]
    suggested_skills: list[str]
# Job Recommendations and Skill Maps
CAREER_MAP = {
    "Backend Engineering": {
        "jobs": ["Backend Developer", "API Engineer", "Systems Architect", "Cloud Backend Engineer", "Python/Java Developer"],
        "core_skills": ["Python", "Java", "Node.js", "C#", "Go", "Sql", "PostgreSQL", "NoSQL", "Docker", "Kubernetes", "Aws", "Azure", "Microservices", "Spring Boot", "Django", "FastAPI", "Redis", "Kafka", "GraphQL", "REST APIs", "CI/CD"]
    },
    "Frontend Engineering": {
        "jobs": ["Frontend Developer", "UI Developer", "React Engineer", "Web Applications Engineer", "Mobile App Developer"],
        "core_skills": ["React", "Javascript", "Typescript", "Html", "Css", "Vue.js", "Next.js", "Angular", "Tailwind CSS", "SASS", "Redux", "Webpack", "Jest", "Cypress", "Responsive Design", "Web Performance", "Figma"]
    },
    "Data Science": {
        "jobs": ["Data Scientist", "Machine Learning Engineer", "Data Analyst", "AI Researcher", "Data Engineer"],
        "core_skills": ["Python", "Machine Learning", "Deep Learning", "Sql", "Pandas", "NumPy", "TensorFlow", "PyTorch", "Scikit-Learn", "Data Visualization", "Tableau", "PowerBI", "NLP", "Computer Vision", "Big Data", "Spark", "Hadoop", "Databricks"]
    },
    "DevOps": {
        "jobs": ["DevOps Engineer", "Cloud Architect", "Site Reliability Engineer (SRE)", "Platform Engineer"],
        "core_skills": ["Aws", "Docker", "Kubernetes", "Linux", "CI/CD", "Terraform", "Git", "Jenkins", "GitHub Actions", "Ansible", "Prometheus", "Grafana", "Bash Scripting", "Python", "Azure", "GCP", "Network Security"]
    },
    "Full Stack Engineering": {
        "jobs": ["Full Stack Developer", "Software Engineer", "Web Application Developer", "Product Engineer"],
        "core_skills": ["React", "Node.js", "Python", "Sql", "Mongodb", "Docker", "Aws", "Javascript", "Typescript", "Next.js", "GraphQL", "Express.js", "PostgreSQL", "Tailwind CSS", "Git", "REST APIs", "System Design"]
    },
    "Design": {
        "jobs": ["UI/UX Designer", "Product Designer", "Graphic Designer", "Interaction Designer"],
        "core_skills": ["Figma", "Adobe XD", "Sketch", "User Research", "Wireframing", "Prototyping", "Css", "Html", "Adobe Illustrator", "Photoshop", "Information Architecture", "Usability Testing", "Design Systems"]
    }
}
@app.on_event("startup")
def load_model():
    """Loads the trained model and vectorizer on startup."""
    global MODEL, VECTORIZER
    model_path = os.path.join("models", "model.pkl")
    vectorizer_path = os.path.join("models", "vectorizer.pkl")
    
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        MODEL = joblib.load(model_path)
        VECTORIZER = joblib.load(vectorizer_path)
        print("Model and Vectorizer loaded successfully.")
    else:
        print("Warning: Model files not found. Please run train.py first.")
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serves a beautiful front-end UI for the API."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Resume Analyzer Pro</title>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
            body {
                font-family: 'Inter', sans-serif;
                background-color: #0e1117;
                color: #fafafa;
                margin: 0;
                padding: 0;
                display: flex;
                flex-direction: column;
                align-items: center;
                min-height: 100vh;
            }
            .container {
                max-width: 900px;
                width: 90%;
                margin: 50px auto;
                background: #1a1c24;
                padding: 40px;
                border-radius: 16px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            }
            h1 {
                text-align: center;
                background: -webkit-linear-gradient(45deg, #4b6cb7, #182848);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-size: 3rem;
                margin-top: 0;
            }
            .upload-btn-wrapper {
                position: relative;
                overflow: hidden;
                display: inline-block;
                width: 100%;
                text-align: center;
                margin-bottom: 20px;
            }
            .btn {
                border: 2px dashed #4b6cb7;
                color: #4b6cb7;
                background-color: transparent;
                padding: 20px 40px;
                border-radius: 8px;
                font-size: 1.2rem;
                font-weight: bold;
                cursor: pointer;
                width: 100%;
                transition: all 0.3s;
            }
            .upload-btn-wrapper:hover .btn {
                background-color: rgba(75, 108, 183, 0.1);
            }
            .upload-btn-wrapper input[type=file] {
                font-size: 100px;
                position: absolute;
                left: 0;
                top: 0;
                opacity: 0;
                cursor: pointer;
                height: 100%;
            }
            #submitBtn {
                background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 15px 30px;
                font-size: 1.2rem;
                font-weight: bold;
                cursor: pointer;
                width: 100%;
                transition: transform 0.2s;
            }
            #submitBtn:hover {
                transform: scale(1.02);
            }
            #results {
                margin-top: 30px;
                display: none;
            }
            .score-card {
                background: linear-gradient(135deg, #1f4037 0%, #99f2c8 100%);
                border-radius: 12px;
                padding: 30px;
                text-align: center;
                box-shadow: 0 8px 32px rgba(0,0,0,0.3);
                margin-bottom: 20px;
            }
            .score-text {
                font-size: 3rem;
                font-weight: 800;
                color: white;
                margin: 0;
            }
            .grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
            }
            .info-box {
                background: #232630;
                padding: 25px;
                border-radius: 12px;
                margin-bottom: 20px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            }
            .info-box h3 {
                margin-top: 0;
                color: #99f2c8;
                border-bottom: 1px solid #333;
                padding-bottom: 10px;
                margin-bottom: 15px;
            }
            .tag {
                display: inline-block;
                background: #2d3748;
                color: #e2e8f0;
                padding: 6px 14px;
                border-radius: 20px;
                margin: 4px;
                font-size: 0.95rem;
                border: 1px solid #4a5568;
            }
            .tag.highlight {
                background: #4b6cb7;
                border-color: #4b6cb7;
                color: white;
            }
            .tag.suggest {
                background: transparent;
                border-color: #99f2c8;
                color: #99f2c8;
                border-style: dashed;
            }
            ul.job-list {
                list-style-type: none;
                padding: 0;
                margin: 0;
            }
            ul.job-list li {
                background: rgba(255,255,255,0.05);
                margin-bottom: 8px;
                padding: 10px 15px;
                border-radius: 6px;
                border-left: 4px solid #99f2c8;
            }
            #loading {
                display: none;
                text-align: center;
                margin-top: 20px;
                color: #99f2c8;
                font-weight: bold;
                font-size: 1.2rem;
            }
            #fileNameDisplay {
                display: block;
                text-align: center;
                margin-bottom: 15px;
                color: #aaa;
            }
            @media (max-width: 768px) {
                .grid {
                    grid-template-columns: 1fr;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>AI Resume Analyzer Pro</h1>
            <p style="text-align: center; color: #aaa; margin-bottom: 30px;">Upload your PDF resume to get an instant Career Analysis Report.</p>
            
            <div class="upload-btn-wrapper">
                <button class="btn">Select Resume (PDF)</button>
                <input type="file" id="resumeFile" accept="application/pdf" onchange="document.getElementById('fileNameDisplay').innerText = 'Selected: ' + this.files[0].name;" />
            </div>
            <span id="fileNameDisplay"></span>
            
            <button id="submitBtn" onclick="analyzeResume()">Generate Career Report</button>
            <div id="loading">Analyzing resume... Please wait...</div>
            <div id="results">
                <div class="score-card">
                    <div style="color: #e0e0e0; margin-bottom: 10px; font-size: 1.2rem; text-transform: uppercase; letter-spacing: 2px;">Your Career Profile</div>
                    <div class="score-text" id="resCategory">Software Engineer</div>
                    <div style="color: #a8d5ba; margin-top: 10px; font-weight: bold;">Match Confidence: <span id="resConfidence">95</span>%</div>
                </div>
                <div class="grid">
                    <div class="info-box">
                        <h3>Roles You Can Apply For</h3>
                        <p style="font-size: 0.9rem; color: #aaa; margin-top: -10px;">Based on your profile, you are a strong fit for:</p>
                        <ul class="job-list" id="resJobs"></ul>
                    </div>
                    <div class="info-box">
                        <h3>Contact Details</h3>
                        <p><strong>Emails:</strong> <br><span id="resEmails" style="color: #aaa;"></span></p>
                        <p><strong>Phones:</strong> <br><span id="resPhones" style="color: #aaa;"></span></p>
                    </div>
                </div>
                <div class="info-box">
                    <h3>Skills Analysis</h3>
                    <p style="font-size: 0.9rem; color: #aaa; margin-top: -10px;">What you have vs. what you should learn to boost your chances.</p>
                    <div style="margin-bottom: 15px;">
                        <strong style="display: block; margin-bottom: 8px;">Your Verified Skills:</strong>
                        <div id="resSkills"></div>
                    </div>
                    <div>
                        <strong style="display: block; margin-bottom: 8px; color: #99f2c8;">Skills to Learn (Recommended):</strong>
                        <div id="resSuggestSkills"></div>
                    </div>
                </div>
            </div>
        </div>
        <script>
            async function analyzeResume() {
                const fileInput = document.getElementById('resumeFile');
                if (fileInput.files.length === 0) {
                    alert('Please select a PDF file first!');
                    return;
                }
                document.getElementById('loading').style.display = 'block';
                document.getElementById('results').style.display = 'none';
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });
                    if (!response.ok) {
                        throw new Error("Server returned " + response.status);
                    }
                    const data = await response.json();
                    
                    document.getElementById('resCategory').innerText = data.category;
                    document.getElementById('resConfidence').innerText = data.confidence;
                    
                    // Render Jobs
                    const jobsUl = document.getElementById('resJobs');
                    jobsUl.innerHTML = '';
                    data.recommended_jobs.forEach(job => {
                        jobsUl.innerHTML += `<li>${job}</li>`;
                    });
                    // Render Skills
                    const skillsDiv = document.getElementById('resSkills');
                    skillsDiv.innerHTML = '';
                    if (data.parsed_entities.Skills && data.parsed_entities.Skills.length > 0) {
                        data.parsed_entities.Skills.forEach(skill => {
                            skillsDiv.innerHTML += `<span class="tag highlight">${skill}</span>`;
                        });
                    } else {
                        skillsDiv.innerHTML = '<span style="color: #aaa;">No technical skills detected.</span>';
                    }
                    // Render Suggested Skills
                    const suggestDiv = document.getElementById('resSuggestSkills');
                    suggestDiv.innerHTML = '';
                    if (data.suggested_skills && data.suggested_skills.length > 0) {
                        data.suggested_skills.forEach(skill => {
                            suggestDiv.innerHTML += `<span class="tag suggest">+ ${skill}</span>`;
                        });
                    } else {
                        suggestDiv.innerHTML = '<span style="color: #aaa;">You have all the core skills for this role!</span>';
                    }
                    // Render Contact Info
                    document.getElementById('resEmails').innerText = data.contact_info.Emails.join(', ') || 'None found';
                    document.getElementById('resPhones').innerText = data.contact_info['Phone Numbers'].join(', ') || 'None found';
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('results').style.display = 'block';
                } catch (error) {
                    alert('Error analyzing resume: ' + error.message);
                    document.getElementById('loading').style.display = 'none';
                }
            }
        </script>
    </body>
    </html>
    """
    return html_content
@app.post("/predict", response_model=PredictionResponse)
async def predict_resume(file: UploadFile = File(...)):
    """
    Endpoint to upload a PDF resume, parse it, and predict the job category.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    if MODEL is None or VECTORIZER is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Ensure train.py has been run.")
    try:
        # Save the uploaded file temporarily to process it
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())
            
        # Parse text and entities using our existing parser module
        with open(temp_file_path, "rb") as f:
            resume_data = process_resume(f)
            
        os.remove(temp_file_path)
        
        resume_text = resume_data['text']
        
        if not resume_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from the provided PDF.")
            
        # Transform the text using the loaded vectorizer
        text_vectorized = VECTORIZER.transform([resume_text])
        
        # Predict category
        prediction = MODEL.predict(text_vectorized)[0]
        
        # Calculate confidence (probability of the predicted class)
        probabilities = MODEL.predict_proba(text_vectorized)[0]
        confidence = float(max(probabilities))
        
        # Retrieve mapped recommendations
        map_data = CAREER_MAP.get(prediction, {"jobs": ["Software Professional"], "core_skills": []})
        recommended_jobs = map_data["jobs"]
        
        # Calculate suggested skills
        found_skills = [s.title() for s in resume_data['entities'].get('Skills', [])]
        core_skills = [s.title() for s in map_data["core_skills"]]
        
        # Find skills in core that are not in found
        suggested_skills = [skill for skill in core_skills if skill.upper() not in [f.upper() for f in found_skills]]
        
        return PredictionResponse(
            category=prediction,
            confidence=round(confidence * 100, 2),
            parsed_entities=resume_data['entities'],
            contact_info=resume_data['contact_info'],
            recommended_jobs=recommended_jobs,
            suggested_skills=suggested_skills[:15]  # Top 15 suggestions for a comprehensive list
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
