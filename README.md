# 🏥 Deep Medical Diagnosis  

Un projet d'intelligence artificielle pour l'analyse des radiographies thoraciques et cérébrales, avec détection assistée des maladies.  

---

## 📌 Contenu du Modèle  

### 🫁 **CHEST (Poitrine)**  

#### 🔹 **X-ray**  
- **Normal**  
- **Tuberculosis**  
- **Pneumonia**  
  - Virus  
  - Bacteria  
- **Covid**  
- **Fibrosis**  

#### 🔹 **CT (Tomodensitométrie)**  
- **Normal**  
- **Infectieux**  
  - Normal  
  - Covid  
  - CAP (Pneumonie acquise en communauté)  
- **Cancer**  
  - Bénin  
  - Adenocarcinoma  
  - Large Cell Carcinoma  
  - Squamous Cell Carcinoma  

---

### 🧠 **BRAIN (Cerveau)**  
- **Normal**  
- **Glioma**  
- **Meningioma**  
- **Pituitary**  

---

## 🚀 Installation & Exécution  

```bash
# Clonez le repo
git clone https://github.com/Abdelmoughitelhassani/deep-medical-diagnosis.git
cd deep-medical-diagnosis

# Installez les dépendances
pip install -r requirements.txt

# Exécutez l'application
python manage.py runserver

 
