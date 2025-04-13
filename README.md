# Deep Medical Diagnosis  

## Description  
**Deep Medical Diagnosis** est une application web développée avec Django pour l'analyse et le diagnostic assisté par IA des radiographies thoraciques et cérébrales. L'application intègre :  
- Des modèles de **Deep Learning** et de **Transfer Learning** pour la classification des images médicales.  
- Un **chatbot RAG (Retrieval-Augmented Generation)** permettant aux utilisateurs de poser des questions sur les maladies détectées et d'obtenir des réponses basées sur une base de connaissances médicale.  
- Un **chatbot multilingue**, capable de comprendre et de répondre aux questions dans plusieurs langues.  
- Une **gestion intelligente de l'historique de conversation**, permettant de répondre de manière cohérente aux questions dépendantes du contexte précédent.

## Types de diagnostics  

### **CHEST :**  
#### **X-ray :**  
- **Normal**  
- **Tuberculosis**  
- **Pneumonia :**  
  - *Virus*  
  - *Bacteria*  
- **Covid**  
- **Fibrosis**  

#### **CT :**  
- **Normal**  
- **Infectieux :**  
  - *Normal*  
  - *Covid*  
  - *CAP*  
- **Cancer :**  
  - *Bénin*  
  - *Adénocarcinome*  
  - *Large Cell Carcinoma*  
  - *Squamous Cell Carcinoma*  

### **BRAIN :**  
- **Normal**  
- **Glioma**  
- **Meningioma**  
- **Pituitary**  

## Installation et Exécution  

```bash
# Clonez le dépôt
git clone https://github.com/Abdelmoughitelhassani/deep-medical-diagnosis.git
cd deep-medical-diagnosis

# Créez et activez un environnement virtuel
python -m venv venv
source venv/bin/activate  # Sur macOS/Linux
venv\Scripts\activate  # Sur Windows

# Installez les dépendances et exécutez l'application
pip install -r requirements.txt
python manage.py runserver
