from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .utils import get_llm_response
from .utils_models import predict_single, predict_image
from django.core.files.storage import FileSystemStorage
import tensorflow as tf
from tensorflow.keras.models import load_model


@csrf_exempt
def chatbot_response(request):
    """
    Vue pour gérer les requêtes du chatbot.
    """
    if request.method == 'POST':
        data = json.loads(request.body)
        user_input = data.get('message', '')
        if user_input:
            try:
                response = get_llm_response(user_input)
                return JsonResponse({'response': response})
            except Exception as e:
                return JsonResponse({'error': str(e)}, status=500)
        return JsonResponse({'error': 'Invalid input'}, status=400)
    return JsonResponse({'error': 'Invalid request'}, status=400)




def chatbot_home(request):
    """
    Vue pour afficher la page du chatbot.
    """
    return render(request, 'app/chatbot_home.html') 





def predict_image_Xray(request):

    class_name = ['COVID', 'Fibrosis', 'Normal', 'PNEUMONIA', 'Tuberculosis']

    # Charger le modèle .keras
    model = load_model("./models/chest_xray_model_5_classes_v2.keras")


    context = predict_image(request, model, class_name)
    while context['confidence']<80:
        context = predict_image(request, model, class_name)

        
    if context['predicted_class']=='PNEUMONIA':
        class_name = ['bacterie', 'virus']
        model = load_model("./models/chest_pneumonia_TYPES_model_finetuned.keras")
        context_2 = predict_image(request, model, class_name)
        while context_2['confidence']<80:
            context_2 = predict_image(request, model, class_name)
        final_class = {'file_path_name':context['file_path_name'],'predicted_class':context['predicted_class']+ " avec une grande possibilité d'être de type "+context_2['predicted_class']}

    else:
        final_class = {'file_path_name':context['file_path_name'],'predicted_class':context['predicted_class']}

    

    return render(request, 'app/chest_Xray.html',context=final_class)



def predict_image_CT(request):

    class_name = ['Cancer', 'Infectieux', 'Normal']

    # Charger le modèle .keras
    model = load_model("./models/CT_model_1_v2.keras")


    context = predict_image(request, model, class_name)
    while context['confidence']<80:
        context = predict_image(request, model, class_name)

    if context['predicted_class']=='Cancer':
        class_name = ['Bengin', 'adenocarcinoma', 'large_cell_carcinoma', 'squamous_cell_carcinoma']
        model = load_model("./models/CT_model_3_v2.keras")
        context_3 = predict_image(request, model, class_name)
        while context_3['confidence']<80:
            context_3 = predict_image(request, model, class_name)
        final_class = {'file_path_name':context['file_path_name'],'predicted_class':context['predicted_class']+ " avec une grande possibilité d'être de type "+context_3['predicted_class']}

        #context['predicted_class']+="_"+context_3['predicted_class']
    elif context['predicted_class']=='Infectieux':
        class_name = ['CAP', 'COVID', 'Normal']
        model = load_model("./models/CT_model_2_v2.keras")
        context_2 = predict_image(request, model, class_name)
        while context_2['confidence']<80:
            context_2 = predict_image(request, model, class_name)
        

        final_class = {'file_path_name':context['file_path_name'],'predicted_class':context['predicted_class']+ "avec une grand possibiliter d'etre de type "+context_2['predicted_class']}
        
        if context_2['predicted_class']=='Normal':
            final_class = {'file_path_name':context['file_path_name'],'predicted_class':context['predicted_class']}
        #context['predicted_class']+="_"+context_2['predicted_class']
    else:
        final_class = {'file_path_name':context['file_path_name'],'predicted_class':context['predicted_class']}
    

    return render(request, 'app/chest_CT.html',context=final_class)



def predict_image_Brain(request):

    class_name = ['Glioma', 'Meningioma', 'Normal', 'Pituitary']

    # Charger le modèle .keras
    model = load_model("./models/Brain_model_v2.keras")


    context = predict_image(request, model, class_name)
    while context['confidence']<80:
        context = predict_image(request, model, class_name)
    return render(request, 'app/brain_MRI.html',context=context)








def home(request):
    return render(request, 'app/home.html')

def chest_Xray(request):
    
    return render(request, 'app/chest_Xray.html')


def chest_CT(request):
    
    return render(request, 'app/chest_CT.html')


def brain_MRI(request):
    
    return render(request, 'app/brain_MRI.html')
