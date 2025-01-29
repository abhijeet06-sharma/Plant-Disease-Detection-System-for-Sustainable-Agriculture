import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# Function to load and predict the plant disease
def model_prediction(test_image):
    model = tf.keras.models.load_model("C:\\Users\\Abhijeet Sharma\\Desktop\\Plant Disease Detection\\Plant Disease Detection System for Sustainable Agriculture\\trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # return index of max element


# Function to fetch disease details
def disease_info(disease_name):
    disease_details = {
        'Apple___Apple_scab': {
            'description': "Apple scab is a fungal disease caused by Venturia inaequalis, characterized by olive-green to black velvety spots on leaves and fruit.",
            'effect': "Reduces fruit quality, causes premature leaf drop, and weakens the tree over time.",
            'prevention': "Plant resistant varieties, ensure proper pruning for air circulation, and apply fungicides during the growing season."
        },
        'Apple___Black_rot': {
            'description': "Black rot is caused by the fungus Botryosphaeria obtusa, leading to dark, sunken lesions on fruit and leaf spotting.",
            'effect': "Causes fruit decay, leaf drop, and can damage branches, reducing yield.",
            'prevention': "Remove and destroy infected plant material, ensure proper sanitation, and apply protective fungicides."
        },
        'Apple___Cedar_apple_rust': {
            'description': "Cedar apple rust is a fungal disease caused by Gymnosporangium juniperi-virginianae, resulting in orange or rust-colored spots on leaves and fruit.",
            'effect': "Weakens the tree, reduces photosynthesis, and lowers fruit quality.",
            'prevention': "Remove nearby juniper hosts, apply fungicides, and plant resistant apple varieties."
        },
        'Apple___healthy': {
            'description': "Healthy apple plants are free from disease, with vibrant green leaves and blemish-free fruit.",
            'effect': "Optimal growth and maximum fruit yield.",
            'prevention': "Maintain proper orchard hygiene, regular monitoring, and timely application of preventive care."
        },
        'Blueberry___healthy': {
            'description': "Healthy blueberry plants exhibit strong growth, green leaves, and an abundant yield of berries.",
            'effect': "Ensures high-quality fruit and overall plant vigor.",
            'prevention': "Provide proper irrigation, fertilization, and pruning to maintain plant health."
        },
        'Cherry_(including_sour)___Powdery_mildew': {
            'description': "Powdery mildew is a fungal disease caused by Podosphaera spp., resulting in a white powdery coating on leaves, stems, and fruit.",
            'effect': "Reduces photosynthesis, weakens the plant, and affects fruit quality and yield.",
            'prevention': "Ensure good air circulation, apply sulfur-based fungicides, and remove infected plant parts."
        },
        'Cherry_(including_sour)___healthy': {
            'description': "Healthy cherry plants are characterized by lush green foliage and high-quality fruit production.",
            'effect': "Promotes optimal growth and maximum yield.",
            'prevention': "Practice regular pruning, proper watering, and timely fertilization."
        },
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
            'description': "Gray leaf spot is caused by the fungus Cercospora zeae-maydis, leading to rectangular lesions on leaves.",
            'effect': "Reduces photosynthesis, weakens the plant, and decreases crop yield.",
            'prevention': "Rotate crops, plant resistant varieties, and use fungicides if necessary."
        },
        'Corn_(maize)___Common_rust_': {
            'description': "Common rust is caused by Puccinia sorghi, resulting in red or brown pustules on leaves.",
            'effect': "Weakens the plant and reduces photosynthetic efficiency, leading to lower yields.",
            'prevention': "Plant resistant hybrids, practice crop rotation, and apply fungicides as needed."
        },
        'Corn_(maize)___Northern_Leaf_Blight': {
            'description': "Northern leaf blight is caused by Exserohilum turcicum, producing cigar-shaped lesions on leaves.",
            'effect': "Weakens the plant and reduces grain production.",
            'prevention': "Plant resistant varieties, use crop rotation, and apply fungicides during outbreaks."
        },
        'Corn_(maize)___healthy': {
            'description': "Healthy corn plants show strong, upright growth and disease-free leaves, resulting in high yields.",
            'effect': "Ensures optimal growth and grain production.",
            'prevention': "Provide proper irrigation, fertilization, and pest control."
        },
        'Grape___Black_rot': {
            'description': "Black rot is caused by the fungus Guignardia bidwellii, leading to brown spots on leaves and shriveled black fruit.",
            'effect': "Reduces fruit yield and quality, weakens the vine.",
            'prevention': "Prune infected parts, ensure good air circulation, and apply fungicides."
        },
        'Grape___Esca_(Black_Measles)': {
            'description': "Esca or black measles is a fungal disease causing discoloration of leaves and black spots on grapes.",
            'effect': "Reduces fruit quality, weakens the vine, and may lead to plant death.",
            'prevention': "Avoid overwatering, prune infected vines, and apply fungicides as needed."
        },
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
            'description': "Leaf blight is caused by Isariopsis spp., leading to dark lesions on leaves and premature leaf drop.",
            'effect': "Weakens the plant and reduces yield.",
            'prevention': "Remove infected leaves, ensure good ventilation, and apply fungicides."
        },
        'Grape___healthy': {
            'description': "Healthy grapevines are characterized by lush green foliage and high-quality fruit clusters.",
            'effect': "Ensures vigorous growth and maximum fruit production.",
            'prevention': "Maintain proper pruning, irrigation, and fertilization schedules."
        },
        'Orange___Haunglongbing_(Citrus_greening)': {
            'description': "Citrus greening, caused by Candidatus Liberibacter, results in yellowing of leaves, misshapen fruit, and tree decline.",
            'effect': "Drastically reduces fruit quality, tree health, and lifespan.",
            'prevention': "Control psyllid vectors, remove infected trees, and use certified disease-free saplings."
        },
        'Peach___Bacterial_spot': {
            'description': "Bacterial spot is caused by Xanthomonas campestris, leading to dark spots on leaves and fruit.",
            'effect': "Reduces fruit quality and may cause premature leaf drop.",
            'prevention': "Plant resistant varieties, avoid overhead irrigation, and apply copper-based bactericides."
        },
        'Peach___healthy': {
            'description': "Healthy peach trees produce high-quality fruit and exhibit disease-free foliage.",
            'effect': "Promotes optimal tree growth and yield.",
            'prevention': "Ensure proper pruning, fertilization, and irrigation."
        },
        'Pepper,_bell___Bacterial_spot': {
            'description': "Bacterial spot is caused by Xanthomonas campestris, leading to water-soaked lesions on leaves and fruit.",
            'effect': "Reduces fruit quality and weakens the plant.",
            'prevention': "Use resistant varieties, avoid overhead watering, and apply copper-based sprays."
        },
        'Pepper,_bell___healthy': {
            'description': "Healthy bell pepper plants exhibit strong growth and produce high-quality fruit.",
            'effect': "Ensures optimal yield and plant health.",
            'prevention': "Provide proper fertilization, watering, and pest management."
        },
        'Potato___Early_blight': {
            'description': "Early blight is caused by Alternaria solani, resulting in concentric rings on leaves and tuber rot.",
            'effect': "Reduces tuber quality and yield.",
            'prevention': "Practice crop rotation, remove infected debris, and apply fungicides."
        },
        'Potato___Late_blight': {
            'description': "Late blight is caused by Phytophthora infestans, leading to water-soaked lesions and rapid plant decay.",
            'effect': "Destroys leaves and tubers, causing severe yield loss.",
            'prevention': "Use resistant varieties, avoid overhead irrigation, and apply fungicides."
        },
        'Potato___healthy': {
            'description': "Healthy potato plants show vigorous growth and disease-free leaves, ensuring high tuber yield.",
            'effect': "Promotes optimal tuber production and plant health.",
            'prevention': "Maintain proper watering, fertilization, and pest management practices."
        },
        'Raspberry___healthy': {
            'description': "Healthy raspberry plants exhibit robust growth and produce high-quality fruit.",
            'effect': "Ensures maximum yield and plant vigor.",
            'prevention': "Provide proper pruning, irrigation, and pest control."
        },
        'Soybean___healthy': {
            'description': "Healthy soybean plants are characterized by green foliage and optimal pod production.",
            'effect': "Promotes maximum yield and plant health.",
            'prevention': "Maintain proper fertilization, pest control, and crop rotation."
        },
        'Squash___Powdery_mildew': {
            'description': "Powdery mildew is caused by Erysiphe cichoracearum, resulting in white powdery growth on leaves and stems.",
            'effect': "Reduces photosynthesis and weakens the plant, affecting yield.",
            'prevention': "Ensure good air circulation, apply fungicides, and avoid overhead watering."
        },
        'Strawberry___Leaf_scorch': {
            'description': "Leaf scorch is caused by Diplocarpon earlianum, leading to dark spots on leaves and eventual leaf death.",
            'effect': "Weakens the plant and reduces fruit production.",
            'prevention': "Remove infected leaves, apply fungicides, and maintain proper spacing for air circulation."
        },
        'Strawberry___healthy': {
            'description': "Healthy strawberry plants exhibit lush green growth and produce high-quality fruit.",
            'effect': "Ensures maximum fruit yield and plant health.",
            'prevention': "Provide proper fertilization, watering, and pest control."
        },
        'Tomato___Bacterial_spot': {
            'description': "Bacterial spot is caused by Xanthomonas campestris, leading to dark, water-soaked lesions on leaves and fruit.",
            'effect': "Reduces fruit quality and weakens the plant.",
            'prevention': "Use resistant varieties, avoid overhead watering, and apply copper-based sprays."
        },
        'Tomato___Early_blight': {
            'description': "Early blight is caused by Alternaria solani, resulting in target-like spots on leaves and fruit rot.",
            'effect': "Reduces fruit yield and plant vigor.",
            'prevention': "Practice crop rotation, remove infected debris, and apply fungicides."
        },
        'Tomato___Late_blight': {
            'description': "Late blight is caused by Phytophthora infestans, leading to dark lesions and rapid plant decay.",
            'effect': "Destroys leaves and fruit, causing severe yield loss.",
            'prevention': "Use resistant varieties, avoid overhead irrigation, and apply fungicides."
        },
        'Tomato___Leaf_Mold': {
            'description': "Leaf mold is caused by Passalora fulva, resulting in yellow spots on leaves and mold growth on the underside.",
            'effect': "Weakens the plant and reduces fruit production.",
            'prevention': "Ensure good air circulation, apply fungicides, and avoid overhead watering."
        },
        'Tomato___Septoria_leaf_spot': {
            'description': "Septoria leaf spot is caused by Septoria lycopersici, resulting in small circular spots on leaves.",
            'effect': "Weakens the plant and reduces yield.",
            'prevention': "Remove infected leaves, practice crop rotation, and apply fungicides."
        },
        'Tomato___Spider_mites Two-spotted_spider_mite': {
            'description': "Spider mites are tiny pests that cause yellowing and speckling of leaves, leading to webbing on plants.",
            'effect': "Reduces photosynthesis and weakens the plant, affecting yield.",
            'prevention': "Use insecticidal soap, encourage natural predators, and maintain proper irrigation."
        },
        'Tomato___Target_Spot': {
            'description': "Target spot is caused by Corynespora cassiicola, resulting in brown lesions with concentric rings on leaves and fruit.",
            'effect': "Reduces fruit quality and plant vigor.",
            'prevention': "Ensure good air circulation, remove infected debris, and apply fungicides."
        },
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
            'description': "This viral disease is transmitted by whiteflies, causing yellowing and curling of leaves and stunted growth.",
            'effect': "Drastically reduces fruit yield and plant health.",
            'prevention': "Control whiteflies, use resistant varieties, and remove infected plants."
        },
        'Tomato___Tomato_mosaic_virus': {
            'description': "Tomato mosaic virus causes mottling, yellowing, and distortion of leaves, reducing plant vigor.",
            'effect': "Reduces fruit quality and yield.",
            'prevention': "Use disease-free seeds, practice good sanitation, and remove infected plants."
        },
        'Tomato___healthy': {
            'description': "Healthy tomato plants exhibit strong growth, green foliage, and high-quality fruit production.",
            'effect': "Ensures optimal yield and plant health.",
            'prevention': "Provide proper fertilization, watering, and pest control."
        }
    }
    return disease_details.get(disease_name, {
        "description": "No details available for this disease.",
        "effects": "N/A",
        "prevention": "N/A"
    })


# Function to fetch medicine and shop details
def disease_medicine_info(disease_name):
    medicine_details = {
        "Apple___Apple_scab": {
            "medicine": "Captan Fungicide",
            "shops": ["Green Valley Agro Store", "Farmers' Supply Co.", "AgriLife Distributors"]
        },
        "Apple___Black_rot": {
            "medicine": "Thiophanate-Methyl",
            "shops": ["Orchard Care Supplies", "Nature's Best Agri Products", "Growers' Choice"]
        },
        "Apple___Cedar_apple_rust": {
            "medicine": "Mancozeb",
            "shops": ["Apple Orchard Supply", "AgriPro Store", "Crop Solutions"]
        },
        "Apple___healthy": {
            "medicine": "None",
            "shops": []
        },
        "Blueberry___healthy": {
            "medicine": "None",
            "shops": []
        },
        "Cherry_(including_sour)___Powdery_mildew": {
            "medicine": "Myclobutanil",
            "shops": ["Cherry Care Center", "Farmers' Supplies", "AgriTech Distributors"]
        },
        "Cherry_(including_sour)___healthy": {
            "medicine": "None",
            "shops": []
        },
        "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot": {
            "medicine": "Azoxystrobin",
            "shops": ["Corn Care Store", "Farm Solutions", "AgriMax Distributors"]
        },
        "Corn_(maize)___Common_rust_": {
            "medicine": "Tebuconazole",
            "shops": ["Corn Grower Supplies", "AgriCure", "Harvest Solutions"]
        },
        "Corn_(maize)___Northern_Leaf_Blight": {
            "medicine": "Propiconazole",
            "shops": ["AgriCare Supplies", "Crop Health Center", "Farmers' Choice"]
        },
        "Corn_(maize)___healthy": {
            "medicine": "None",
            "shops": []
        },
        "Grape___Black_rot": {
            "medicine": "Kocide",
            "shops": ["Grape Growers Shop", "Vineyard Solutions", "AgriMax Distributors"]
        },
        "Grape___Esca_(Black_Measles)": {
            "medicine": "Tebuconazole",
            "shops": ["Grape Care Center", "Vineyard Products", "AgriPro Distributors"]
        },
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
            "medicine": "Thiophanate-Methyl",
            "shops": ["Grape Plant Care", "Farmers' Agri Store", "Vineyard Solutions"]
        },
        "Grape___healthy": {
            "medicine": "None",
            "shops": []
        },
        "Orange___Haunglongbing_(Citrus_greening)": {
            "medicine": "Oxytetracycline",
            "shops": ["Citrus Solutions", "Orange Care Center", "Farmers' Supplies"]
        },
        "Peach___Bacterial_spot": {
            "medicine": "Copper Hydroxide",
            "shops": ["Peach Plant Care", "AgriCo Distributors", "Farm Solutions"]
        },
        "Peach___healthy": {
            "medicine": "None",
            "shops": []
        },
        "Pepper,_bell___Bacterial_spot": {
            "medicine": "Copper Fungicide",
            "shops": ["Pepper Plant Solutions", "AgriCare Distributors", "Farm Supply Co."]
        },
        "Pepper,_bell___healthy": {
            "medicine": "None",
            "shops": []
        },
        "Potato___Early_blight": {
            "medicine": "Chlorothalonil",
            "shops": ["Potato Health Store", "Farm Fresh Co.", "AgriTech Distributors"]
        },
        "Potato___Late_blight": {
            "medicine": "Mancozeb",
            "shops": ["Potato Care Center", "AgroPlus", "Farm Fresh Co."]
        },
        "Potato___healthy": {
            "medicine": "None",
            "shops": []
        },
        "Raspberry___healthy": {
            "medicine": "None",
            "shops": []
        },
        "Soybean___healthy": {
            "medicine": "None",
            "shops": []
        },
        "Squash___Powdery_mildew": {
            "medicine": "Myclobutanil",
            "shops": ["Squash Care Center", "Farmers' Choice", "AgriSmart Solutions"]
        },
        "Strawberry___Leaf_scorch": {
            "medicine": "Copper Sulfate",
            "shops": ["Strawberry Care Center", "Farmers' Supply Hub", "AgriPro Distributors"]
        },
        "Strawberry___healthy": {
            "medicine": "None",
            "shops": []
        },
        "Tomato___Bacterial_spot": {
            "medicine": "Copper Fungicide",
            "shops": ["Tomato Plant Care", "AgriTech Distributors", "Farm Fresh Co."]
        },
        "Tomato___Early_blight": {
            "medicine": "Chlorothalonil",
            "shops": ["Tomato Care Center", "Healthy Harvest Agro Store", "AgriChem Solutions"]
        },
        "Tomato___Late_blight": {
            "medicine": "Mancozeb",
            "shops": ["Tomato Health Store", "AgriPro Distributors", "Farm Solutions"]
        },
        "Tomato___Leaf_Mold": {
            "medicine": "Fosetyl-Aluminum",
            "shops": ["Tomato Plant Care", "AgriMax Solutions", "Farmers' Choice"]
        },
        "Tomato___Septoria_leaf_spot": {
            "medicine": "Tetraconazole",
            "shops": ["Tomato Care Supplies", "Healthy Harvest Store", "AgriLife Distributors"]
        },
        "Tomato___Spider_mites_Two-spotted_spider_mite": {
            "medicine": "Acaricide",
            "shops": ["Tomato Health Store", "AgriPro Distributors", "Farm Fresh Co."]
        },
        "Tomato___Target_Spot": {
            "medicine": "Pyraclostrobin",
            "shops": ["Tomato Care Shop", "AgriCure", "Farmers' Supply Co."]
        },
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
            "medicine": "Imidacloprid",
            "shops": ["Tomato Health Solutions", "AgriMax Co.", "Farm Solutions"]
        },
        "Tomato___Tomato_mosaic_virus": {
            "medicine": "None",
            "shops": []
        },
        "Tomato___healthy": {
            "medicine": "None",
            "shops": []
        }
    }
    return medicine_details.get(disease_name, {
        "medicine": "No medicine available.",
        "shops": ["N/A"]
    })


# Sidebar
st.sidebar.title("Plant Disease Detection System for Sustainable Agriculture")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# Import Image
img = Image.open(
    r"C:\\Users\\Abhijeet Sharma\\Desktop\\Plant Disease Detection\\Plant Disease Detection System for Sustainable Agriculture\\Diseases.png")

# Display image using Streamlit
st.image(img)

# Main Page
if app_mode == "HOME":
    st.markdown("<h1 style='text-align: center;'>Plant Disease Detection System for Sustainable Agriculture</h1>",
                unsafe_allow_html=True)

elif app_mode == "DISEASE RECOGNITION":
    st.header("Plant Disease Detection System for Sustainable Agriculture")
    test_image = st.file_uploader("Choose an Image:")
    if st.button("Show Image"):
        if test_image is not None:
            st.image(test_image, use_column_width=True)
        else:
            st.warning("Please upload an image first.")

    # Predict button
    if st.button("Predict"):
        if test_image is not None:
            st.snow()
            st.write("Our Prediction:")
            result_index = model_prediction(test_image)

            # Reading Labels
            class_names = [
                'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                'Tomato___healthy'
            ]
            predicted_disease = class_names[result_index]
            st.success(f"Model is predicting it's {predicted_disease}")

            # Display disease details
            details = disease_info(predicted_disease)
            st.markdown(f"### Disease Details")
            st.write(f"**Description:** {details['description']}")
            st.write(f"**Effects:** {details['effect']}")
            st.write(f"**Prevention:** {details['prevention']}")

            # Display medicine details
            medicine_info = disease_medicine_info(predicted_disease)
            st.markdown(f"### Recommended Medicine")
            st.write(f"**Medicine Name:** {medicine_info['medicine']}")
            st.write(f"**Available Shops:** {', '.join(medicine_info['shops'])}")
        else:
            st.error("Please upload an image before prediction.")
st.markdown(
    """
    <div style="text-align: center; margin-top: 50px; padding-top: 20px; border-top: 2px solid #ccc;">
        <h3>Consult with Us</h3>
        <p>If you have any questions or need further assistance, feel free to contact us:</p>
        <p><strong>Contact Number:</strong> +91 123 456 7890</p>
    </div>
    """,
    unsafe_allow_html=True
)