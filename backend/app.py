from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image

#Create Flask app

app = Flask(__name__)
CORS(app)

#Home route

@app.route('/')
def home():
    return "Flask API Running Successfully!"

#Load trained AI model

model = tf.keras.models.load_model("crop_disease_model.h5")

#Disease classes

classes = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]
# Disease solutions in multiple languages

solutions = {
    "Tomato_Bacterial_spot": {
        "en": "Disease Name: Tomato Bacterial Spot\n\nRecommended Pesticides:\n• Copper Oxychloride\n• Copper Hydroxide\n\nPrevention Methods:\n• Remove infected leaves\n• Avoid overhead irrigation\n• Use disease-free seeds",
        "te": "వ్యాధి పేరు: టమాటో బ్యాక్టీరియల్ స్పాట్\n\nసిఫార్సు చేసిన మందులు:\n• కాపర్ ఆక్సీక్లోరైడ్\n• కాపర్ హైడ్రాక్సైడ్\n\nనివారణ చర్యలు:\n• సోకిన ఆకులను తొలగించండి\n• పై నుండి నీరు పోయవద్దు\n• ఆరోగ్యకరమైన విత్తనాలు వాడండి",
        "hi": "रोग का नाम: टमाटर बैक्टीरियल स्पॉट\n\nअनुशंसित दवाएं:\n• कॉपर ऑक्सीक्लोराइड\n• कॉपर हाइड्रॉक्साइड\n\nरोकथाम के उपाय:\n• संक्रमित पत्तियां हटाएं\n• ऊपर से सिंचाई न करें\n• रोगमुक्त बीजों का उपयोग करें"
    },

    "Tomato_Early_blight": {
        "en": "Disease Name: Tomato Early Blight\n\nRecommended Pesticides:\n• Mancozeb\n• Chlorothalonil\n• Azoxystrobin\n\nPrevention Methods:\n• Remove infected leaves\n• Practice crop rotation\n• Maintain field sanitation",
        "te": "వ్యాధి పేరు: టమాటో ఎర్లీ బ్లైట్\n\nసిఫార్సు చేసిన మందులు:\n• మాంకోజెబ్\n• క్లోరోథాలోనిల్\n• అజోక్సిస్ట్రోబిన్\n\nనివారణ చర్యలు:\n• సోకిన ఆకులను తొలగించండి\n• పంట మార్పిడి పాటించండి\n• పొలాన్ని శుభ్రంగా ఉంచండి",
        "hi": "रोग का नाम: टमाटर अर्ली ब्लाइट\n\nअनुशंसित दवाएं:\n• मैनकोज़ेब\n• क्लोरोथालोनिल\n• एजोक्सीस्ट्रोबिन\n\nरोकथाम के उपाय:\n• संक्रमित पत्तियां हटाएं\n• फसल चक्र अपनाएं\n• खेत को साफ रखें"
    },

    "Tomato_healthy": {
        "en": "Disease Name: Healthy Tomato Plant\n\nRecommended Action:\n• No pesticide required\n\nPrevention Methods:\n• Continue proper irrigation\n• Apply balanced fertilizers\n• Monitor crop regularly",
        "te": "వ్యాధి పేరు: ఆరోగ్యకరమైన టమాటో మొక్క\n\nసిఫార్సు చేసిన చర్య:\n• మందులు అవసరం లేదు\n\nనివారణ చర్యలు:\n• సరైన నీటి పారుదల కొనసాగించండి\n• సమతుల్య ఎరువులు వాడండి\n• పంటను క్రమం తప్పకుండా పరిశీలించండి",
        "hi": "रोग का नाम: स्वस्थ टमाटर पौधा\n\nअनुशंसित कार्य:\n• किसी दवा की आवश्यकता नहीं\n\nरोकथाम के उपाय:\n• उचित सिंचाई जारी रखें\n• संतुलित उर्वरक दें\n• नियमित निगरानी करें"
    },

    "Tomato_Late_blight": {
        "en": "Disease Name: Tomato Late Blight\n\nRecommended Pesticides:\n• Metalaxyl + Mancozeb\n• Cymoxanil + Mancozeb\n• Chlorothalonil\n\nPrevention Methods:\n• Avoid excessive moisture\n• Improve air circulation\n• Remove infected plant parts",
        "te": "వ్యాధి పేరు: టమాటో లేట్ బ్లైట్\n\nసిఫార్సు చేసిన మందులు:\n• మెటాలాక్సిల్ + మాంకోజెబ్\n• సైమోక్సానిల్ + మాంకోజెబ్\n• క్లోరోథాలోనిల్\n\nనివారణ చర్యలు:\n• అధిక తేమను నివారించండి\n• గాలి ప్రసరణను మెరుగుపరచండి\n• సోకిన భాగాలను తొలగించండి",
        "hi": "रोग का नाम: टमाटर लेट ब्लाइट\n\nअनुशंसित दवाएं:\n• मेटालैक्सिल + मैनकोज़ेब\n• साइमोक्सानिल + मैनकोज़ेब\n• क्लोरोथालोनिल\n\nरोकथाम के उपाय:\n• अधिक नमी से बचें\n• वायु संचार बढ़ाएं\n• संक्रमित भागों को हटाएं"
    },

    "Tomato_Leaf_Mold": {
        "en": "Disease Name: Tomato Leaf Mold\n\nRecommended Pesticides:\n• Chlorothalonil\n• Mancozeb\n\nPrevention Methods:\n• Improve ventilation\n• Reduce humidity\n• Remove infected leaves",
        "te": "వ్యాధి పేరు: టమాటో లీఫ్ మోల్డ్\n\nసిఫార్సు చేసిన మందులు:\n• క్లోరోథాలోనిల్\n• మాంకోజెబ్\n\nనివారణ చర్యలు:\n• గాలి ప్రసరణ మెరుగుపరచండి\n• తేమ తగ్గించండి\n• సోకిన ఆకులను తొలగించండి",
        "hi": "रोग का नाम: टमाटर लीफ मोल्ड\n\nअनुशंसित दवाएं:\n• क्लोरोथालोनिल\n• मैनकोज़ेब\n\nरोकथाम के उपाय:\n• वेंटिलेशन बढ़ाएं\n• नमी कम करें\n• संक्रमित पत्तियां हटाएं"
    },

    "Tomato_Septoria_leaf_spot": {
        "en": "Disease Name: Tomato Septoria Leaf Spot\n\nRecommended Pesticides:\n• Chlorothalonil\n• Mancozeb\n• Copper Oxychloride\n\nPrevention Methods:\n• Remove infected leaves\n• Avoid overhead watering\n• Maintain field hygiene",
        "te": "వ్యాధి పేరు: టమాటో సెప్టోరియా లీఫ్ స్పాట్\n\nసిఫార్సు చేసిన మందులు:\n• క్లోరోథాలోనిల్\n• మాంకోజెబ్\n• కాపర్ ఆక్సీక్లోరైడ్\n\nనివారణ చర్యలు:\n• సోకిన ఆకులను తొలగించండి\n• పై నుండి నీరు పోయవద్దు\n• పొలాన్ని శుభ్రంగా ఉంచండి",
        "hi": "रोग का नाम: टमाटर सेप्टोरिया लीफ स्पॉट\n\nअनुशंसित दवाएं:\n• क्लोरोथालोनिल\n• मैनकोज़ेब\n• कॉपर ऑक्सीक्लोराइड\n\nरोकथाम के उपाय:\n• संक्रमित पत्तियां हटाएं\n• ऊपर से सिंचाई न करें\n• खेत की स्वच्छता बनाए रखें"
    },

    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "en": "Disease Name: Two-Spotted Spider Mite\n\nRecommended Pesticides:\n• Abamectin\n• Spiromesifen\n• Fenazaquin\n\nPrevention Methods:\n• Maintain adequate humidity\n• Remove heavily infested leaves\n• Monitor plants regularly",
        "te": "వ్యాధి పేరు: టూ-స్పాటెడ్ స్పైడర్ మైట్\n\nసిఫార్సు చేసిన మందులు:\n• అబామెక్టిన్\n• స్పైరోమెసిఫెన్\n• ఫెనాజాక్విన్\n\nనివారణ చర్యలు:\n• తగిన తేమను నిర్వహించండి\n• తీవ్రంగా సోకిన ఆకులను తొలగించండి\n• మొక్కలను క్రమం తప్పకుండా పరిశీలించండి",
        "hi": "रोग का नाम: टू-स्पॉटेड स्पाइडर माइट\n\nअनुशंसित दवाएं:\n• एबामेक्टिन\n• स्पाइरोमेसिफेन\n• फेनाज़ाक्विन\n\nरोकथाम के उपाय:\n• पर्याप्त नमी बनाए रखें\n• अधिक संक्रमित पत्तियां हटाएं\n• नियमित निगरानी करें"
    },

    "Tomato__Target_Spot": {
        "en": "Disease Name: Tomato Target Spot\n\nRecommended Pesticides:\n• Azoxystrobin\n• Chlorothalonil\n• Mancozeb\n\nPrevention Methods:\n• Remove infected foliage\n• Improve air circulation\n• Avoid excess moisture",
        "te": "వ్యాధి పేరు: టమాటో టార్గెట్ స్పాట్\n\nసిఫార్సు చేసిన మందులు:\n• అజోక్సిస్ట్రోబిన్\n• క్లోరోథాలోనిల్\n• మాంకోజెబ్\n\nనివారణ చర్యలు:\n• సోకిన ఆకులను తొలగించండి\n• గాలి ప్రసరణను మెరుగుపరచండి\n• అధిక తేమను నివారించండి",
        "hi": "रोग का नाम: टमाटर टार्गेट स्पॉट\n\nअनुशंसित दवाएं:\n• एजोक्सीस्ट्रोबिन\n• क्लोरोथालोनिल\n• मैनकोज़ेब\n\nरोकथाम के उपाय:\n• संक्रमित पत्तियां हटाएं\n• वायु संचार बढ़ाएं\n• अधिक नमी से बचें"
    },

    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "en": "Disease Name: Tomato Yellow Leaf Curl Virus\n\nRecommended Pesticides:\n• Imidacloprid\n• Thiamethoxam\n\nPrevention Methods:\n• Remove infected plants\n• Control whiteflies\n• Use resistant varieties",
        "te": "వ్యాధి పేరు: టమాటో యెల్లో లీఫ్ కర్ల్ వైరస్\n\nసిఫార్సు చేసిన మందులు:\n• ఇమిడాక్లోప్రిడ్\n• థియామెథోక్సామ్\n\nనివారణ చర్యలు:\n• సోకిన మొక్కలను తొలగించండి\n• వైట్ ఫ్లైలను నియంత్రించండి\n• నిరోధక రకాలను ఉపయోగించండి",
        "hi": "रोग का नाम: टमाटर येलो लीफ कर्ल वायरस\n\nअनुशंसित दवाएं:\n• इमिडाक्लोप्रिड\n• थायामेथोक्साम\n\nरोकथाम के उपाय:\n• संक्रमित पौधों को हटाएं\n• व्हाइटफ्लाई को नियंत्रित करें\n• प्रतिरोधी किस्मों का उपयोग करें"
    },

    "Tomato__Tomato_mosaic_virus": {
        "en": "Disease Name: Tomato Mosaic Virus\n\nRecommended Pesticides:\n• No direct pesticide available\n• Imidacloprid (for aphid control)\n\nPrevention Methods:\n• Remove infected plants immediately\n• Disinfect tools and hands\n• Use resistant varieties",
        "te": "వ్యాధి పేరు: టమాటో మొజాయిక్ వైరస్\n\nసిఫార్సు చేసిన మందులు:\n• వైరస్‌కు ప్రత్యక్ష మందు లేదు\n• ఇమిడాక్లోప్రిడ్ (ఆఫిడ్స్ నియంత్రణకు)\n\nనివారణ చర్యలు:\n• సోకిన మొక్కలను వెంటనే తొలగించండి\n• పనిముట్లను శుభ్రపరచండి\n• నిరోధక రకాలను ఉపయోగించండి",
        "hi": "रोग का नाम: टमाटर मोज़ेक वायरस\n\nअनुशंसित दवाएं:\n• वायरस के लिए कोई प्रत्यक्ष दवा उपलब्ध नहीं है\n• इमिडाक्लोप्रिड (एफिड नियंत्रण हेतु)\n\nरोकथाम के उपाय:\n• संक्रमित पौधों को तुरंत हटाएं\n• उपकरणों को साफ रखें\n• प्रतिरोधी किस्मों का उपयोग करें"
    },

    "Pepper__bell___Bacterial_spot": {
        "en": "Disease Name: Bell Pepper Bacterial Spot\n\nRecommended Pesticides:\n• Copper Oxychloride\n• Copper Hydroxide\n\nPrevention Methods:\n• Remove infected leaves\n• Avoid overhead watering\n• Use disease-free seeds",
        "te": "వ్యాధి పేరు: బెల్ పెప్పర్ బ్యాక్టీరియల్ స్పాట్\n\nసిఫార్సు చేసిన మందులు:\n• కాపర్ ఆక్సీక్లోరైడ్\n• కాపర్ హైడ్రాక్సైడ్\n\nనివారణ చర్యలు:\n• సోకిన ఆకులను తొలగించండి\n• పై నుండి నీరు పోయవద్దు\n• ఆరోగ్యకరమైన విత్తనాలు వాడండి",
        "hi": "रोग का नाम: बेल पेपर बैक्टीरियल स्पॉट\n\nअनुशंसित दवाएं:\n• कॉपर ऑक्सीक्लोराइड\n• कॉपर हाइड्रॉक्साइड\n\nरोकथाम के उपाय:\n• संक्रमित पत्तियां हटाएं\n• ऊपर से सिंचाई न करें\n• रोगमुक्त बीजों का उपयोग करें"
    },

    "Pepper__bell___healthy": {
         "en": "Disease Name: Healthy Bell Pepper Plant\n\nRecommended Action:\n• No pesticide required\n\nPrevention Methods:\n• Maintain proper irrigation\n• Apply balanced nutrients\n• Monitor crop regularly",
        "te": "వ్యాధి పేరు: ఆరోగ్యకరమైన బెల్ పెప్పర్ మొక్క\n\nసిఫార్సు చేసిన చర్య:\n• మందులు అవసరం లేదు\n\nనివారణ చర్యలు:\n• సరైన నీటి పారుదల కొనసాగించండి\n• సమతుల్య పోషకాలు అందించండి\n• పంటను క్రమం తప్పకుండా పరిశీలించండి",
        "hi": "रोग का नाम: स्वस्थ बेल पेपर पौधा\n\nअनुशंसित कार्य:\n• किसी दवा की आवश्यकता नहीं\n\nरोकथाम के उपाय:\n• उचित सिंचाई बनाए रखें\n• संतुलित पोषक तत्व दें\n• नियमित निगरानी करें"
    },

    "Potato___Late_blight": {
        "en": "Disease Name: Potato Late Blight\n\nRecommended Pesticides:\n• Metalaxyl + Mancozeb\n• Cymoxanil + Mancozeb\n• Chlorothalonil\n\nPrevention Methods:\n• Avoid excessive moisture\n• Improve drainage\n• Remove infected plants",
        "te": "వ్యాధి పేరు: బంగాళాదుంప లేట్ బ్లైట్\n\nసిఫార్సు చేసిన మందులు:\n• మెటాలాక్సిల్ + మాంకోజెబ్\n• సైమోక్సానిల్ + మాంకోజెబ్\n• క్లోరోథాలోనిల్\n\nనివారణ చర్యలు:\n• అధిక తేమను నివారించండి\n• నీటి పారుదల మెరుగుపరచండి\n• సోకిన మొక్కలను తొలగించండి",
        "hi": "रोग का नाम: आलू लेट ब्लाइट\n\nअनुशंसित दवाएं:\n• मेटालैक्सिल + मैनकोज़ेब\n• साइमोक्सानिल + मैनकोज़ेब\n• क्लोरोथालोनिल\n\nरोकथाम के उपाय:\n• अधिक नमी से बचें\n• जल निकासी में सुधार करें\n• संक्रमित पौधों को हटाएं"
    },

    "Potato___healthy": {
        "en": "Disease Name: Healthy Potato Plant\n\nRecommended Action:\n• No pesticide required\n\nPrevention Methods:\n• Maintain proper irrigation\n• Apply balanced fertilizers\n• Monitor crop health regularly",
        "te": "వ్యాధి పేరు: ఆరోగ్యకరమైన బంగాళాదుంప మొక్క\n\nసిఫార్సు చేసిన చర్య:\n• మందులు అవసరం లేదు\n\nనివారణ చర్యలు:\n• సరైన నీటి పారుదల కొనసాగించండి\n• సమతుల్య ఎరువులు వాడండి\n• పంట ఆరోగ్యాన్ని క్రమం తప్పకుండా పరిశీలించండి",
        "hi": "रोग का नाम: स्वस्थ आलू पौधा\n\nअनुशंसित कार्य:\n• किसी दवा की आवश्यकता नहीं\n\nरोकथाम के उपाय:\n• उचित सिंचाई बनाए रखें\n• संतुलित उर्वरक दें\n• नियमित रूप से पौधे की निगरानी करें"
    },

    "Potato___Early_blight": {
        "en": "Disease Name: Potato Early Blight\n\nRecommended Pesticides:\n• Mancozeb\n• Chlorothalonil\n• Azoxystrobin\n\nPrevention Methods:\n• Remove infected leaves\n• Practice crop rotation\n• Maintain field sanitation",
        "te": "వ్యాధి పేరు: బంగాళాదుంప ఎర్లీ బ్లైట్\n\nసిఫార్సు చేసిన మందులు:\n• మాంకోజెబ్\n• క్లోరోథాలోనిల్\n• అజోక్సిస్ట్రోబిన్\n\nనివారణ చర్యలు:\n• సోకిన ఆకులను తొలగించండి\n• పంట మార్పిడి పాటించండి\n• పొలాన్ని శుభ్రంగా ఉంచండి",
        "hi": "रोग का नाम: आलू अर्ली ब्लाइट\n\nअनुशंसित दवाएं:\n• मैनकोज़ेब\n• क्लोरोथालोनिल\n• एजोक्सीस्ट्रोबिन\n\nरोकथाम के उपाय:\n• संक्रमित पत्तियां हटाएं\n• फसल चक्र अपनाएं\n• खेत को साफ रखें"
    }
}

#Prediction API

@app.route('/predict', methods=['POST'])
def predict():
    print("PREDICT API CALLED")

    file = request.files['image']

    image = Image.open(file).convert('RGB')

    image = image.resize((128, 128))

    image = np.array(image) / 255.0

    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)

    predicted_index = np.argmax(prediction)
    confidence = float(np.max(prediction)) * 100

    predicted_class = classes[predicted_index]

    print("Predicted Index:", predicted_index)
    print("Predicted Class:", predicted_class)
    print("Confidence:", round(confidence, 2), "%")

    predicted_class = classes[predicted_index]

    language = request.form.get("language", "en")

    return jsonify({
      "disease": predicted_class,
      "solution": solutions[predicted_class][language]
})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)