from typing import Dict, Any, List, Optional
import os
import json

# The 29 Standardized Clinical Health Assessment Questions
# Mapped to feature keys, multilingual phrasings (English, Hindi, Bengali),
# conversational acknowledgments, and relevant disease domains.
CLINICAL_QUESTIONS_CATALOG: List[Dict[str, Any]] = [
    {
        "id": "q1_age",
        "index": 1,
        "feature_keys": ["age", "Age"],
        "category": "demographics",
        "questions": {
            "en": "What is your age?",
            "hi": "आपकी उम्र क्या है?",
            "bn": "আপনার বয়স কত?"
        },
        "acknowledgments": {
            "en": "Got it. I've recorded your age as {value}.",
            "hi": "ठीक है, मैंने आपकी उम्र {value} दर्ज कर ली है।",
            "bn": "বুঝেছি, আমি আপনার বয়স {value} রেকর্ড করেছি।"
        },
        "diseases": ["all", "diabetes", "heart_disease", "hypertension", "kidney_disease", "liver_disease", "stroke", "breast_cancer", "parkinsons"]
    },
    {
        "id": "q2_sex",
        "index": 2,
        "feature_keys": ["Sex", "sex", "gender", "Gender"],
        "category": "demographics",
        "questions": {
            "en": "What is your biological sex?",
            "hi": "आपका जैविक लिंग (Biological Sex) क्या है?",
            "bn": "আপনার জৈবিক লিঙ্গ কি?"
        },
        "acknowledgments": {
            "en": "Thank you, recorded as {value}.",
            "hi": "धन्यवाद, {value} दर्ज कर लिया गया है।",
            "bn": "ধন্যবাদ, {value} হিসেবে রেকর্ড করা হয়েছে।"
        },
        "diseases": ["all", "diabetes", "heart_disease", "hypertension", "kidney_disease", "liver_disease", "stroke", "breast_cancer"]
    },
    {
        "id": "q3_height",
        "index": 3,
        "feature_keys": ["height", "BMI", "bmi"],
        "category": "vitals",
        "questions": {
            "en": "What is your height?",
            "hi": "आपकी लंबाई (Height) कितनी है?",
            "bn": "আপনার উচ্চতা কত?"
        },
        "acknowledgments": {
            "en": "Thank you. I've recorded {value}.",
            "hi": "धन्यवाद, आपकी लंबाई {value} नोट कर ली गई है।",
            "bn": "ধন্যবাদ, আপনার উচ্চতা {value} রেকর্ড করা হয়েছে।"
        },
        "diseases": ["all", "diabetes", "hypertension", "stroke", "kidney_disease"]
    },
    {
        "id": "q4_weight",
        "index": 4,
        "feature_keys": ["weight", "BMI", "bmi"],
        "category": "vitals",
        "questions": {
            "en": "What is your weight?",
            "hi": "आपका वजन (Weight) कितना है?",
            "bn": "আপনার ওজন কত?"
        },
        "acknowledgments": {
            "en": "Recorded {value}. Your calculated BMI is {bmi}.",
            "hi": "वजन {value} दर्ज किया गया। आपका बीएमआई (BMI) {bmi} है।",
            "bn": "ওজন {value} রেকর্ড করা হয়েছে। আপনার বিএমআই (BMI) {bmi}।"
        },
        "diseases": ["all", "diabetes", "hypertension", "stroke", "kidney_disease"]
    },
    {
        "id": "q5_family_history",
        "index": 5,
        "feature_keys": ["family_history", "HeartDiseaseorAttack", "Stroke"],
        "category": "history",
        "questions": {
            "en": "Do you have a family history of diabetes or other relevant disease?",
            "hi": "क्या आपके परिवार में मधुमेह (Diabetes) या दिल की बीमारी का इतिहास रहा है?",
            "bn": "আপনার পরিবারে কি ডায়াবেটিস বা অন্যান্য রোগের ইতিহাস আছে?"
        },
        "acknowledgments": {
            "en": "Understood. Family history noted.",
            "hi": "समझ गया, पारिवारिक इतिहास नोट कर लिया है।",
            "bn": "বুঝেছি, পারিবারিক ইতিহাস নোট করা হয়েছে।"
        },
        "diseases": ["all", "diabetes", "hypertension", "heart_disease"]
    },
    {
        "id": "q6_high_bp",
        "index": 6,
        "feature_keys": ["HighBP", "hypertension", "htn"],
        "category": "vitals",
        "questions": {
            "en": "Do you have high blood pressure?",
            "hi": "क्या आपको उच्च रक्तचाप (High Blood Pressure) की समस्या है?",
            "bn": "আপনার কি উচ্চ রক্তচাপ (High Blood Pressure) আছে?"
        },
        "acknowledgments": {
            "en": "Noted. Blood pressure status recorded.",
            "hi": "नोट कर लिया गया, ब्लड प्रेशर की स्थिति दर्ज हो गई।",
            "bn": "নোট করা হয়েছে, রক্তচাপের তথ্য রেকর্ড করা হয়েছে।"
        },
        "diseases": ["all", "diabetes", "heart_disease", "hypertension", "stroke", "kidney_disease"]
    },
    {
        "id": "q7_bp_reading",
        "index": 7,
        "feature_keys": ["systolic_bp", "diastolic_bp", "trestbps", "bp"],
        "category": "vitals",
        "questions": {
            "en": "Do you know your blood pressure reading?",
            "hi": "क्या आपको अपना ब्लड प्रेशर रीडिंग (जैसे 120/80) याद है?",
            "bn": "আপনি কি আপনার সাম্প্রতিক রক্তচাপের রিডিং জানেন?"
        },
        "acknowledgments": {
            "en": "Recorded blood pressure reading {value}.",
            "hi": "ब्लड प्रेशर रीडिंग {value} दर्ज कर ली गई है।",
            "bn": "রক্তচাপের রিডিং {value} রেকর্ড করা হয়েছে।"
        },
        "diseases": ["all", "heart_disease", "hypertension", "kidney_disease"]
    },
    {
        "id": "q8_heart_rate",
        "index": 8,
        "feature_keys": ["heart_rate", "thalach"],
        "category": "vitals",
        "questions": {
            "en": "What is your heart rate, if you know it?",
            "hi": "यदि आपको पता हो, तो आपकी हृदय गति (Heart Rate) कितनी है?",
            "bn": "আপনার জানা থাকলে, আপনার হৃদস্পন্দন (Heart Rate) কত?"
        },
        "acknowledgments": {
            "en": "Recorded heart rate as {value} beats per minute.",
            "hi": "हृदय गति {value} बीपीएम दर्ज की गई।",
            "bn": "হৃদস্পন্দন {value} বিপিএম রেকর্ড করা হয়েছে।"
        },
        "diseases": ["all", "heart_disease", "hypertension"]
    },
    {
        "id": "q9_blood_glucose",
        "index": 9,
        "feature_keys": ["fasting_glucose", "avg_glucose_level", "bgr", "fbs", "dm"],
        "category": "biomarkers",
        "questions": {
            "en": "Do you know your blood glucose level?",
            "hi": "क्या आपको अपना ब्लड शुगर या ग्लूकोज स्तर पता है?",
            "bn": "আপনি কি আপনার রক্তের গ্লুকোজ বা সুগারের মাত্রা জানেন?"
        },
        "acknowledgments": {
            "en": "Recorded blood glucose as {value} mg/dL.",
            "hi": "ब्लड ग्लूकोज {value} mg/dL दर्ज किया गया।",
            "bn": "রক্তের গ্লুকোজ {value} mg/dL রেকর্ড করা হয়েছে।"
        },
        "diseases": ["all", "diabetes", "heart_disease", "hypertension", "stroke", "kidney_disease"]
    },
    {
        "id": "q10_spo2",
        "index": 10,
        "feature_keys": ["spo2", "oxygen_saturation"],
        "category": "vitals",
        "questions": {
            "en": "Do you know your oxygen saturation or SpO₂?",
            "hi": "क्या आपको अपना ऑक्सीजन स्तर (SpO₂) पता है?",
            "bn": "আপনি কি আপনার অক্সিজেন স্যাচুরেশন বা SpO₂ জানেন?"
        },
        "acknowledgments": {
            "en": "Recorded oxygen saturation as {value} percent.",
            "hi": "ऑक्सीजन स्तर {value}% दर्ज किया गया।",
            "bn": "অক্সিজেন স্যাচুরেশন {value}% রেকর্ড করা হয়েছে।"
        },
        "diseases": ["all", "hypertension"]
    },
    {
        "id": "q11_waist",
        "index": 11,
        "feature_keys": ["waist_circumference", "waist"],
        "category": "vitals",
        "questions": {
            "en": "What is your waist circumference, if you know it?",
            "hi": "यदि आपको पता हो, तो आपकी कमर का माप (Waist Circumference) कितना है?",
            "bn": "আপনার জানা থাকলে, আপনার কোমরের মাপ কত?"
        },
        "acknowledgments": {
            "en": "Recorded waist circumference {value}.",
            "hi": "कमर का माप {value} दर्ज किया गया।",
            "bn": "কোমরের পরিধি {value} রেকর্ড করা হয়েছে।"
        },
        "diseases": ["all", "diabetes", "hypertension"]
    },
    {
        "id": "q12_physical_activity",
        "index": 12,
        "feature_keys": ["PhysActivity", "physical_activity_hours"],
        "category": "lifestyle",
        "questions": {
            "en": "How physically active are you on a typical day?",
            "hi": "एक सामान्य दिन में आप शारीरिक रूप से कितने सक्रिय रहते हैं?",
            "bn": "একটি সাধারণ দিনে আপনি শারীরিকভাবে কতটা সক্রিয় থাকেন?"
        },
        "acknowledgments": {
            "en": "Thank you, daily physical activity recorded.",
            "hi": "धन्यवाद, आपकी दैनिक शारीरिक सक्रियता दर्ज हो गई है।",
            "bn": "ধন্যবাদ, দৈনিক শারীরিক কার্যকলাপের তথ্য রেকর্ড করা হয়েছে।"
        },
        "diseases": ["all", "diabetes", "hypertension", "heart_disease"]
    },
    {
        "id": "q13_exercise_freq",
        "index": 13,
        "feature_keys": ["PhysActivity", "physical_activity_hours"],
        "category": "lifestyle",
        "questions": {
            "en": "How often do you exercise?",
            "hi": "आप सप्ताह में कितनी बार व्यायाम या कसरत करते हैं?",
            "bn": "আপনি কত ঘন ঘন ব্যায়াম বা শরীরচর্চা করেন?"
        },
        "acknowledgments": {
            "en": "Exercise frequency noted.",
            "hi": "व्यायाम की जानकारी नोट कर ली गई है।",
            "bn": "ব্যায়ামের নিয়মিততার তথ্য নোট করা হয়েছে।"
        },
        "diseases": ["all", "diabetes", "hypertension", "heart_disease"]
    },
    {
        "id": "q14_sedentary_hours",
        "index": 14,
        "feature_keys": ["sedentary_hours", "DiffWalk"],
        "category": "lifestyle",
        "questions": {
            "en": "Approximately how many hours do you spend sitting or using screens each day?",
            "hi": "प्रतिदिन आप लगभग कितने घंटे बैठकर या स्क्रीन पर बिताते हैं?",
            "bn": "প্রতিদিন আপনি আনুমানিক কত ঘণ্টা বসে বা স্ক্রিনের সামনে কাটান?"
        },
        "acknowledgments": {
            "en": "Noted, approximately {value} sedentary hours daily.",
            "hi": "नोट कर लिया, प्रतिदिन लगभग {value} घंटे बैठने की अवधि।",
            "bn": "নোট করা হয়েছে, দৈনিক আনুমানিক {value} ঘণ্টা বসে থাকার সময়।"
        },
        "diseases": ["all", "diabetes", "hypertension"]
    },
    {
        "id": "q15_smoker",
        "index": 15,
        "feature_keys": ["Smoker", "smoking_status", "smoking_history"],
        "category": "lifestyle",
        "questions": {
            "en": "Do you smoke or use tobacco?",
            "hi": "क्या आप धूम्रपान या तंबाकू का सेवन करते हैं?",
            "bn": "আপনি কি ধূমপান বা তামাক সেবন করেন?"
        },
        "acknowledgments": {
            "en": "Thank you, tobacco status recorded.",
            "hi": "धन्यवाद, तंबाकू/धूम्रपान की जानकारी दर्ज हो गई।",
            "bn": "ধন্যবাদ, ধূমপানের তথ্য রেকর্ড করা হয়েছে।"
        },
        "diseases": ["all", "diabetes", "heart_disease", "hypertension", "stroke"]
    },
    {
        "id": "q16_alcohol",
        "index": 16,
        "feature_keys": ["HvyAlcoholConsump", "alcohol"],
        "category": "lifestyle",
        "questions": {
            "en": "Do you drink alcohol?",
            "hi": "क्या आप शराब (Alcohol) का सेवन करते हैं?",
            "bn": "আপনি কি অ্যালকোহল বা মদ পান করেন?"
        },
        "acknowledgments": {
            "en": "Noted, alcohol intake recorded.",
            "hi": "नोट कर लिया गया, शराब सेवन की जानकारी दर्ज हुई।",
            "bn": "নোট করা হয়েছে, অ্যালকোহল সংক্রান্ত তথ্য রেকর্ড করা হয়েছে।"
        },
        "diseases": ["all", "diabetes", "hypertension", "liver_disease"]
    },
    {
        "id": "q17_water_intake",
        "index": 17,
        "feature_keys": ["water_intake"],
        "category": "lifestyle",
        "questions": {
            "en": "How much water do you usually drink each day?",
            "hi": "आप आमतौर पर रोजाना कितना पानी पीते हैं?",
            "bn": "আপনি সাধারণত প্রতিদিন কতটা জল পান করেন?"
        },
        "acknowledgments": {
            "en": "Recorded daily hydration as {value}.",
            "hi": "दैनिक पानी का सेवन {value} दर्ज किया गया।",
            "bn": "দৈনিক জলের পরিমাণ {value} রেকর্ড করা হয়েছে।"
        },
        "diseases": ["all", "kidney_disease"]
    },
    {
        "id": "q18_junk_food",
        "index": 18,
        "feature_keys": ["GenHlth", "junk_food_freq"],
        "category": "lifestyle",
        "questions": {
            "en": "How often do you eat junk or highly processed food?",
            "hi": "आप कितनी बार जंक फूड या प्रोसेस्ड खाना खाते हैं?",
            "bn": "আপনি কত ঘন ঘন জাঙ্ক ফুড বা প্রক্রিয়াজাত খাবার খান?"
        },
        "acknowledgments": {
            "en": "Dietary habits noted.",
            "hi": "खान-पान की आदतें नोट कर ली गई हैं।",
            "bn": "খাদ্যাভ্যাসের তথ্য নোট করা হয়েছে।"
        },
        "diseases": ["all", "diabetes", "hypertension"]
    },
    {
        "id": "q19_fruits_veggies",
        "index": 19,
        "feature_keys": ["Fruits", "Veggies"],
        "category": "lifestyle",
        "questions": {
            "en": "How often do you eat fruits and vegetables?",
            "hi": "आप फलों और सब्जियों का सेवन कितनी बार करते हैं?",
            "bn": "আপনি কত ঘন ঘন ফল ও শাকসবজি খান?"
        },
        "acknowledgments": {
            "en": "Fruit and vegetable consumption recorded.",
            "hi": "फल और सब्जियों के सेवन की जानकारी दर्ज हो गई।",
            "bn": "ফল ও শাকসবজি খাওয়ার তথ্য রেকর্ড করা হয়েছে।"
        },
        "diseases": ["all", "diabetes"]
    },
    {
        "id": "q20_sleep_hours",
        "index": 20,
        "feature_keys": ["sleep_hours", "PhysHlth"],
        "category": "lifestyle",
        "questions": {
            "en": "How many hours do you usually sleep?",
            "hi": "आप आमतौर पर कितने घंटे सोते हैं?",
            "bn": "আপনি সাধারণত কত ঘণ্টা ঘুমান?"
        },
        "acknowledgments": {
            "en": "Recorded approximately {value} hours of sleep.",
            "hi": "लगभग {value} घंटे की नींद दर्ज की गई।",
            "bn": "আনুমানিক {value} ঘণ্টা ঘুমের সময় রেকর্ড করা হয়েছে।"
        },
        "diseases": ["all", "hypertension", "diabetes"]
    },
    {
        "id": "q21_sleep_quality",
        "index": 21,
        "feature_keys": ["MentHlth", "sleep_quality"],
        "category": "lifestyle",
        "questions": {
            "en": "How would you describe your sleep quality?",
            "hi": "आप अपनी नींद की गुणवत्ता (Sleep Quality) को कैसा बताएंगे?",
            "bn": "আপনার ঘুমের মান কেমন বলে মনে করেন?"
        },
        "acknowledgments": {
            "en": "Sleep quality noted.",
            "hi": "नींद की गुणवत्ता नोट कर ली गई है।",
            "bn": "ঘুমের মানের তথ্য নোট করা হয়েছে।"
        },
        "diseases": ["all", "diabetes"]
    },
    {
        "id": "q22_sleep_regular",
        "index": 22,
        "feature_keys": ["sleep_regular"],
        "category": "lifestyle",
        "questions": {
            "en": "Do you usually maintain a regular sleep schedule?",
            "hi": "क्या आप आमतौर पर एक नियमित समय पर सोते और जागते हैं?",
            "bn": "আপনি কি নিয়মিত ঘুমের সময়সূচী বজায় রাখেন?"
        },
        "acknowledgments": {
            "en": "Sleep schedule consistency noted.",
            "hi": "सोने के समय की नियमितता नोट कर ली गई।",
            "bn": "ঘুমের নিয়মিততার তথ্য নোট করা হয়েছে।"
        },
        "diseases": ["all", "diabetes"]
    },
    {
        "id": "q23_snoring",
        "index": 23,
        "feature_keys": ["snoring", "sleep_apnea"],
        "category": "symptoms",
        "questions": {
            "en": "Do you snore while sleeping?",
            "hi": "क्या आप सोते समय खर्राटे लेते हैं?",
            "bn": "ঘুমানোর সময় কি আপনার নাক ডাকার সমস্যা আছে?"
        },
        "acknowledgments": {
            "en": "Snoring status noted.",
            "hi": "खर्राटों की स्थिति नोट कर ली गई है।",
            "bn": "নাক ডাকার তথ্য নোট করা হয়েছে।"
        },
        "diseases": ["all", "hypertension", "heart_disease"]
    },
    {
        "id": "q24_thirst_urination",
        "index": 24,
        "feature_keys": ["thirst_urination", "dm"],
        "category": "symptoms",
        "questions": {
            "en": "Do you frequently experience unusual thirst or frequent urination?",
            "hi": "क्या आपको बार-बार असामान्य प्यास या बार-बार पेशाब आने की समस्या होती है?",
            "bn": "আপনার কি ঘন ঘন অতিরিক্ত তৃষ্ণা বা ঘন ঘন প্রস্রাবের সমস্যা হয়?"
        },
        "acknowledgments": {
            "en": "Noted metabolic symptom response.",
            "hi": "लक्षण की जानकारी नोट कर ली गई है।",
            "bn": "উপসর্গের তথ্য নোট করা হয়েছে।"
        },
        "diseases": ["all", "diabetes", "kidney_disease"]
    },
    {
        "id": "q25_weight_changes",
        "index": 25,
        "feature_keys": ["unexplained_weight_change"],
        "category": "symptoms",
        "questions": {
            "en": "Have you experienced unexplained weight changes recently?",
            "hi": "क्या हाल ही में आपके वजन में अचानक या बिना कारण कोई बदलाव आया है?",
            "bn": "সম্প্রতি আপনার ওজনে কি কোনো অস্বাভাবিক পরিবর্তন হয়েছে?"
        },
        "acknowledgments": {
            "en": "Weight change status recorded.",
            "hi": "वजन में बदलाव की जानकारी दर्ज कर ली गई है।",
            "bn": "ওজন পরিবর্তনের তথ্য রেকর্ড করা হয়েছে।"
        },
        "diseases": ["all", "diabetes", "kidney_disease", "liver_disease"]
    },
    {
        "id": "q26_fatigue",
        "index": 26,
        "feature_keys": ["PhysHlth", "fatigue"],
        "category": "symptoms",
        "questions": {
            "en": "Have you been experiencing persistent fatigue?",
            "hi": "क्या आप लगातार थकान या कमजोरी महसूस कर रहे हैं?",
            "bn": "আপনি কি ক্রমাগত ক্লান্তি বা দুর্বলতা অনুভব করছেন?"
        },
        "acknowledgments": {
            "en": "Noted your response regarding fatigue.",
            "hi": "थकान संबंधी प्रतिक्रिया नोट कर ली गई है।",
            "bn": "ক্লান্তি সম্পর্কিত তথ্য নোট করা হয়েছে।"
        },
        "diseases": ["all", "diabetes", "kidney_disease", "heart_disease"]
    },
    {
        "id": "q27_shortness_breath",
        "index": 27,
        "feature_keys": ["dyspnea", "shortness_of_breath"],
        "category": "symptoms",
        "questions": {
            "en": "Do you experience shortness of breath?",
            "hi": "क्या आपको सांस लेने में तकलीफ या सांस फूलने की समस्या होती है?",
            "bn": "আপনার কি শ্বাসকষ্ট বা শ্বাস নিতে সমস্যা হয়?"
        },
        "acknowledgments": {
            "en": "Recorded your response on shortness of breath.",
            "hi": "सांस संबंधी लक्षण दर्ज कर लिया गया है।",
            "bn": "শ্বাসকষ্ট সংক্রান্ত তথ্য রেকর্ড করা হয়েছে।"
        },
        "diseases": ["all", "heart_disease", "hypertension"]
    },
    {
        "id": "q28_chest_discomfort",
        "index": 28,
        "feature_keys": ["cp", "chest_discomfort"],
        "category": "symptoms",
        "questions": {
            "en": "Do you experience chest discomfort?",
            "hi": "क्या आपको सीने में कोई दर्द या असहजता (Chest Discomfort) महसूस होती है?",
            "bn": "আপনার কি বুকে ব্যথা বা অস্বস্তি অনুভূত হয়?"
        },
        "acknowledgments": {
            "en": "Understood, chest discomfort status recorded.",
            "hi": "समझ गया, सीने की असहजता की जानकारी नोट कर ली गई।",
            "bn": "বুঝেছি, বুকে অস্বস্তির তথ্য রেকর্ড করা হয়েছে।"
        },
        "diseases": ["all", "heart_disease"]
    },
    {
        "id": "q29_cough",
        "index": 29,
        "feature_keys": ["persistent_cough", "cough"],
        "category": "symptoms",
        "questions": {
            "en": "Have you had a persistent cough?",
            "hi": "क्या आपको लंबे समय से खांसी (Persistent Cough) की समस्या है?",
            "bn": "আপনার কি দীর্ঘদিন ধরে স্থায়ী কাশির সমস্যা আছে?"
        },
        "acknowledgments": {
            "en": "Persistent cough status recorded.",
            "hi": "खांसी संबंधी जानकारी दर्ज कर ली गई है।",
            "bn": "কাশির তথ্য রেকর্ড করা হয়েছে।"
        },
        "diseases": ["all"]
    }
]

def get_questions_for_disease(disease_id: str = "diabetes") -> List[Dict[str, Any]]:
    """
    Returns the ordered, tailored list of clinical questions relevant to the target disease model.
    Avoids asking irrelevant questions when the selected disease model does not require them.
    """
    d_id = (disease_id or "diabetes").lower()
    
    if d_id == "all":
        return list(CLINICAL_QUESTIONS_CATALOG)

    # Only include questions that explicitly match the requested disease model
    selected_questions: List[Dict[str, Any]] = []
    for q in CLINICAL_QUESTIONS_CATALOG:
        target_diseases = q.get("diseases", [])
        if d_id in target_diseases:
            selected_questions.append(q)

    # Ensure questions maintain canonical order 1..29
    selected_questions.sort(key=lambda x: x["index"])
    return selected_questions

def get_question_by_id(question_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves question definition by unique question ID."""
    for q in CLINICAL_QUESTIONS_CATALOG:
        if q["id"] == question_id:
            return q
    return None
