import re
from typing import Dict, Any, Optional, Tuple

SUPPORTED_LANGUAGES: Dict[str, Dict[str, str]] = {
    "auto": {"name": "Auto Detect", "native": "Auto Detect", "bcp47": "auto", "script": "universal"},
    "en": {"name": "English", "native": "English", "bcp47": "en-US", "script": "latin"},
    "hi": {"name": "Hindi", "native": "हिंदी", "bcp47": "hi-IN", "script": "devanagari"},
    "bn": {"name": "Bengali", "native": "বাংলা", "bcp47": "bn-IN", "script": "bengali"},
    "ta": {"name": "Tamil", "native": "தமிழ்", "bcp47": "ta-IN", "script": "tamil"},
    "te": {"name": "Telugu", "native": "తెలుగు", "bcp47": "te-IN", "script": "telugu"},
    "mr": {"name": "Marathi", "native": "मराठी", "bcp47": "mr-IN", "script": "devanagari"},
    "gu": {"name": "Gujarati", "native": "ગુજરાતી", "bcp47": "gu-IN", "script": "gujarati"},
    "kn": {"name": "Kannada", "native": "ಕನ್ನಡ", "bcp47": "kn-IN", "script": "kannada"},
    "ml": {"name": "Malayalam", "native": "മലയാളം", "bcp47": "ml-IN", "script": "malayalam"},
    "pa": {"name": "Punjabi", "native": "ਪੰਜਾਬੀ", "bcp47": "pa-IN", "script": "gurmukhi"},
    "ur": {"name": "Urdu", "native": "اردو", "bcp47": "ur-PK", "script": "arabic"},
}

NATIVE_DIGIT_MAP = {
    # Devanagari (Hindi / Marathi)
    '०': '0', '१': '1', '२': '2', '३': '3', '४': '4', '५': '5', '६': '6', '७': '7', '८': '8', '९': '9',
    # Bengali
    '০': '0', '১': '1', '২': '2', '৩': '3', '৪': '4', '৫': '5', '৬': '6', '৭': '7', '৮': '8', '৯': '9',
    # Gujarati
    '૦': '0', '૧': '1', '૨': '2', '૩': '3', '૪': '4', '૫': '5', '૬': '6', '૭': '7', '૮': '8', '૯': '9',
    # Gurmukhi (Punjabi)
    '੦': '0', '੧': '1', '੨': '2', '੩': '3', '੪': '4', '੫': '5', '੬': '6', '੭': '7', '੮': '8', '੯': '9',
    # Tamil
    '௦': '0', '௧': '1', '௨': '2', '௩': '3', '௪': '4', '௫': '5', '௬': '6', '௭': '7', '௮': '8', '௯': '9',
    # Telugu
    '౦': '0', '౧': '1', '౨': '2', '౩': '3', '౪': '4', '౫': '5', '౬': '6', '౭': '7', '౮': '8', '౯': '9',
    # Kannada
    '೦': '0', '೧': '1', '೨': '2', '೩': '3', '೪': '4', '೫': '5', '೬': '6', '೭': '7', '೮': '8', '೯': '9',
    # Malayalam
    '൦': '0', '൧': '1', '൨': '2', '൩': '3', '൪': '4', '<ctrl42>': '5', '<ctrl42>': '6', '<ctrl42>': '7', '<ctrl42>': '8', '<ctrl42>': '9',
    # Urdu / Eastern Arabic
    '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4', '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9'
}

MULTILINGUAL_NUMBER_WORDS = {
    # Hindi / Devanagari / Hinglish
    "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पांच": 5, "छह": 6, "सात": 7, "आठ": 8, "नौ": 9, "दस": 10,
    "पचीस": 25, "तीस": 30, "पैंतीस": 35, "चालिस": 40, "पैंतालीस": 45, "पचास": 50, "पचपन": 55, "साठ": 60, "पैंसठ": 65, "सत्तर": 70, "अस्सी": 80, "नब्बे": 90, "सौ": 100,
    "ek": 1, "do": 2, "teen": 3, "char": 4, "paanch": 5, "che": 6, "saat": 7, "aath": 8, "nau": 9, "das": 10,
    "pachis": 25, "tees": 30, "chalis": 40, "pachas": 50, "saath": 60, "sattar": 70, "assi": 80, "nabbe": 90, "sau": 100,
    # Bengali / Benglish
    "এক": 1, "দুই": 2, "তিন": 3, "চার": 4, "পাঁচ": 5, "ছয়": 6, "সাত": 7, "আট": 8, "নয়": 9, "দশ": 10,
    "পঁচিশ": 25, "ত্রিশ": 30, "পঁয়ত্রিশ": 35, "চল্লিশ": 40, "পঁয়তাল্লিশ": 45, "পঞ্চাশ": 50, "ষাট": 60, "সত্তর": 70, "আশি": 80, "নব্বই": 90, "একশো": 100,
    "ek": 1, "dui": 2, "tin": 3, "char": 4, "panch": 5, "choy": 6, "sat": 7, "aat": 8, "noy": 9, "dosh": 10,
    "ponchish": 25, "trish": 30, "chollish": 40, "ponchash": 50, "shat": 60, "shottor": 70, "ashi": 80, "nobboi": 90, "eksho": 100,
    # Tamil
    "ஒன்று": 1, "இரண்டு": 2, "மூன்று": 3, "நான்கு": 4, "ஐந்து": 5, "ஆறு": 6, "ஏழு": 7, "எட்டு": 8, "ஒன்பது": 9, "பத்து": 10,
    "இருபத்தைந்து": 25, "முப்பது": 30, "நாற்பது": 40, "ஐம்பது": 50, "அறுபது": 60, "எழுபது": 70, "எண்பது": 80, "தொண்ணூறு": 90, "நூறு": 100,
    # Telugu
    "ఒకటి": 1, "రెండు": 2, "మూడు": 3, "నాలుగు": 4, "ఐదు": 5, "ఆరు": 6, "ఏడు": 7, "ఎనిమిది": 8, "తొమ్మిది": 9, "పది": 10,
    "ఇరవై ఐదు": 25, "ముప్పై": 30, "నలభై": 40, "యాభై": 50, "అరవై": 60, "దెబ్బై": 70, "ఎనభై": 80, "తొమ్మిది": 90, "వంద": 100,
    # Marathi
    "एक": 1, "दोन": 2, "तीन": 3, "चार": 4, "पाच": 5, "सहा": 6, "सात": 7, "आठ": 8, "ऊऊ": 9, "दहा": 10,
    "पंचवीस": 25, "तीस": 30, "चाळीस": 40, "पन्नास": 50, "साठ": 60, "सत्तर": 70, "अंशी": 80, "नव्वद": 90, "शंभर": 100,
    # Gujarati
    "એક": 1, "બે": 2, "ત્રણ": 3, "ચાર": 4, "પાંચ": 5, "છ": 6, "સાત": 7, "આઠ": 8, "નવ": 9, "દસ": 10,
    "પંચાવન": 55, "ત્રીસ": 30, "ચાળીસ": 40, "પચાસ": 50, "સાઠ": 60, "સિત્તેર": 70, "એંસી": 80, "નેવુ": 90, "સો": 100,
    # Kannada
    "ಒಂದು": 1, "ಎರಡು": 2, "ಮೂರು": 3, "ನಾಲ್ಕು": 4, "ಐದು": 5, "ಆರು": 6, "ಏಳು": 7, "ಎಂಟು": 8, "ಒಂಬತ್ತು": 9, "ಹತ್ತು": 10,
    "ಇಪ್ಪತ್ತೈದು": 25, "ಮೂವತ್ತು": 30, "ನಲವತ್ತು": 40, "ಐವತ್ತು": 50, "ಅರವತ್ತು": 60, "ಎಪ್ಪತ್ತು": 70, "ಎಂಭತ್ತು": 80, "ತೊಂಬತ್ತು": 90, "ನೂರು": 100,
    # Malayalam
    "ഒന്ന്": 1, "രണ്ട്": 2, "മൂന്ന്": 3, "നാല്": 4, "അഞ്ച്": 5, "ആറ്": 6, "ഏഴ്": 7, "എട്ട്": 8, "ഒൻപത്": 9, "പത്ത്": 10,
    "ഇരുപത്തഞ്ച്": 25, "മുപ്പത്": 30, "നാൽപ്പത്": 40, "അമ്പത്": 50, "അറുപത്": 60, "എഴുപത്": 70, "എൺപത്": 80, "തൊണ്ണൂറ്": 90, "നൂറ്": 100,
    # Punjabi
    "ਇੱਕ": 1, "ਦੋ": 2, "ਤਿੰਨ": 3, "ਚਾਰ": 4, "ਪੰਜ": 5, "ਛੇ": 6, "ਸੱਤ": 7, "ਅੱਠ": 8, "ਨੌਂ": 9, "ਦੱਸ": 10,
    "ਪੱਚੀ": 25, "ਤੀਹ": 30, "ਚਾਲੀ": 40, "ਪੰਜਾਬ": 50, "ਸੱਠ": 60, "ਸੱਤਰ": 70, "ਅੱਸੀ": 80, "ਨੱਬੇ": 90, "ਸੌ": 100,
    # Urdu
    "ایک": 1, "دو": 2, "تین": 3, "چار": 4, "پانچ": 5, "چھ": 6, "سات": 7, "آٹھ": 8, "نو": 9, "دس": 10,
    "پچیس": 25, "تیس": 30, "چالیس": 40, "پچاس": 50, "ساٹھ": 60, "ستر": 70, "اسی": 80, "نوے": 90, "سو": 100
}

LOCALIZED_PROMPTS = {
    "en": {
        "disclaimer": "AI-Assisted Early Disease Risk Support: Probabilistic estimation, NOT a medical diagnosis.",
        "age_ask": "To assess your risk for {disease}, may I ask how old you are?",
        "bmi_ask": "Could you share your height and weight, or your current Body Mass Index (BMI)?",
        "bp_ask": "Have you ever been diagnosed with high blood pressure, or do you know your recent BP numbers?",
        "chol_ask": "Have you been diagnosed with high cholesterol, or do you have a recent cholesterol reading?",
        "smoke_ask": "Have you smoked at least 100 cigarettes in your lifetime, or do you currently smoke?",
        "risk_high": "Based on your clinical profile, our Hybrid Classical-Quantum Ensemble model estimates an elevated risk score of {risk}% for {disease}.",
        "risk_low": "Based on your clinical profile, our Hybrid Classical-Quantum Ensemble model estimates a low risk score of {risk}% for {disease}.",
        "recommendation": "We recommend discussing these indicators with a healthcare professional for clinical validation."
    },
    "hi": {
        "disclaimer": "एआई-सहायक रोग जोखिम सहायता: यह एक संभावित अनुमान है, चिकित्सीय निदान नहीं।",
        "age_ask": "{disease} जोखिम मूल्यांकन के लिए, कृपया अपनी उम्र बताएं।",
        "bmi_ask": "क्या आप अपनी लंबाई और वजन, या बीएमआई (BMI) शेयर कर सकते हैं?",
        "bp_ask": "क्या आपको उच्च रक्तचाप (High Blood Pressure) की शिकायत है?",
        "chol_ask": "क्या आपका कोलेस्ट्रॉल (Cholesterol) बढ़ा हुआ है?",
        "smoke_ask": "क्या आप धूम्रपान (Smoking) करते हैं?",
        "risk_high": "आपकी नैदानिक जानकारी के अनुसार, हमारे हाइब्रिड क्लासिकल-क्वांटम मॉडल ने {disease} का जोखिम {risk}% (उच्च) आंका है।",
        "risk_low": "आपकी जानकारी के अनुसार, हमारे हाइब्रिड मॉडल ने {disease} का जोखिम {risk}% (कम) आंका है।",
        "recommendation": "हम अनुशंसा करते हैं कि आप चिकित्सीय पुष्टि के लिए डॉक्टर से परामर्श लें।"
    },
    "bn": {
        "disclaimer": "এআই-সহায়তা স্বাস্থ্য ঝুঁকি মূল্যায়ন: এটি একটি আনুমানিক সম্ভাবনা, কোনো চিকিৎসা রোগনির্ণয় নয়।",
        "age_ask": "{disease} ঝুঁকি মূল্যায়নের জন্য, আপনার বয়স কত তা বলবেন কি?",
        "bmi_ask": "আপনার উচ্চতা ও ওজন, অথবা বিএমআই (BMI) কত তা বলবেন কি?",
        "bp_ask": "আপনার কি উচ্চ রক্তচাপ (High BP) বা হাইপারটেনশনের সমস্যা আছে?",
        "chol_ask": "আপনার কি হাই কোলেস্টেরল (High Cholesterol) আছে?",
        "smoke_ask": "আপনি কি ধূপপান (Smoking) করেন?",
        "risk_high": "আপনার ক্লিনিকাল প্রোফাইল অনুযায়ী, আমাদের হাইব্রিড কোয়ান্টাম মডেল {disease}-এর ঝুঁকি {risk}% (উচ্চ) নির্ধারণ করেছে।",
        "risk_low": "আপনার তথ্য অনুযায়ী, আমাদের মডেল {disease}-এর ঝুঁকি {risk}% (কম) নির্ধারণ করেছে।",
        "recommendation": "আমরা একজন নিবন্ধিত চিকিৎসকের সাথে পরামর্শ করার পরামর্শ দিচ্ছি।"
    },
    "ta": {
        "disclaimer": "AI-உதவி நோய் அபாய மதிப்பீடு: இது ஒரு முன்கணிப்பு மட்டுமே, மருத்துவ நோயறிதல் அல்ல.",
        "age_ask": "{disease} அபாயத்தை மதிப்பிட, உங்கள் வயது என்ன என்று கூற முடியுமா?",
        "bmi_ask": "உங்கள் உயரம் மற்றும் எடையைப் பகிர முடியுமா?",
        "bp_ask": "உங்களுக்கு உயர் இரத்த அழுத்தம் (High BP) உள்ளதா?",
        "chol_ask": "உங்களுக்கு கொலஸ்ட்ரால் (Cholesterol) அதிகமாக உள்ளதா?",
        "smoke_ask": "நீங்கள் புகைபிடிக்கும் பழக்கம் உள்ளவரா (Smoking)?",
        "risk_high": "உங்கள் தகவல்களின்படி, எங்கள் ஹைபிரிட் குவாண்டம் மாடல் {disease} அபாயம் {risk}% (அதிகம்) எனக் கணக்கிட்டுள்ளது.",
        "risk_low": "உங்கள் தகவல்களின்படி, எங்கள் மாடல் {disease} அபாயம் {risk}% (குறைவு) எனக் கணக்கிட்டுள்ளது.",
        "recommendation": "மருத்துவ உறுதிப்படுத்தலுக்கு ஒரு மருத்துவரை அணுகுமாறு பரிந்துரைக்கிறோம்."
    },
    "te": {
        "disclaimer": "AI-ఆధారిత వ్యాధి ప్రమాద అంచనా: ఇది సంభావ్యత అంచనా మాత్రమే, వైద్య రోగనిర్ధారణ కాదు.",
        "age_ask": "{disease} ప్రమాదాన్ని అంచనా వేయడానికి, మీ వయస్సు ఎంత అని చెప్పగలరా?",
        "bmi_ask": "మీ ఎత్తు మరియు బరువు, లేదా BMI చెప్పగలరా?",
        "bp_ask": "మీకు అధిక రక్తపోటు (High BP) ఉందా?",
        "chol_ask": "మీకు హై కొలెస్ట్రాల్ (High Cholesterol) ఉందా?",
        "smoke_ask": "మీరు పొగతాగే అలవాటు (Smoking) కలిగి ఉన్నారా?",
        "risk_high": "మీ సమాచారం ప్రకారం, మా హైబ్రిడ్ మోడల్ {disease} ప్రమాదం {risk}% (అధికం) అని అంచనా వేసింది.",
        "risk_low": "మీ సమాచారం ప్రకారం, మా మోడల్ {disease} ప్రమాదం {risk}% (తక్కువ) అని అంచనా వేసింది.",
        "recommendation": "తదుపరి నిర్ధారణ కోసం వైద్యుడిని సంప్రదించాల్సిందిగా సిఫార్సు చేస్తున్నాము."
    },
    "mr": {
        "disclaimer": "AI-सहाय्यित आजार धोका मूल्यांकन: हा संभाव्यता अंदाज आहे, वैद्यकीय निदान नाही.",
        "age_ask": "{disease} धोक्याच्या मूल्यांकनासाठी, कृपया तुमचे वय सांगा.",
        "bmi_ask": "तुमची उंची आणि वजन, किंवा BMI शेअर करू शकता का?",
        "bp_ask": "तुम्हाला उच्च रक्तदाब (High BP) आहे का?",
        "chol_ask": "तुमचे कोलेस्ट्रॉल (High Cholesterol) वाढलेले आहे का?",
        "smoke_ask": "तुम्ही धूम्रपान (Smoking) करता का?",
        "risk_high": "तुमच्या माहितीनुसार, आमच्या हायब्रिड मॉडेलने {disease} चा धोका {risk}% (जास्त) दर्शवला आहे.",
        "risk_low": "तुमच्या माहितीनुसार, आमच्या मॉडेलने {disease} चा धोका {risk}% (कमी) दर्शवला आहे.",
        "recommendation": "वैद्यकीय सल्ल्यासाठी डॉक्टरांशी संपर्क साधावा."
    },
    "gu": {
        "disclaimer": "AI-સહાયિત રોગ જોખમ મૂલ્યાંકન: આ એક સંભવિત અંદાજ છે, તબીબી નિદાન નથી.",
        "age_ask": "{disease} જોખમ મૂલ્યાંકન માટે, તમારી ઉંમર કેટલી છે?",
        "bmi_ask": "શું તમે તમારી ઊંચાઈ અને વજન, અથવા BMI કહી શકો છો?",
        "bp_ask": "શું તમને હાઈ બ્લડ પ્રેશર (High BP) ની તકલીફ છે?",
        "chol_ask": "શું તમારું કોલેસ્ટ્રોલ (High Cholesterol) વધેલું છે?",
        "smoke_ask": "શું તમે ધૂમ્રપાન (Smoking) કરો છો?",
        "risk_high": "તમારી વિગતો અનુસાર, અમારા હાઇબ્રિડ મોડેલે {disease} નું જોખમ {risk}% (ઉચ્ચ) આંક્યું છે.",
        "risk_low": "તમારી વિગતો અનુસાર, અમારા મોડેલે {disease} નું જોખમ {risk}% (ઓછું) આંક્યું છે.",
        "recommendation": "તબીબી ચકાસણી માટે ડોક્ટરની સલાહ લેવા વિનંતી."
    },
    "kn": {
        "disclaimer": "AI-ಸಹಾಯಿತ ರೋಗದ ಅಪಾಯದ ಮೌಲ್ಯಮಾಪನ: ಇದು ಸಂಭವನೀಯತೆಯ ಅಂದಾಜು, ವೈದ್ಯಕೀಯ ರೋಗನಿರ್ಣಯವಲ್ಲ.",
        "age_ask": "{disease} ಅಪಾಯವನ್ನು ಮೌಲ್ಯಮಾಪನ ಮಾಡಲು, ನಿಮ್ಮ ವಯಸ್ಸು ಎಷ್ಟು?",
        "bmi_ask": "ನಿಮ್ಮ ಎತ್ತರ ಮತ್ತು ತೂಕ ಅಥವಾ BMI ತಿಳಿಸುವಿರಾ?",
        "bp_ask": "ನಿಮಗೆ ರಕ್ತದೊತ್ತಡ (High BP) ಇದೆಯೇ?",
        "chol_ask": "ನಿಮಗೆ ಅಧಿಕ ಕೊಲೆಸ್ಟ್ರಾಲ್ (High Cholesterol) ಇದೆಯೇ?",
        "smoke_ask": "ನೀವು ಧೂಮಪಾನ (Smoking) ಮಾಡುತ್ತೀರಾ?",
        "risk_high": "ನಿಮ್ಮ ವಿವರಗಳಂತೆ, ನಮ್ಮ ಹೈಬ್ರಿಡ್ ಮಾದರಿಯು {disease} ಅಪಾಯವನ್ನು {risk}% (ಹೆಚ್ಚು) ಎಂದು ಅಂದಾಜಿಸಿದೆ.",
        "risk_low": "ನಿಮ್ಮ ವಿವರಗಳಂತೆ, ನಮ್ಮ ಮಾದರಿಯು {disease} ಅಪಾಯವನ್ನು {risk}% (ಕಡಿಮೆ) ಎಂದು ಅಂದಾಜಿಸಿದೆ.",
        "recommendation": "ವೈದ್ಯಕೀಯ ಸಲಹೆಗಾಗಿ ವೈದ್ಯರನ್ನು ಸಂಪರ್ಕಿಸಲು ಶಿಫಾರಸು ಮಾಡುತ್ತೇವೆ."
    },
    "ml": {
        "disclaimer": "AI-സഹായത്തോടെയുള്ള രോഗസാധ്യത വിലയിരുത്തൽ: ഇതൊരു അനുമാനം മാത്രമാണ്, വൈദ്യശാസ്ത്രപരമായ രോഗനിർണ്ണയമല്ല.",
        "age_ask": "{disease} സാധ്യത അറിയുന്നതിനായി, നിങ്ങളുടെ വയസ്സ് എത്രയാണെന്ന് പറയാമോ?",
        "bmi_ask": "നിങ്ങളുടെ ഉയരവും ഭാരവും അല്ലെങ്കിൽ BMI എത്രയാണെന്ന് പറയാമോ?",
        "bp_ask": "നിങ്ങൾക്ക് ഉയർന്ന രക്തസമ്മർദ്ദം (High BP) ഉണ്ടോ?",
        "chol_ask": "നിങ്ങൾക്ക് കൊളസ്ട്രോൾ (High Cholesterol) കൂടുതലാണോ?",
        "smoke_ask": "നിങ്ങൾക്ക് പുകവലി (Smoking) ശീലമുണ്ടോ?",
        "risk_high": "നിങ്ങളുടെ വിവരങ്ങൾ അനുസരിച്ച്, ഞങ്ങളുടെ ഹൈബ്രിഡ് മോഡൽ {disease} സാധ്യത {risk}% (കൂടുതൽ) ആയി കണക്കാക്കിയിരിക്കുന്നു.",
        "risk_low": "നിങ്ങളുടെ വിവരങ്ങൾ അനുസരിച്ച്, ഞങ്ങളുടെ മോഡൽ {disease} സാധ്യത {risk}% (കുറവ്) ആയി കണക്കാക്കിയിരിക്കുന്നു.",
        "recommendation": "കൂടുതൽ പരിശോധനകൾക്കായി ഒരു ഡോക്ടറെ കാണാൻ നിർദ്ദേശിക്കുന്നു."
    },
    "pa": {
        "disclaimer": "AI-ਸਹਾਇਤਾ ਪ੍ਰਾਪਤ ਬਿਮਾਰੀ ਜੋਖਮ ਮੁਲਾਂਕਣ: ਇਹ ਇੱਕ ਸੰਭਾਵੀ ਅੰਦਾਜ਼ਾ ਹੈ, ਡਾਕਟਰੀ ਨਿਦਾਨ ਨਹੀਂ।",
        "age_ask": "{disease} ਜੋਖਮ ਦੇ ਮੁਲਾਂਕਣ ਲਈ, ਤੁਹਾਡੀ ਉਮਰ ਕਿੰਨੀ ਹੈ?",
        "bmi_ask": "ਕੀ ਤੁਸੀਂ ਆਪਣਾ ਕੱਦ ਅਤੇ ਵਜ਼ਨ, ਜਾਂ BMI ਦੱਸ ਸਕਦੇ ਹੋ?",
        "bp_ask": "ਕੀ ਤੁਹਾਨੂੰ ਹਾਈ ਬਲੱਡ ਪ੍ਰੈਸ਼ਰ (High BP) ਹੈ?",
        "chol_ask": "ਕੀ ਤੁਹਾਡਾ ਕੋਲੇਸਟ੍ਰੋਲ (High Cholesterol) ਵਧਿਆ ਹੋਇਆ ਹੈ?",
        "smoke_ask": "ਕੀ ਤੁਸੀਂ ਸਿਗਰਟਨੋਸ਼ੀ (Smoking) ਕਰਦੇ ਹੋ?",
        "risk_high": "ਤੁਹਾਡੀ ਜਾਣਕਾਰੀ ਦੇ ਅਨੁਸਾਰ, ਸਾਡੇ ਹਾਈਬ੍ਰਿਡ ਮਾਡਲ ਨੇ {disease} ਦਾ ਜੋਖਮ {risk}% (ਉੱਚ) ਦੱਸਿਆ ਹੈ।",
        "risk_low": "ਤੁਹਾਡੀ ਜਾਣਕਾਰੀ ਦੇ ਅਨੁਸਾਰ, ਸਾਡੇ ਮਾਡਲ ਨੇ {disease} ਦਾ ਜੋਖਮ {risk}% (ਘੱਟ) ਦੱਸਿਆ ਹੈ।",
        "recommendation": "ਡਾਕਟਰੀ ਜਾਂਚ ਲਈ ਡਾਕਟਰ ਦੀ ਸਲਾਹ ਲਓ।"
    },
    "ur": {
        "disclaimer": "AI امدادی بیماری کا خطرہ: یہ ایک تخمینہ ہے، حتمی طبی تشخیص نہیں ہے۔",
        "age_ask": "{disease} کے خطرے کی تشخیص کے لیے، براہ کرم اپنی عمر بتائیں۔",
        "bmi_ask": "کیا اپ اپنی لمبائی اور وزن، یا BMI بتا سکتے ہیں؟",
        "bp_ask": "کیا آپ کو ہائی بلڈ پریشر (High BP) ہے؟",
        "chol_ask": "کیا آپ کا کولیسٹرول (High Cholesterol) زیادہ ہے؟",
        "smoke_ask": "کیا آپ تمباکو نوشی (Smoking) کرتے ہیں؟",
        "risk_high": "آپ کی معلومات کے مطابق، ہمارے ہائبرڈ ماڈل نے {disease} کا خطرہ {risk}% (زیادہ) بتایا ہے۔",
        "risk_low": "آپ کی معلومات کے مطابق، ہمارے ماڈل نے {disease} کا خطرہ {risk}% (کم) بتایا ہے۔",
        "recommendation": "طبی تصدیق کے لیے ڈاکٹر سے مشورہ کرنے کی ہدایت کی جاتی ہے۔"
    }
}

UNICODE_RANGES = {
    "hi": (0x0900, 0x097F),
    "mr": (0x0900, 0x097F),
    "bn": (0x0980, 0x09FF),
    "pa": (0x0A00, 0x0A7F),
    "gu": (0x0A80, 0x0AFF),
    "ta": (0x0B80, 0x0BFF),
    "te": (0x0C00, 0x0C7F),
    "kn": (0x0C80, 0x0CFF),
    "ml": (0x0D00, 0x0D7F),
    "ur": (0x0600, 0x06FF),
}

class LanguageManager:
    """
    Central Manager for Multilingual Language Registration, Script Auto-Detection,
    Native Digit Normalization, Code-Switching Parsing, and Localized Prompts.
    Designed for zero architecture changes when extending to additional languages.
    """

    @staticmethod
    def register_language(code: str, name: str, native: str, bcp47: str, script: str = "latin", prompts: Optional[Dict[str, str]] = None):
        code_clean = code.lower().strip()
        SUPPORTED_LANGUAGES[code_clean] = {
            "name": name,
            "native": native,
            "bcp47": bcp47,
            "script": script
        }
        if prompts:
            LOCALIZED_PROMPTS[code_clean] = prompts

    @staticmethod
    def detect_language_from_text(text: str) -> str:
        if not text:
            return "en"

        counts: Dict[str, int] = {lang: 0 for lang in UNICODE_RANGES}
        for char in text:
            code_pt = ord(char)
            for lang, (start, end) in UNICODE_RANGES.items():
                if start <= code_pt <= end:
                    counts[lang] += 1

        best_lang, max_count = max(counts.items(), key=lambda x: x[1])
        if max_count > 0:
            return best_lang

        text_lower = text.lower()
        hinglish_markers = ["mera", "meri", "umar", "umr", "sahab", "hai", "kya", "hu", "ha", "nahi", "saal", "vajan"]
        benglish_markers = ["amar", "amr", "byas", "boyos", "ache", "achi", "bochor", "na", "hobe", "ki"]

        if any(w in text_lower.split() for w in hinglish_markers):
            return "hi"
        if any(w in text_lower.split() for w in benglish_markers):
            return "bn"

        return "en"

    @staticmethod
    def normalize_native_digits(text: str) -> str:
        """Converts native script digits (Devanagari, Bengali, Arabic/Urdu, Tamil, etc.) to ASCII digits."""
        if not text:
            return ""
        result = []
        for char in text:
            if char in NATIVE_DIGIT_MAP:
                result.append(NATIVE_DIGIT_MAP[char])
            else:
                result.append(char)
        return "".join(result)

    @staticmethod
    def get_language_meta(lang_code: str) -> Dict[str, str]:
        code = (lang_code or "en").lower().split("-")[0]
        return SUPPORTED_LANGUAGES.get(code, SUPPORTED_LANGUAGES["en"])

    @staticmethod
    def get_prompt_template(lang_code: str, prompt_key: str) -> str:
        code = (lang_code or "en").lower().split("-")[0]
        lang_dict = LOCALIZED_PROMPTS.get(code, LOCALIZED_PROMPTS["en"])
        return lang_dict.get(prompt_key, LOCALIZED_PROMPTS["en"].get(prompt_key, ""))

    @staticmethod
    def is_supported_language(lang_code: str) -> bool:
        if not lang_code: return False
        code = lang_code.lower().split("-")[0]
        return code in SUPPORTED_LANGUAGES

    @staticmethod
    def get_fallback_warning(requested_lang: str) -> Optional[str]:
        if not LanguageManager.is_supported_language(requested_lang):
            return f"Requested language '{requested_lang}' is currently unavailable. Falling back to English (en-US)."
        return None

