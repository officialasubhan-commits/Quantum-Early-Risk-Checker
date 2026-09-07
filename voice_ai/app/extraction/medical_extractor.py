import re
from typing import Dict, Any, Tuple, List, Optional
from voice_ai.app.languages.language_manager import LanguageManager, MULTILINGUAL_NUMBER_WORDS

class MedicalEntityExtractor:
    """
    Parses natural language speech transcripts (multilingual & code-switching)
    into standardized clinical parameter dictionaries compatible with the platform's
    multi-disease FastAPI prediction engine (`PatientFeatures`).
    Supports full 29-question clinical intake schema.
    """

    @staticmethod
    def age_to_brfss_category(age_years: float) -> int:
        """Maps continuous age in years to CDC BRFSS 13-level Age Category (1-13)."""
        if age_years < 25: return 1
        elif age_years < 30: return 2
        elif age_years < 35: return 3
        elif age_years < 40: return 4
        elif age_years < 45: return 5
        elif age_years < 50: return 6
        elif age_years < 55: return 7
        elif age_years < 60: return 8
        elif age_years < 65: return 9
        elif age_years < 70: return 10
        elif age_years < 75: return 11
        elif age_years < 80: return 12
        else: return 13

    def preprocess_text(self, text: str) -> str:
        """Converts native digits to ASCII and normalizes number words."""
        norm = LanguageManager.normalize_native_digits(text or "")
        norm_lower = norm.lower()

        for word, val in MULTILINGUAL_NUMBER_WORDS.items():
            norm_lower = re.sub(rf'\b{re.escape(word)}\b', str(val), norm_lower)

        return norm_lower

    def extract_clinical_entities(
        self,
        text: str,
        existing_params: Optional[Dict[str, Any]] = None,
        active_question_id: Optional[str] = None
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Extracts clinical features from user text transcript across all 29 questions.
        Supports continuous natural speech, contextual single-number answers,
        multi-entity responses, and answer corrections.
        Returns (updated_params, list_of_updated_feature_keys).
        """
        params = dict(existing_params or {})
        updated_keys: List[str] = []
        text_norm = self.preprocess_text(text)
        negation_terms = r'(?:\b(?:no|don\'t|not|never|none|negative|zero)\b|नहीं|ना|নেই|না|இல்லை|లేదు|નથી|ਨਹੀਂ|نہیں|ഇല്ല)'

        # 1. Multilingual Age Extraction
        age_patterns = [
            r'\b(?:i am|age is|age|i\'m|am|уम्र|उम्र|वय|বয়স|വയസ്സ്|വയസ്|ವಯಸ್ಸು|વય|ઉંમર|வயது|వయస్సు|ਉਮਰ|عمر)\s*[:=]?\s*(\d{1,3})\b',
            r'\b(\d{1,3})\s*(?:years old|yrs old|yr old|yo|years|yrs|साल|वर्ष|বছর|വയസ്സ്|வருடம்|సంవత్సరాలు|વર્ષ|ਸਾਲ|سال)\b',
            r'\b(?:उम्र|বয়স|عمر|வயது|వయస్సు)\s+(\d{1,3})\b'
        ]
        found_age = None
        for pattern in age_patterns:
            match = re.search(pattern, text_norm)
            if match:
                val = float(match.group(1))
                if 1 <= val <= 120:
                    found_age = val
                    break

        # Contextual single number for age question
        if found_age is None and active_question_id == "q1_age":
            num_match = re.search(r'\b(\d{1,3})\b', text_norm)
            if num_match:
                val = float(num_match.group(1))
                if 1 <= val <= 120:
                    found_age = val

        if found_age is not None:
            params["age"] = found_age
            params["Age"] = self.age_to_brfss_category(found_age)
            updated_keys.extend(["age", "Age"])

        # 2. Multilingual Biological Sex / Gender
        male_terms = r'\b(?:male|man|boy|gentleman|guy|father|पुरुष|आदमी|ছেলে|ஆண்|పురుషుడు|પુરુષ|ਮਰਦ|مرد)\b'
        female_terms = r'\b(?:female|woman|girl|lady|mother|महिला|औरत|स्त्री|মহিলা|মেয়ে|பெண்|స్త్రీ|મહિલા|ਔਰਤ|عورت)\b'

        if re.search(male_terms, text_norm) and not re.search(female_terms, text_norm):
            params["Sex"] = 1
            params["sex"] = 1.0
            params["Gender"] = 1.0
            params["gender"] = 1.0
            updated_keys.extend(["Sex", "sex", "Gender", "gender"])
        elif re.search(female_terms, text_norm) and not re.search(male_terms, text_norm):
            params["Sex"] = 0
            params["sex"] = 0.0
            params["Gender"] = 0.0
            params["gender"] = 0.0
            updated_keys.extend(["Sex", "sex", "Gender", "gender"])

        # 3. Height, Weight, and BMI
        height_match = re.search(r'(?:\b(?:height|tall|h)\b|लंबाई|ऊंचाई|कद|উচ্চতা|உயரம்|ఎత్తు|ઊંચાઈ|ਕੱਦ|قد)\s*(?:is|of|:|=)?\s*(\d{2,3}(?:\.\d+)?)\s*(?:cm|centimeters|meters|m|सेमी|সেন্টিমিটার)?', text_norm)
        # Feet/inches format (e.g. 5 feet 8 inches, 5'8")
        ft_match = re.search(r'\b(\d)\s*(?:feet|ft|\')\s*(\d{1,2})?\s*(?:inches|in|\")?\b', text_norm)
        weight_match = re.search(r'(?:\b(?:weight|weigh|w)\b|वजन|ওজন|எடை|బరువు|વજન|ਵਜ਼ਨ|وزن)\s*(?:is|of|:|=)?\s*(\d{2,3}(?:\.\d+)?)\s*(?:kg|kilos|kilograms|pounds|lbs|किलो|केज़ी|কেজি|কেজী)?', text_norm)
        bmi_match = re.search(r'(?:\b(?:bmi|body mass index)\b|बीएमआई|বিএমআই)\s*(?:is|of|:|=)?\s*(\d{2}(?:\.\d+)?)\b', text_norm)

        # Contextual single number fallbacks
        if not height_match and not ft_match and active_question_id == "q3_height":
            h_val = re.search(r'\b(\d{2,3}(?:\.\d+)?)\b', text_norm)
            if h_val:
                val = float(h_val.group(1))
                if 50 <= val <= 250:
                    params["height"] = val
                    updated_keys.append("height")

        if not weight_match and active_question_id == "q4_weight":
            w_val = re.search(r'\b(\d{2,3}(?:\.\d+)?)\b', text_norm)
            if w_val:
                val = float(w_val.group(1))
                if 25 <= val <= 300:
                    params["weight"] = val
                    updated_keys.append("weight")

        if height_match:
            val = float(height_match.group(1))
            if val < 3.0: # given in meters like 1.75
                val = val * 100.0
            params["height"] = val
            updated_keys.append("height")
        elif ft_match:
            feet = float(ft_match.group(1))
            inches = float(ft_match.group(2) or 0)
            total_cm = (feet * 12 + inches) * 2.54
            params["height"] = round(total_cm, 1)
            updated_keys.append("height")

        if weight_match:
            val = float(weight_match.group(1))
            if "lbs" in text_norm or "pounds" in text_norm:
                val = val * 0.453592
            params["weight"] = round(val, 1)
            updated_keys.append("weight")

        if bmi_match:
            bmi_val = float(bmi_match.group(1))
            params["BMI"] = bmi_val
            params["bmi"] = bmi_val
            updated_keys.extend(["BMI", "bmi"])

        # Automatically calculate BMI if both height and weight are available
        cur_h = params.get("height")
        cur_w = params.get("weight")
        if cur_h and cur_w and cur_h > 0:
            h_m = cur_h / 100.0
            calc_bmi = round(cur_w / (h_m * h_m), 1)
            params["BMI"] = calc_bmi
            params["bmi"] = calc_bmi
            updated_keys.extend(["BMI", "bmi"])

        # 4. Family History
        fam_keywords = r'(?:family history|family|parents|father|mother|genetic|वंशानुगत|पारिवारिक|বংশগত|পারিবারিক)'
        if active_question_id == "q5_family_history":
            if re.search(r'\b(?:yes|yeah|sure|have|positive|हाँ|हां|হ্যাঁ)\b', text_norm):
                params["family_history"] = 1
                params["fam_hist"] = 1.0
                updated_keys.extend(["family_history", "fam_hist"])
            elif re.search(negation_terms, text_norm):
                params["family_history"] = 0
                params["fam_hist"] = 0.0
                updated_keys.extend(["family_history", "fam_hist"])
        elif re.search(fam_keywords, text_norm):
            is_neg = bool(re.search(f"{negation_terms}.{{0,20}}{fam_keywords}", text_norm))
            val = 0 if is_neg else 1
            params["family_history"] = val
            updated_keys.append("family_history")

        # 5. High Blood Pressure & BP Readings
        indic_bp = r'(?:बीपी|उच्च रक्तचाप|हाई बीपी|हाई ब्लड प्रेशर|ब्लड प्रेशर|উচ্চ রক্তচাপ|হাই ব্লাড প্রেসার|ব্লাড প্রেসার|রক্তচাপ)'
        english_bp = r'\b(?:high blood pressure|high bp|blood pressure high|bp high|bp\s*(?:is\s*)?high|blood pressure\s*(?:is\s*)?high|hypertension|hypertensive|elevated bp)\b'
        bp_keywords = f"(?:{english_bp}|{indic_bp})"

        # Contextual yes/no for high BP question
        if active_question_id == "q6_high_bp":
            if re.search(r'\b(?:yes|yeah|sure|have|positive|हाँ|हां|হ্যাঁ|আছে)\b', text_norm) and not re.search(negation_terms, text_norm):
                params["HighBP"] = 1
                params["hypertension"] = 1.0
                params["htn"] = 1.0
                updated_keys.extend(["HighBP", "hypertension", "htn"])
            elif re.search(negation_terms, text_norm):
                params["HighBP"] = 0
                params["hypertension"] = 0.0
                params["htn"] = 0.0
                updated_keys.extend(["HighBP", "hypertension", "htn"])
        elif re.search(bp_keywords, text_norm):
            is_negated = bool(re.search(f"{negation_terms}.{{0,20}}{bp_keywords}", text_norm) or re.search(f"{bp_keywords}.{{0,20}}{negation_terms}", text_norm))
            val = 0 if is_negated else 1
            params["HighBP"] = val
            params["hypertension"] = float(val)
            params["htn"] = float(val)
            updated_keys.extend(["HighBP", "hypertension", "htn"])

        # BP Reading (120/80, 130 over 85, etc.)
        bp_val_match = re.search(r'(?:(?:bp|blood pressure|बीपी|বিপি)\s*(?:is|of|:|=)?\s*)?(\d{2,3})\s*(?:/|over|\s*-\s*|\s+)\s*(\d{2,3})\b', text_norm)
        if bp_val_match:
            sbp = float(bp_val_match.group(1))
            dbp = float(bp_val_match.group(2))
            if 70 <= sbp <= 250 and 40 <= dbp <= 150:
                params["systolic_bp"] = sbp
                params["diastolic_bp"] = dbp
                params["trestbps"] = sbp
                params["bp"] = dbp
                if sbp >= 130 or dbp >= 85:
                    params["HighBP"] = 1
                    params["hypertension"] = 1.0
                    params["htn"] = 1.0
                updated_keys.extend(["systolic_bp", "diastolic_bp", "trestbps", "bp", "HighBP", "hypertension", "htn"])

        # 6. Heart Rate / Pulse
        hr_match = re.search(r'(?:\b(?:heart rate|pulse|hr|bpm)\b|हृदय गति|नाड़ी|হৃদস্পন্দন)\s*(?:is|of|:|=)?\s*(\d{2,3})\b', text_norm)
        if not hr_match and active_question_id == "q8_heart_rate":
            hr_single = re.search(r'\b(\d{2,3})\b', text_norm)
            if hr_single and 40 <= float(hr_single.group(1)) <= 220:
                hr_match = hr_single

        if hr_match:
            hr_val = float(hr_match.group(1))
            if 40 <= hr_val <= 220:
                params["heart_rate"] = hr_val
                params["thalach"] = hr_val
                updated_keys.extend(["heart_rate", "thalach"])

        # 7. Blood Glucose / Blood Sugar
        glucose_match = re.search(r'(?:\b(?:blood sugar|glucose|fasting glucose|sugar|bgr)\b|शुगर|सुगर|ग्लूकोज|ব্লাড সুগার|গ্লুকোজ)\s*(?:is|of|:|=)?\s*(\d{2,3})\b', text_norm)
        if not glucose_match and active_question_id == "q9_blood_glucose":
            g_single = re.search(r'\b(\d{2,3})\b', text_norm)
            if g_single and 40 <= float(g_single.group(1)) <= 500:
                glucose_match = g_single

        if glucose_match:
            gval = float(glucose_match.group(1))
            if 40 <= gval <= 500:
                params["avg_glucose_level"] = gval
                params["bgr"] = gval
                params["fasting_glucose"] = gval
                if gval >= 126:
                    params["dm"] = 1.0
                    params["fbs"] = 1.0
                else:
                    params["fbs"] = 0.0
                updated_keys.extend(["avg_glucose_level", "bgr", "fasting_glucose", "fbs"])

        # 8. SpO2 / Oxygen Saturation
        spo2_match = re.search(r'(?:\b(?:spo2|oxygen saturation|oxygen level|o2)\b|ऑक्सीजन|অক্সিজেন)\s*(?:is|of|:|=)?\s*(\d{2,3})\b', text_norm)
        if not spo2_match and active_question_id == "q10_spo2":
            s_single = re.search(r'\b(\d{2,3})\b', text_norm)
            if s_single and 70 <= float(s_single.group(1)) <= 100:
                spo2_match = s_single

        if spo2_match:
            s_val = float(spo2_match.group(1))
            if 70 <= s_val <= 100:
                params["spo2"] = s_val
                params["oxygen_saturation"] = s_val
                updated_keys.extend(["spo2", "oxygen_saturation"])

        # 9. Waist Circumference
        waist_match = re.search(r'(?:\b(?:waist|waist circumference)\b|कमर|কোমর)\s*(?:is|of|:|=)?\s*(\d{2,3}(?:\.\d+)?)\s*(?:inches|in|cm|सेमी)?', text_norm)
        if not waist_match and active_question_id == "q11_waist":
            w_single = re.search(r'\b(\d{2,3}(?:\.\d+)?)\b', text_norm)
            if w_single and 20 <= float(w_single.group(1)) <= 180:
                waist_match = w_single

        if waist_match:
            w_val = float(waist_match.group(1))
            if "cm" in text_norm:
                params["waist_circumference"] = w_val
            else: # assume inches or direct
                params["waist_circumference"] = round(w_val * 2.54 if w_val < 50 else w_val, 1)
            updated_keys.append("waist_circumference")

        # 10. Physical Activity & Exercise Frequency
        exercise_keywords = r'(?:exercise|workout|physical activity|walk daily|active|gym|व्यायाम|कसरत|सक्रिय|ব্যায়াম|শরীরচর্চা)'
        if active_question_id in ["q12_physical_activity", "q13_exercise_freq"]:
            if re.search(r'\b(?:yes|daily|regularly|often|very|active|हाँ|हां|হ্যাঁ|নিয়মিত|প্রতিদিন)\b', text_norm) and not re.search(negation_terms, text_norm):
                params["PhysActivity"] = 1
                params["physical_activity_hours"] = 3.5
                updated_keys.extend(["PhysActivity", "physical_activity_hours"])
            elif re.search(negation_terms, text_norm) or re.search(r'\b(?:sedentary|rarely|never|कम|একদম না)\b', text_norm):
                params["PhysActivity"] = 0
                params["physical_activity_hours"] = 0.5
                updated_keys.extend(["PhysActivity", "physical_activity_hours"])
        elif re.search(exercise_keywords, text_norm):
            is_neg = bool(re.search(f"{negation_terms}.{{0,20}}{exercise_keywords}", text_norm))
            val = 0 if is_neg else 1
            params["PhysActivity"] = val
            params["physical_activity_hours"] = 3.5 if val == 1 else 0.5
            updated_keys.extend(["PhysActivity", "physical_activity_hours"])

        # 11. Sedentary / Screen Hours
        sedentary_match = re.search(r'\b(\d{1,2})\s*(?:hours|hrs|घंटे|ঘণ্টা)\b', text_norm)
        if (sedentary_match and active_question_id == "q14_sedentary_hours") or re.search(r'(?:sitting|screen|बैठकर|বসে)', text_norm):
            s_hours = float(sedentary_match.group(1)) if sedentary_match else 6.0
            params["sedentary_hours"] = s_hours
            if s_hours >= 8:
                params["DiffWalk"] = params.get("DiffWalk", 0)
            updated_keys.append("sedentary_hours")

        # 12. Smoking & Tobacco
        smoke_keywords = r'(?:smoke|smoker|smoking|cigarettes|tobacco|धूम्रपान|सिगरेट|तंबाकू|ধূমপান|তামাক)'
        if active_question_id == "q15_smoker":
            if re.search(r'\b(?:yes|yeah|sure|smoke|cigarettes|tobacco|हाँ|हां|হ্যাঁ)\b', text_norm) and not re.search(negation_terms, text_norm):
                params["Smoker"] = 1
                params["smoking_status"] = 1.0
                params["smoking_history"] = 1
                updated_keys.extend(["Smoker", "smoking_status", "smoking_history"])
            elif re.search(negation_terms, text_norm):
                params["Smoker"] = 0
                params["smoking_status"] = 0.0
                params["smoking_history"] = 0
                updated_keys.extend(["Smoker", "smoking_status", "smoking_history"])
        elif re.search(smoke_keywords, text_norm):
            is_negated = bool(re.search(f"{negation_terms}.{{0,20}}{smoke_keywords}", text_norm))
            val = 0 if is_negated else 1
            params["Smoker"] = val
            params["smoking_status"] = float(val)
            params["smoking_history"] = val
            updated_keys.extend(["Smoker", "smoking_status", "smoking_history"])

        # 13. Alcohol Consumption
        alcohol_keywords = r'(?:alcohol|beer|wine|liquor|drinks|drinking|शराब|मदिरा|অ্যালকোহল|মদ)'
        if active_question_id == "q16_alcohol":
            if re.search(r'\b(?:yes|yeah|regularly|often|daily|हाँ|हां|হ্যাঁ)\b', text_norm) and not re.search(negation_terms, text_norm):
                params["HvyAlcoholConsump"] = 1
                params["alcohol"] = 1.0
                updated_keys.extend(["HvyAlcoholConsump", "alcohol"])
            elif re.search(negation_terms, text_norm) or re.search(r'\b(?:never|rarely|occasionally|no)\b', text_norm):
                params["HvyAlcoholConsump"] = 0
                params["alcohol"] = 0.0
                updated_keys.extend(["HvyAlcoholConsump", "alcohol"])
        elif re.search(alcohol_keywords, text_norm):
            is_neg = bool(re.search(f"{negation_terms}.{{0,20}}{alcohol_keywords}", text_norm))
            val = 0 if is_neg else 1
            params["HvyAlcoholConsump"] = val
            params["alcohol"] = float(val)
            updated_keys.extend(["HvyAlcoholConsump", "alcohol"])

        # 14. Water Intake
        water_match = re.search(r'\b(\d{1,2}(?:\.\d+)?)\s*(?:liters|litres|l|glasses|लीटर|গ্লাস)\b', text_norm)
        if active_question_id == "q17_water_intake" and not water_match:
            water_match = re.search(r'\b(\d{1,2}(?:\.\d+)?)\b', text_norm)
        if water_match and (active_question_id == "q17_water_intake" or "water" in text_norm or "पानी" in text_norm):
            params["water_intake"] = float(water_match.group(1))
            updated_keys.append("water_intake")

        # 15. Diet: Junk Food, Fruits & Vegetables
        if active_question_id == "q18_junk_food":
            if re.search(r'\b(?:often|daily|frequently|a lot|yes|ज्यादा|अक्सर|বেশি|প্রায়ই)\b', text_norm):
                params["junk_food_freq"] = 1
                params["GenHlth"] = max(params.get("GenHlth", 2), 3)
                updated_keys.extend(["junk_food_freq", "GenHlth"])
            elif re.search(negation_terms, text_norm) or re.search(r'\b(?:rarely|never|hardly|कम|খুব কম)\b', text_norm):
                params["junk_food_freq"] = 0
                updated_keys.append("junk_food_freq")

        fruit_veg_keywords = r'(?:fruits|vegetables|veggies|फल|सब्जियां|সবজি|ফলমূল)'
        if active_question_id == "q19_fruits_veggies" or re.search(fruit_veg_keywords, text_norm):
            if re.search(r'\b(?:yes|daily|regularly|often|हाँ|हां|হ্যাঁ|প্রতিদিন)\b', text_norm) and not re.search(negation_terms, text_norm):
                params["Fruits"] = 1
                params["Veggies"] = 1
                updated_keys.extend(["Fruits", "Veggies"])
            elif re.search(negation_terms, text_norm) or re.search(r'\b(?:rarely|never|कम|না)\b', text_norm):
                params["Fruits"] = 0
                params["Veggies"] = 0
                updated_keys.extend(["Fruits", "Veggies"])

        # 16. Sleep Hours, Quality, Regularity, Snoring
        sleep_hours_match = re.search(r'\b(\d{1,2}(?:\.\d+)?)\s*(?:hours|hrs|घंटे|ঘণ্টা)?\b', text_norm)
        if active_question_id == "q20_sleep_hours" and sleep_hours_match:
            s_val = float(sleep_hours_match.group(1))
            if 2 <= s_val <= 18:
                params["sleep_hours"] = s_val
                updated_keys.append("sleep_hours")

        if active_question_id == "q21_sleep_quality":
            if re.search(r'\b(?:poor|bad|disturbed|insomnia|खराब|कम|খারাপ)\b', text_norm):
                params["sleep_quality"] = 0
                params["MentHlth"] = 5.0
                updated_keys.extend(["sleep_quality", "MentHlth"])
            elif re.search(r'\b(?:good|great|sound|well|अच्छी|ভালো)\b', text_norm):
                params["sleep_quality"] = 1
                params["MentHlth"] = 0.0
                updated_keys.extend(["sleep_quality", "MentHlth"])

        if active_question_id == "q22_sleep_regular":
            val = 0 if re.search(negation_terms, text_norm) else 1
            params["sleep_regular"] = val
            updated_keys.append("sleep_regular")

        snoring_keywords = r'(?:snore|snoring|খররাটা|खर्राटे|নাক ডাকা)'
        if active_question_id == "q23_snoring" or re.search(snoring_keywords, text_norm):
            is_pos = bool(re.search(r'\b(?:yes|yeah|often|loudly|हाँ|हां|হ্যাঁ)\b', text_norm) and not re.search(negation_terms, text_norm))
            params["snoring"] = 1 if is_pos else 0
            updated_keys.append("snoring")

        # 17. Metabolic Symptoms: Thirst & Urination
        thirst_keywords = r'(?:thirst|urination|urinate|frequent urination|प्यास|पेशाब|তৃষ্ণা|প্রস্রাব)'
        if active_question_id == "q24_thirst_urination" or re.search(thirst_keywords, text_norm):
            is_pos = bool(re.search(r'\b(?:yes|yeah|frequently|often|always|हाँ|हां|হ্যাঁ|বেশি|बार-बार)\b', text_norm) and not re.search(negation_terms, text_norm))
            params["thirst_urination"] = 1 if is_pos else 0
            if is_pos:
                params["dm"] = 1.0
            updated_keys.extend(["thirst_urination"])

        # 18. Unexplained Weight Changes
        if active_question_id == "q25_weight_changes" or re.search(r'(?:weight change|weight loss|वजन बदलना|ওজন পরিবর্তন)', text_norm):
            val = 0 if re.search(negation_terms, text_norm) else 1
            params["unexplained_weight_change"] = val
            updated_keys.append("unexplained_weight_change")

        # 19. Persistent Fatigue
        fatigue_keywords = r'(?:fatigue|tired|exhausted|weakness|कमजोरी|थकान|দুর্বলতা|ক্লান্তি)'
        if active_question_id == "q26_fatigue" or re.search(fatigue_keywords, text_norm):
            is_pos = bool(re.search(r'\b(?:yes|yeah|always|often|very|हाँ|हां|হ্যাঁ|অনেক)\b', text_norm) and not re.search(negation_terms, text_norm))
            params["fatigue"] = 1 if is_pos else 0
            if is_pos:
                params["PhysHlth"] = max(params.get("PhysHlth", 0.0), 5.0)
            updated_keys.extend(["fatigue", "PhysHlth"])

        # 20. Shortness of Breath / Dyspnea
        sob_keywords = r'(?:shortness of breath|breathless|dyspnea|breathing problem|सांस फूलना|सांस की तकलीफ|শ্বাসকষ্ট)'
        if active_question_id == "q27_shortness_breath" or re.search(sob_keywords, text_norm):
            val = 0 if re.search(negation_terms, text_norm) else 1
            params["dyspnea"] = val
            params["shortness_of_breath"] = val
            updated_keys.extend(["dyspnea", "shortness_of_breath"])

        # 21. Chest Discomfort / Angina / Chest Pain
        chest_keywords = r'(?:chest pain|chest discomfort|tightness|angina|सीने में दर्द|सीने में तकलीफ|বুকে ব্যথা)'
        if active_question_id == "q28_chest_discomfort" or re.search(chest_keywords, text_norm):
            val = 0 if re.search(negation_terms, text_norm) else 1
            params["cp"] = 1.0 if val == 1 else 0.0
            params["chest_discomfort"] = val
            updated_keys.extend(["cp", "chest_discomfort"])

        # 22. Persistent Cough
        cough_keywords = r'(?:cough|persistent cough|chronic cough|खांसी|কাশি)'
        if active_question_id == "q29_cough" or re.search(cough_keywords, text_norm):
            val = 0 if re.search(negation_terms, text_norm) else 1
            params["persistent_cough"] = val
            params["cough"] = val
            updated_keys.extend(["persistent_cough", "cough"])

        # 23. Cholesterol
        chol_keywords = r'(?:\b(?:high cholesterol|elevated cholesterol|hypercholesterolemia|high chol|cholesterol)\b|कोलेस्ट्रॉल|হাই কোলেস্টেরল|কোলেস্টেরল)'
        if re.search(chol_keywords, text_norm):
            is_negated = bool(re.search(f"{negation_terms}.{{0,20}}{chol_keywords}", text_norm))
            val = 0 if is_negated else 1
            params["HighChol"] = val
            params["CholCheck"] = 1
            updated_keys.extend(["HighChol", "CholCheck"])

        chol_val_match = re.search(r'(?:\b(?:cholesterol|chol)\b|कोलेस्ट्रॉल|কোলেস্টেরল)\s*(?:is|of)?\s*(\d{2,3})\b', text_norm)
        if chol_val_match:
            cval = float(chol_val_match.group(1))
            params["chol"] = cval
            params["HighChol"] = 1 if cval >= 200 else 0
            params["CholCheck"] = 1
            updated_keys.extend(["chol", "HighChol", "CholCheck"])

        # 24. Stroke History
        stroke_keywords = r'\b(?:stroke|cerebrovascular|स्ट्रोक|পক্ষাঘাত|பக்கவாதம்)\b'
        if re.search(stroke_keywords, text_norm):
            is_negated = bool(re.search(f"{negation_terms}.{{0,20}}{stroke_keywords}", text_norm))
            val = 0 if is_negated else 1
            params["Stroke"] = val
            updated_keys.append("Stroke")

        unique_updated = sorted(list(set(updated_keys)))
        return params, unique_updated
