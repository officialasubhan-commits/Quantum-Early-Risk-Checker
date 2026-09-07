import requests
from typing import Dict, Any, Optional
from voice_ai.app.config import config
from voice_ai.app.schemas.voice_schemas import MEDICAL_DISCLAIMER_TEXT
from voice_ai.app.languages.language_manager import LanguageManager, LOCALIZED_PROMPTS

class LocalLLMManager:
    """
    Manages local LLM inference via Ollama / llama.cpp or local clinical dialogue engine.
    Supports multilingual empathetic explanations across 11 target languages.
    100% self-hosted, 0 cloud dependencies.
    """
    def __init__(self, ollama_url: Optional[str] = None, model_name: Optional[str] = None):
        self.ollama_url = ollama_url or config.OLLAMA_URL
        self.model_name = model_name or config.OLLAMA_MODEL

    def generate_chat_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Queries local Ollama API if running; otherwise uses local rule-guided medical engine.
        """
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
            if system_prompt:
                payload["system"] = system_prompt

            resp = requests.post(f"{self.ollama_url}/api/generate", json=payload, timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("response", "").strip()
        except Exception:
            pass

        return ""

    def format_patient_risk_explanation(self, prediction_result: Dict[str, Any], disease_name: str = "Diabetes", language: str = "en") -> str:
        """
        Generates patient-friendly, empathetic clinical risk explanation in the user's active language.
        Preserves critical technical accuracy (e.g. Hybrid Classical-Quantum Ensemble model).
        """
        lang = (language or "en").lower().split("-")[0]

        prob = prediction_result.get("risk_probability", 0.0)
        risk_pct = round(prob * 100, 1)
        is_high = prediction_result.get("is_high_risk", False) or prob >= 0.5
        model_used = prediction_result.get("model_used", "Hybrid Classical-Quantum Ensemble")

        # 1. Try Local Ollama LLM with language-tuned prompt
        system_prompt = (
            f"You are an empathetic medical assistant communicating in language '{lang}'. "
            "Explain the disease risk prediction results clearly to the patient. "
            "Preserve technical terms like 'Hybrid Classical-Quantum Ensemble', 'BMI', and exact risk percentage accurately. "
            "Do NOT provide a formal medical diagnosis."
        )

        user_prompt = (
            f"Disease: {disease_name}\n"
            f"Model: {model_used}\n"
            f"Calculated Risk Score: {risk_pct}%\n"
            f"Risk Status: {'Elevated Risk' if is_high else 'Low Risk'}\n\n"
            f"Synthesize a friendly 2-sentence response in language '{lang}' explaining this score."
        )

        llm_reply = self.generate_chat_response(user_prompt, system_prompt)
        if llm_reply and len(llm_reply) > 10:
            return llm_reply

        # 2. Rule-Guided Localized Synthesizer (Works 100% offline without LLM latency)
        risk_key = "risk_high" if is_high else "risk_low"
        main_msg = LanguageManager.get_prompt_template(lang, risk_key).format(disease=disease_name, risk=risk_pct)
        rec_msg = LanguageManager.get_prompt_template(lang, "recommendation")

        return f"{main_msg} {rec_msg}"
