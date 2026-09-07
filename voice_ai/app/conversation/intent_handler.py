import re
from typing import Optional, Tuple

class ConversationalIntentHandler:
    """
    Parses conversational control commands and special patient intents across
    English, Hindi, Bengali, and mixed dialects.
    """

    COMMAND_PATTERNS = {
        "skip": [
            r'\b(?:skip|next|pass|skip this|skip question|move on)\b',
            r'\b(?:छोड़ें|अगला|आगे बढ़ें|स्किप|छोड़िए)\b',
            r'\b(?:পরের প্রশ্ন|পরবর্তী|বাদ দিন|স্কিপ|এগিয়ে যান)\b'
        ],
        "back": [
            r'\b(?:go back|back|previous|last question|return)\b',
            r'\b(?:पीछे|पिछला|वापस|पहले वाला)\b',
            r'\b(?:আগেরটা|পূর্ববর্তী|পেছনে|আগের প্রশ্ন)\b'
        ],
        "repeat": [
            r'\b(?:repeat|say again|pardon|what was that|once more|say it again)\b',
            r'\b(?:दोहराएं|फिर से|फिर से बोलिए|एक बार फिर|दोबारा)\b',
            r'\b(?:আবার বলুন|পুনরায়|আরেকবার|একবার বলুন)\b'
        ],
        "change": [
            r'\b(?:change my answer|change answer|correct my answer|correction|actually|mistake)\b',
            r'\b(?:उत्तर बदलें|बदलना है|गलती हो गई|दरअसल|असल में)\b',
            r'\b(?:উত্তর পরিবর্তন|সংশোধন|আসলে|ভুল হয়েছে)\b'
        ],
        "stop": [
            r'\b(?:stop|pause|halt|quit|cancel|exit|end assessment)\b',
            r'\b(?:रुकिए|रोकें|रुकें|बंद करें|समाप्त)\b',
            r'\b(?:থামুন|বন্ধ করুন|বিরতি|বাতিল)\b'
        ],
        "dont_know": [
            r'\b(?:i don\'t know|dont know|not sure|no idea|unknown|unsure|cannot say|can\'t say)\b',
            r'\b(?:पता नहीं|मालूम नहीं|याद नहीं|जानकारी नहीं|नहीं पता)\b',
            r'\b(?:জানিনা|জানা নেই|মনে নেই|বলতে পারছি না|অজানা)\b'
        ]
    }

    @classmethod
    def detect_command(cls, user_text: str) -> Optional[str]:
        """
        Detects if user input is an explicit conversational control command.
        Returns one of: 'skip', 'back', 'repeat', 'change', 'stop', 'dont_know', or None.
        """
        if not user_text:
            return None
        text = user_text.strip().lower()

        for cmd, patterns in cls.COMMAND_PATTERNS.items():
            for pat in patterns:
                if re.search(pat, text, re.IGNORECASE):
                    return cmd
        return None

    @classmethod
    def is_affirmative(cls, text: str) -> bool:
        """Checks if input indicates a clear yes/affirmative response."""
        norm = text.lower()
        aff_patterns = [
            r'\b(?:yes|yeah|yep|yup|sure|correct|true|positive|definitely|i do|i have|affirmative)\b',
            r'\b(?:हाँ|हां|जी हाँ|जी हां|अवश्य|बिल्कुल|होता है|है)\b',
            r'\b(?:হ্যাঁ|হ্যা|নিশ্চয়ই|অবশ্যই|আছে|হাঁ)\b'
        ]
        return any(re.search(p, norm) for p in aff_patterns)

    @classmethod
    def is_negative(cls, text: str) -> bool:
        """Checks if input indicates a clear no/negative response."""
        norm = text.lower()
        neg_patterns = [
            r'\b(?:no|nope|nah|never|not at all|negative|none|neither|don\'t|dont|i do not|i don\'t)\b',
            r'\b(?:नहीं|ना|कभी नहीं|बिल्कुल नहीं|नाही|नहीं है)\b',
            r'\b(?:না|নয়|কখনো না|একদম না|নেই|নাই)\b'
        ]
        return any(re.search(p, norm) for p in neg_patterns)
