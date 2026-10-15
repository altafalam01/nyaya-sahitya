"""UI text in English and Hindi."""

LANG_CHOICES = {"auto": "Auto", "en": "English", "hi": "हिन्दी"}

TEXT = {
    "en": {
        "title": "Nyaya Sahitya",
        "subtitle": "Answers from Indian legal sources, with citations.",
        "disclaimer": "Legal information only, not legal advice. Consult a qualified advocate.",
        "placeholder": "Ask about the Constitution, BNS, BNSS, BSA or a judgment…",
        "language": "Answer language",
        "sources": "Sources",
        "not_grounded": "Not found in the indexed sources",
        "repealed": "not in force",
        "clear": "Clear conversation",
        "thinking": "Searching the sources…",
        "err_unreachable": "Cannot reach the answer service. Start the API (uvicorn api.app:app) and try again.",
        "err_unconfigured": "The answer service is not configured (missing API key).",
        "err_invalid": "Please enter a question of 3 to 1000 characters.",
        "err_server": "Something went wrong while answering. Please try again.",
    },
    "hi": {
        "title": "न्याय साहित्य",
        "subtitle": "भारतीय कानूनी स्रोतों से, उद्धरण सहित उत्तर।",
        "disclaimer": "केवल कानूनी जानकारी, कानूनी सलाह नहीं। किसी योग्य अधिवक्ता से परामर्श लें।",
        "placeholder": "संविधान, BNS, BNSS, BSA या किसी निर्णय के बारे में पूछें…",
        "language": "उत्तर की भाषा",
        "sources": "स्रोत",
        "not_grounded": "इंडेक्स किए गए स्रोतों में नहीं मिला",
        "repealed": "लागू नहीं",
        "clear": "बातचीत साफ़ करें",
        "thinking": "स्रोत खोजे जा रहे हैं…",
        "err_unreachable": "उत्तर सेवा से संपर्क नहीं हो पा रहा। API (uvicorn api.app:app) चालू करके फिर कोशिश करें।",
        "err_unconfigured": "उत्तर सेवा कॉन्फ़िगर नहीं है (API कुंजी नहीं मिली)।",
        "err_invalid": "कृपया 3 से 1000 अक्षरों का प्रश्न लिखें।",
        "err_server": "उत्तर देते समय कुछ गड़बड़ हुई। कृपया फिर कोशिश करें।",
    },
}


def t(lang: str, key: str) -> str:
    return TEXT.get(lang, TEXT["en"])[key]
