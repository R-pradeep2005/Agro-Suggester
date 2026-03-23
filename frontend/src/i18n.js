import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';

// the translations
const resources = {
  en: {
    translation: {
      "appTitle": "AgroData Collection",
      "language_en": "English",
      "language_ta": "தமிழ்",
      "language_hi": "हिन्दी",
      "next": "Next",
      "back": "Back",
      "submit": "Submit",
      "edit": "Edit",
      "reviewStep": "Final Review",
      "reviewDesc": "Please review your details before submitting.",
      
      // Questions
      "geolocation": "Please provide your farm's location.",
      "plannedDate": "When do you plan to start cultivation?",
      "prevCropName": "What was the previous crop grown?",
      "prevCropYield": "What was the yield of the previous crop?",
      "prevFertilizerName": "Which fertilizer did you use previously?",
      "prevFertilizerAmount": "How much fertilizer was used?",
      "harvestDate": "When was the previous crop harvested?",
      "residueStatus": "What is the residue status?",
      
      // Options/Helpers
      "getLocation": "Get Location",
      "locationCaptured": "Location Captured!",
      "selectCrop": "Select Crop...",
      "selectFertilizer": "Select Fertilizer...",
      "leftInLand": "Left in land",
      "disposedElsewhere": "Disposed elsewhere",
      "kg": "kg",
      "tons": "tons",
      "liters": "liters"
    }
  },
  ta: {
    translation: {
      "appTitle": "வேளாண் தரவு சேகரிப்பு",
      "language_en": "English",
      "language_ta": "தமிழ்",
      "language_hi": "हिन्दी",
      "next": "அடுத்து",
      "back": "பின்னே",
      "submit": "சமர்ப்பி",
      "edit": "திருத்து",
      "reviewStep": "இறுதி சரிபார்ப்பு",
      "reviewDesc": "சமர்ப்பிப்பதற்கு முன் உங்கள் விவரங்களைச் சரிபார்க்கவும்.",
      
      "geolocation": "உங்கள் பண்ணையின் இருப்பிடத்தை வழங்கவும்.",
      "plannedDate": "நீங்கள் எப்போது சாகுபடியைத் தொடங்கத் திட்டமிட்டுள்ளீர்கள்?",
      "prevCropName": "முன்பு பயிரிடப்பட்ட பயிர் என்ன?",
      "prevCropYield": "முந்தைய பயிரின் மகசூல் எவ்வளவு?",
      "prevFertilizerName": "நீங்கள் முன்பு எந்த உரத்தைப் பயன்படுத்தினீர்கள்?",
      "prevFertilizerAmount": "எவ்வளவு உரம் பயன்படுத்தப்பட்டது?",
      "harvestDate": "முந்தைய பயிர் எப்போது அறுவடை செய்யப்பட்டது?",
      "residueStatus": "பயிர் எச்சத்தின் நிலை என்ன?",
      
      "getLocation": "இருப்பிடத்தைப் பெறுங்கள்",
      "locationCaptured": "இருப்பிடம் பெறப்பட்டது!",
      "selectCrop": "பயிரைத் தேர்ந்தெடுக்கவும்...",
      "selectFertilizer": "உரத்தைத் தேர்ந்தெடுக்கவும்...",
      "leftInLand": "நிலத்தில் விடப்பட்டது",
      "disposedElsewhere": "வேறு இடத்தில் அகற்றப்பட்டது",
      "kg": "கிலோ",
      "tons": "டன்கள்",
      "liters": "லிட்டர்"
    }
  },
  hi: {
    translation: {
      "appTitle": "कृषि डेटा संग्रह",
      "language_en": "English",
      "language_ta": "தமிழ்",
      "language_hi": "हिन्दी",
      "next": "अगला",
      "back": "वापस",
      "submit": "जमा करें",
      "edit": "संपादित करें",
      "reviewStep": "अंतिम समीक्षा",
      "reviewDesc": "सबमिट करने से पहले कृपया अपने विवरण की समीक्षा करें।",
      
      "geolocation": "कृपया अपने खेत का स्थान प्रदान करें।",
      "plannedDate": "आप खेती कब शुरू करने की योजना बना रहे हैं?",
      "prevCropName": "पिछली फसल कौन सी बोई गई थी?",
      "prevCropYield": "पिछली फसल की उपज कितनी थी?",
      "prevFertilizerName": "आपने पहले किस उर्वरक का उपयोग किया था?",
      "prevFertilizerAmount": "कितनी खाद का उपयोग किया गया था?",
      "harvestDate": "पिछली फसल की कटाई कब की गई थी?",
      "residueStatus": "फसल के अवशेष की स्थिति क्या है?",
      
      "getLocation": "स्थान प्राप्त करें",
      "locationCaptured": "स्थान कैप्चर किया गया!",
      "selectCrop": "फसल चुनें...",
      "selectFertilizer": "उर्वरक चुनें...",
      "leftInLand": "जमीन में छोड़ दिया",
      "disposedElsewhere": "कहीं और निपटाया गया",
      "kg": "किलो",
      "tons": "टन",
      "liters": "लीटर"
    }
  }
};

i18n
  .use(initReactI18next) // passes i18n down to react-i18next
  .init({
    resources,
    lng: "en", // default language
    fallbackLng: "en",
    interpolation: {
      escapeValue: false // react already safes from xss
    }
  });

export default i18n;
