import React, { useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { Sprout } from 'lucide-react';
import TTSButton from './TTSButton';

const Navbar = ({ currentInstructionText }) => {
  const { i18n, t } = useTranslation();
  const [isPlaying, setIsPlaying] = React.useState(false);

  useEffect(() => {
    document.title = t('appTitle');
  }, [i18n.language, t]);

  const changeLanguage = (lng) => {
    i18n.changeLanguage(lng);
    if (window.speechSynthesis.speaking) {
      window.speechSynthesis.cancel();
      setIsPlaying(false);
    }
  };

  return (
    <nav className="navbar glass-panel">
      <div className="navbar-brand">
        <div style={{
          width: '36px', height: '36px',
          background: 'linear-gradient(135deg, #10b981, #059669)',
          borderRadius: '10px',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          boxShadow: '0 4px 10px rgba(16,185,129,0.25)'
        }}>
          <Sprout size={20} color="white" />
        </div>
        <h1 style={{ 
          fontSize: '1.25rem', 
          fontWeight: '700', 
          background: 'linear-gradient(135deg, #059669, #0d9488)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          margin: 0
        }}>
          {t('appTitle')}
        </h1>
      </div>

      <div className="navbar-actions">
        <div className="lang-switch">
          <button 
            className={`lang-btn ${i18n.language === 'en' ? 'active' : ''}`}
            onClick={() => changeLanguage('en')}
          >
            {t('language_en')}
          </button>
          <button 
            className={`lang-btn ${i18n.language === 'ta' ? 'active' : ''}`}
            onClick={() => changeLanguage('ta')}
          >
            {t('language_ta')}
          </button>
          <button 
            className={`lang-btn ${i18n.language === 'hi' ? 'active' : ''}`}
            onClick={() => changeLanguage('hi')}
          >
            {t('language_hi')}
          </button>
        </div>
        
        <TTSButton 
          textToRead={currentInstructionText} 
          isPlaying={isPlaying} 
          setIsPlaying={setIsPlaying} 
        />
      </div>
    </nav>
  );
};

export default Navbar;
