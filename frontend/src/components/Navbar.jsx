import React from 'react';
import { useTranslation } from 'react-i18next';
import TTSButton from './TTSButton';

const Navbar = ({ currentInstructionText }) => {
  const { i18n, t } = useTranslation();
  const [isPlaying, setIsPlaying] = React.useState(false);

  const changeLanguage = (lng) => {
    i18n.changeLanguage(lng);
    if (window.speechSynthesis.speaking) {
      window.speechSynthesis.cancel();
      setIsPlaying(false);
    }
  };

  return (
    <nav className="navbar glass-panel">
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
        <h1 style={{ fontSize: '1.25rem', fontWeight: '700', color: 'var(--primary-hover)' }}>
          {t('appTitle')}
        </h1>
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
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
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
