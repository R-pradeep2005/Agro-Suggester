import React from 'react';
import { useTranslation } from 'react-i18next';
import TTSButton from './TTSButton';
import { Leaf } from 'lucide-react';

const Navbar = ({ currentInstructionText }) => {
  const { i18n, t } = useTranslation();
  const [isPlaying, setIsPlaying] = React.useState(false);

  React.useEffect(() => {
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
    <nav className="navbar">
      <div style={{ display: 'flex', alignItems: 'center', gap: '1.25rem' }}>
        {/* Logo icon */}
        <div style={{
          background: 'linear-gradient(135deg, #00d68f 0%, #06d6d6 100%)',
          borderRadius: '10px',
          padding: '0.45rem',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          boxShadow: '0 4px 16px rgba(0,214,143,0.35)',
          color: '#001a0f',
        }}>
          <Leaf size={18} strokeWidth={2.5} />
        </div>
        <h1>{t('appTitle')}</h1>
        <div className="lang-switch">
          <button className={`lang-btn ${i18n.language === 'en' ? 'active' : ''}`} onClick={() => changeLanguage('en')}>
            {t('language_en')}
          </button>
          <button className={`lang-btn ${i18n.language === 'ta' ? 'active' : ''}`} onClick={() => changeLanguage('ta')}>
            {t('language_ta')}
          </button>
          <button className={`lang-btn ${i18n.language === 'hi' ? 'active' : ''}`} onClick={() => changeLanguage('hi')}>
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
