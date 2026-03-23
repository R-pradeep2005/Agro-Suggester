import React from 'react';
import { Volume2 } from 'lucide-react';

const TTSButton = ({ textToRead, isPlaying, setIsPlaying }) => {
  const handleSpeak = () => {
    if (!textToRead) return;

    // if already speaking, stop
    if (window.speechSynthesis.speaking) {
      window.speechSynthesis.cancel();
      setIsPlaying(false);
      return;
    }

    const utterance = new SpeechSynthesisUtterance(textToRead);
    
    // Optional: map current document language to a voice if available
    const currentLang = localStorage.getItem('i18nextLng') || 'en';
    if (currentLang.startsWith('ta')) utterance.lang = 'ta-IN';
    else if (currentLang.startsWith('hi')) utterance.lang = 'hi-IN';
    else utterance.lang = 'en-US';

    utterance.onstart = () => setIsPlaying(true);
    utterance.onend = () => setIsPlaying(false);
    utterance.onerror = () => setIsPlaying(false);

    window.speechSynthesis.speak(utterance);
  };

  return (
    <button 
      onClick={handleSpeak}
      className={`btn ${isPlaying ? 'btn-primary' : 'btn-secondary'}`}
      style={{ padding: '0.5rem', borderRadius: '50%' }}
      aria-label="Read Aloud"
      title="Read Aloud"
    >
      <Volume2 size={24} color={isPlaying ? '#fff' : 'var(--text-main)'} />
    </button>
  );
};

export default TTSButton;
