import React, { useEffect } from 'react';
import { Volume2 } from 'lucide-react';

const TTSButton = ({ textToRead, isPlaying, setIsPlaying }) => {
  // Cleanup on unmount
  useEffect(() => {
    return () => {
      window.speechSynthesis.cancel();
      if (window.currentAudio) {
        window.currentAudio.pause();
        window.currentAudio = null;
      }
      window.currentAudioPlayingFlag = false;
    };
  }, []);

  const handleSpeak = async () => {
    // 1. Stop any current speech
    window.speechSynthesis.cancel();
    if (window.currentAudio) {
      window.currentAudio.pause();
      window.currentAudio = null;
    }
    
    // Toggle off if currently playing
    if (isPlaying || window.currentAudioPlayingFlag) {
      setIsPlaying(false);
      window.currentAudioPlayingFlag = false;
      return;
    }

    // 2. Determine what to read
    let finalContentToRead = textToRead || '';
    const container = document.getElementById('step-content');
    if (container && container.innerText) {
      finalContentToRead = container.innerText;
    }

    if (!finalContentToRead) return;

    // 3. Determine Language
    const currentLang = localStorage.getItem('i18nextLng') || 'en';
    let langCode = 'en';
    if (currentLang.startsWith('ta')) langCode = 'ta';
    else if (currentLang.startsWith('hi')) langCode = 'hi';

    // Set state
    setIsPlaying(true);
    window.currentAudioPlayingFlag = true;

    // 4. Split text into 150-char chunks to bypass Google Translate TTS limits, respecting word boundaries
    // Split by newlines first to keep logical sentences together normally
    const lines = finalContentToRead.split('\n').filter(l => l.trim().length > 0);
    const chunks = [];
    
    for (const line of lines) {
      if (line.length < 150) {
        chunks.push(line);
      } else {
        const words = line.split(' ');
        let currentChunk = '';
        for (const word of words) {
          if ((currentChunk + word).length < 150) {
            currentChunk += word + ' ';
          } else {
            chunks.push(currentChunk.trim());
            currentChunk = word + ' ';
          }
        }
        if (currentChunk) chunks.push(currentChunk.trim());
      }
    }

    // 5. Play sequentially
    for (let i = 0; i < chunks.length; i++) {
      if (!window.currentAudioPlayingFlag) break; // Check if user canceled mid-way

      const chunk = chunks[i];
      const url = `https://translate.googleapis.com/translate_tts?ie=UTF-8&q=${encodeURIComponent(chunk)}&tl=${langCode}&client=tw-ob`;
      
      try {
        await new Promise((resolve, reject) => {
          const audio = new Audio(url);
          window.currentAudio = audio;
          
          audio.onended = resolve;
          audio.onerror = reject; // Failed network
          
          audio.play().catch(reject);
        });
      } catch (error) {
        console.error("Cloud TTS failed, falling back to native TTS", error);
        
        // Final Fallback: use the robotic native OS API if Google blocks/fails
        const utterance = new SpeechSynthesisUtterance(chunk);
        utterance.lang = langCode === 'ta' ? 'ta-IN' : langCode === 'hi' ? 'hi-IN' : 'en-US';
        
        await new Promise((resolve) => {
          utterance.onend = resolve;
          utterance.onerror = resolve;
          window.speechSynthesis.speak(utterance);
        });
      }
    }

    // Done reading all chunks
    setIsPlaying(false);
    window.currentAudioPlayingFlag = false;
    window.currentAudio = null;
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
