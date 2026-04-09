import React, { useState } from 'react';
import Navbar from './components/Navbar';
import ConversationalForm from './components/ConversationalForm';
import Dashboard from './components/Dashboard';
import { FormProvider, useFormContext } from './context/FormContext';
import './index.css';

const STAGES = [
  { icon: '📡', label: 'Connecting to services…' },
  { icon: '🌍', label: 'Fetching location soil data from SoilGrids…' },
  { icon: '🌦️', label: 'Fetching real-time weather data…' },
  { icon: '⚙️', label: 'Preparing feature vector…' },
  { icon: '🤖', label: 'Running AI crop models…' },
  { icon: '📊', label: 'Calculating SHAP explainability…' },
  { icon: '✅', label: 'Finalizing recommendations…' },
];

const LoadingScreen = () => {
  const [stepIdx, setStepIdx] = useState(0);

  React.useEffect(() => {
    const intervals = [800, 3000, 2500, 1200, 4000, 2000, 1500];
    let current = 0;
    const advance = () => {
      current++;
      if (current < STAGES.length) {
        setStepIdx(current);
        setTimeout(advance, intervals[current] ?? 1500);
      }
    };
    const t = setTimeout(advance, intervals[0]);
    return () => clearTimeout(t);
  }, []);

  return (
    <div style={{
      display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
      minHeight: '70vh', gap: '2rem'
    }}>
      <div style={{
        width: '72px', height: '72px', borderRadius: '50%',
        border: '5px solid rgba(16,185,129,0.15)',
        borderTop: '5px solid var(--primary-color)',
        animation: 'spin 0.9s linear infinite'
      }} />

      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem', width: '340px' }}>
        {STAGES.map((stage, i) => {
          const done = i < stepIdx;
          const active = i === stepIdx;
          return (
            <div key={i} style={{
              display: 'flex', alignItems: 'center', gap: '0.75rem',
              padding: '0.5rem 0.75rem', borderRadius: '8px',
              background: active ? 'rgba(16,185,129,0.1)' : 'transparent',
              transition: 'background 0.3s',
              opacity: done || active ? 1 : 0.3
            }}>
              <span style={{ fontSize: '1.1rem' }}>{done ? '✅' : stage.icon}</span>
              <span style={{
                fontSize: '0.88rem', fontWeight: active ? 600 : 400,
                color: active ? 'var(--primary-hover)' : 'var(--text-muted)'
              }}>
                {stage.label}
              </span>
              {active && (
                <span style={{
                  marginLeft: 'auto', fontSize: '0.8rem',
                  color: 'var(--primary-color)',
                  animation: 'pulse 1s ease-in-out infinite'
                }}>●</span>
              )}
            </div>
          );
        })}
      </div>

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.2; } }
      `}</style>
    </div>
  );
};

const AppContent = () => {
  const [currentInstructionText, setCurrentInstructionText] = useState('');
  const { isSubmitted, isLoading } = useFormContext();

  return (
    <div className="app-layout">
      <Navbar currentInstructionText={isSubmitted ? 'Your AI Prediction Dashboard is ready.' : currentInstructionText} />
      <main>
        {isLoading && !isSubmitted ? (
          <LoadingScreen />
        ) : isSubmitted ? (
          <Dashboard />
        ) : (
          <ConversationalForm onQuestionChange={setCurrentInstructionText} />
        )}
      </main>
    </div>
  );
};

function App() {
  return (
    <FormProvider>
      <AppContent />
    </FormProvider>
  );
}

export default App;
