import React, { useState } from 'react';
import Navbar from './components/Navbar';
import ConversationalForm from './components/ConversationalForm';
import Dashboard from './components/Dashboard';
import LandingPage from './components/LandingPage';
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
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      minHeight: 'calc(100vh - 62px)',
      gap: '2.5rem',
      padding: '2rem',
    }}>
      {/* Dual counter-rotating rings */}
      <div style={{ position: 'relative', width: '72px', height: '72px' }}>
        <div style={{
          position: 'absolute', inset: 0,
          borderRadius: '50%',
          border: '2px solid rgba(0,214,143,0.1)',
          borderTop: '2px solid #00d68f',
          animation: 'spin-icon 0.9s linear infinite',
        }} />
        <div style={{
          position: 'absolute', inset: '10px',
          borderRadius: '50%',
          border: '2px solid rgba(6,214,214,0.1)',
          borderBottom: '2px solid #06d6d6',
          animation: 'spin-icon 1.3s linear infinite reverse',
        }} />
        <div style={{
          position: 'absolute', inset: '22px',
          borderRadius: '50%',
          background: 'rgba(0,214,143,0.1)',
          boxShadow: '0 0 20px rgba(0,214,143,0.3)',
        }} />
      </div>

      {/* Stage card */}
      <div style={{
        background: 'rgba(14,20,25,0.90)',
        backdropFilter: 'blur(32px)',
        WebkitBackdropFilter: 'blur(32px)',
        border: '1px solid rgba(255,255,255,0.08)',
        borderTop: '1px solid rgba(255,255,255,0.12)',
        borderRadius: '24px',
        padding: '2.5rem 3rem',
        minWidth: '380px',
        maxWidth: '480px',
        boxShadow: '0 24px 80px rgba(0,0,0,0.6), 0 0 0 1px rgba(0,214,143,0.05)',
        position: 'relative',
        overflow: 'hidden',
      }}>
        {/* top glow */}
        <div style={{
          position: 'absolute', top: '-40px', right: '-40px',
          width: '120px', height: '120px',
          background: 'radial-gradient(circle, rgba(0,214,143,0.1), transparent 65%)',
          pointerEvents: 'none',
        }} />

        <p style={{
          fontFamily: '"Space Grotesk", sans-serif',
          fontSize: '0.75rem',
          fontWeight: 800,
          color: 'rgba(0,214,143,0.6)',
          textTransform: 'uppercase',
          letterSpacing: '0.15em',
          marginBottom: '1.25rem',
        }}>
          Processing
        </p>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
          {STAGES.map((stage, i) => {
            const done = i < stepIdx;
            const active = i === stepIdx;
            return (
              <div key={i} style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.85rem',
                padding: '0.65rem 0.85rem',
                borderRadius: '12px',
                background: active ? 'rgba(0,214,143,0.06)' : 'transparent',
                border: `1px solid ${active ? 'rgba(0,214,143,0.2)' : 'transparent'}`,
                transition: 'all 0.4s ease',
                opacity: done || active ? 1 : 0.28,
              }}>
                <span style={{ fontSize: '1rem', width: '22px', textAlign: 'center' }}>
                  {done ? '✓' : stage.icon}
                </span>
                <span style={{
                  fontFamily: '"Plus Jakarta Sans", sans-serif',
                  fontSize: '0.9rem',
                  fontWeight: active ? 700 : 500,
                  color: active ? '#00d68f' : done ? 'rgba(255,255,255,0.5)' : 'rgba(255,255,255,0.25)',
                  flex: 1,
                }}>
                  {stage.label}
                </span>
                {active && (
                  <div style={{
                    width: '7px', height: '7px', borderRadius: '50%',
                    background: '#00d68f',
                    boxShadow: '0 0 8px rgba(0,214,143,0.8)',
                    animation: 'pulse-dot 1s ease-in-out infinite',
                    flexShrink: 0,
                  }} />
                )}
              </div>
            );
          })}
        </div>
      </div>

      <style>{`
        @keyframes pulse-dot {
          0%, 100% { opacity: 1; transform: scale(1); }
          50%       { opacity: 0.3; transform: scale(0.65); }
        }
      `}</style>
    </div>
  );
};

const AppContent = () => {
  const [currentInstructionText, setCurrentInstructionText] = useState('');
  const [hasStarted, setHasStarted] = useState(false);
  const { isSubmitted, isLoading } = useFormContext();

  if (!hasStarted) {
    return <LandingPage onGetStarted={() => setHasStarted(true)} />;
  }

  return (
    <div>
      <Navbar currentInstructionText={isSubmitted ? 'Your AI Prediction Dashboard is ready.' : currentInstructionText} />
      <main>
        {isLoading && !isSubmitted    ? <LoadingScreen /> :
         isSubmitted                  ? <Dashboard /> :
                                        <ConversationalForm onQuestionChange={setCurrentInstructionText} />
        }
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
