import React, { useState } from 'react';
import Navbar from './components/Navbar';
import ConversationalForm from './components/ConversationalForm';
import Dashboard from './components/Dashboard';
import { FormProvider, useFormContext } from './context/FormContext';
import './index.css';

const AppContent = () => {
  const [currentInstructionText, setCurrentInstructionText] = useState('');
  const { isSubmitted } = useFormContext();

  return (
    <div className="app-layout">
      <Navbar currentInstructionText={isSubmitted ? 'Your AI Prediction Dashboard is ready.' : currentInstructionText} />
      <main>
        {isSubmitted ? (
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
