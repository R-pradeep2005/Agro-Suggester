import React, { useState } from 'react';
import Navbar from './components/Navbar';
import ConversationalForm from './components/ConversationalForm';
import { FormProvider } from './context/FormContext';
import './index.css';

function App() {
  const [currentInstructionText, setCurrentInstructionText] = useState('');

  return (
    <FormProvider>
      <div className="app-layout">
        <Navbar currentInstructionText={currentInstructionText} />
        <main>
          <ConversationalForm onQuestionChange={setCurrentInstructionText} />
        </main>
      </div>
    </FormProvider>
  );
}

export default App;
