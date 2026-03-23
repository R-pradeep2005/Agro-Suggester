import React from 'react';
import { useTranslation } from 'react-i18next';
import { useFormContext } from '../context/FormContext';
import { FormStepsList } from './FormSteps';
import { motion, AnimatePresence } from 'framer-motion';

const ConversationalForm = ({ onQuestionChange }) => {
  const { t } = useTranslation();
  const { currentStep, prevStep } = useFormContext();

  const handleStepRender = () => {
    const StepComponent = FormStepsList[currentStep];
    if (!StepComponent) return null;
    return <StepComponent />;
  };

  // Extract the specific question text from the current step to pass to the TTS engine via App.js
  React.useEffect(() => {
    const keys = [
      'geolocation', 'plannedDate', 'prevCropName', 'prevCropYield',
      'prevFertilizerName', 'prevFertilizerAmount', 'harvestDate', 'residueStatus', 'reviewDesc'
    ];
    if (keys[currentStep]) {
      onQuestionChange(t(keys[currentStep]));
    }
  }, [currentStep, t, onQuestionChange]);

  const totalSteps = FormStepsList.length;
  const progressPercentage = ((currentStep) / (totalSteps - 1)) * 100;

  return (
    <div className="container" style={{ marginTop: '2rem' }}>
      <div className="glass-panel" style={{ padding: '2rem', minHeight: '400px', display: 'flex', flexDirection: 'column' }}>
        
        {/* Progress Bar */}
        <div style={{ width: '100%', height: '8px', backgroundColor: '#e2e8f0', borderRadius: '4px', marginBottom: '2rem', overflow: 'hidden' }}>
          <div 
            style={{ 
              height: '100%', 
              backgroundColor: 'var(--primary-color)', 
              width: `${progressPercentage}%`,
              transition: 'width 0.4s ease-out'
            }} 
          />
        </div>

        <div style={{ flex: 1, position: 'relative' }}>
          <AnimatePresence mode="wait">
            <motion.div
              key={currentStep}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.3 }}
              style={{ width: '100%', height: '100%' }}
            >
              {handleStepRender()}
            </motion.div>
          </AnimatePresence>
        </div>
        
        {/* Back Button (Next is handled inside the steps usually) */}
        {currentStep > 0 && currentStep < totalSteps - 1 && (
          <div style={{ marginTop: '2rem', display: 'flex', justifyContent: 'flex-start' }}>
            <button className="btn btn-secondary" onClick={prevStep}>
              {t('back')}
            </button>
          </div>
        )}

      </div>
    </div>
  );
};

export default ConversationalForm;
