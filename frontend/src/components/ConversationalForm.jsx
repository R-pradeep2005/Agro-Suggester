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
  const progressPercentage = (currentStep / (totalSteps - 1)) * 100;

  return (
    <div className="form-screen-bg">
      {/* Step chip */}
      <div className="form-step-chip">
        Step {currentStep + 1} <span style={{ color: 'rgba(0,214,143,0.45)' }}>/ {totalSteps}</span>
      </div>

      <div className="form-card" style={{ width: '100%', maxWidth: '780px' }}>
        {/* Progress Bar */}
        <div className="progress-track">
          <div className="progress-track-fill" style={{ width: `${progressPercentage}%` }} />
        </div>

        <div id="step-content" style={{ flex: 1, position: 'relative' }}>
          <AnimatePresence mode="wait">
            <motion.div
              key={currentStep}
              initial={{ opacity: 0, y: 18, filter: 'blur(4px)' }}
              animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
              exit={{ opacity: 0, y: -14, filter: 'blur(4px)' }}
              transition={{ duration: 0.32, ease: [0.16, 1, 0.3, 1] }}
              style={{ width: '100%', height: '100%' }}
            >
              {handleStepRender()}
            </motion.div>
          </AnimatePresence>
        </div>

        {/* Back Button */}
        {currentStep > 0 && currentStep < totalSteps - 1 && (
          <div style={{ marginTop: '2rem' }}>
            <button className="btn btn-secondary" onClick={prevStep}>
              ← {t('back')}
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default ConversationalForm;
