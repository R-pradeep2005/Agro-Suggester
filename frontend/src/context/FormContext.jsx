import React, { createContext, useState, useContext } from 'react';

const FormContext = createContext();

export const FormProvider = ({ children }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [formData, setFormData] = useState({
    geolocation: null,
    plannedDate: '',
    prevCropName: '',
    prevCropYieldValue: '',
    prevCropYieldUnit: 'kg',
    prevFertilizerName: '',
    prevFertilizerAmountValue: '',
    prevFertilizerAmountUnit: 'kg',
    harvestDate: '',
    residueStatus: '' // 'left' or 'disposed'
  });

  const updateFormData = (key, value) => {
    setFormData(prev => ({ ...prev, [key]: value }));
  };

  const nextStep = () => setCurrentStep(prev => prev + 1);
  const prevStep = () => setCurrentStep(prev => Math.max(0, prev - 1));
  const jumpToStep = (stepIndex) => setCurrentStep(stepIndex);

  return (
    <FormContext.Provider value={{
      currentStep,
      formData,
      updateFormData,
      nextStep,
      prevStep,
      jumpToStep
    }}>
      {children}
    </FormContext.Provider>
  );
};

export const useFormContext = () => {
  const context = useContext(FormContext);
  if (!context) {
    throw new Error('useFormContext must be used within a FormProvider');
  }
  return context;
};
