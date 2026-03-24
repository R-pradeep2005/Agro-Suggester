import React, { createContext, useState, useContext } from 'react';

const FormContext = createContext();

export const FormProvider = ({ children }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [isSubmitted, setIsSubmitted] = useState(false);
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

  const resetForm = () => {
    setCurrentStep(0);
    setIsSubmitted(false);
    setFormData({
      geolocation: null,
      plannedDate: '',
      prevCropName: '',
      prevCropYieldValue: '',
      prevCropYieldUnit: 'kg',
      prevFertilizerName: '',
      prevFertilizerAmountValue: '',
      prevFertilizerAmountUnit: 'kg',
      harvestDate: '',
      residueStatus: '' 
    });
  };

  return (
    <FormContext.Provider value={{
      currentStep,
      isSubmitted,
      setIsSubmitted,
      formData,
      updateFormData,
      nextStep,
      prevStep,
      jumpToStep,
      resetForm
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
