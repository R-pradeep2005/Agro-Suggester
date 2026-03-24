import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useFormContext } from '../context/FormContext';
import { MapPin } from 'lucide-react';

const StepWrapper = ({ title, children, canProceed, onNext }) => {
  const { t } = useTranslation();
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', height: '100%' }}>
      <h2 style={{ fontSize: '1.5rem', fontWeight: '600', color: 'var(--primary-hover)' }}>{title}</h2>
      <div style={{ flex: 1 }}>{children}</div>
      <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '1rem' }}>
        {onNext && (
          <button 
            className="btn btn-primary" 
            onClick={onNext} 
            disabled={!canProceed}
          >
            {t('next')}
          </button>
        )}
      </div>
    </div>
  );
};

// 1. Geolocation
export const GeolocationStep = () => {
  const { t } = useTranslation();
  const { formData, updateFormData, nextStep } = useFormContext();
  const [loading, setLoading] = useState(false);

  const handleGetLocation = () => {
    setLoading(true);
    if ("geolocation" in navigator) {
      navigator.geolocation.getCurrentPosition((pos) => {
        updateFormData('geolocation', `${pos.coords.latitude}, ${pos.coords.longitude}`);
        setLoading(false);
      }, () => setLoading(false));
    } else {
      setLoading(false);
    }
  };

  return (
    <StepWrapper 
      title={t('geolocation')} 
      canProceed={!!formData.geolocation} 
      onNext={nextStep}
    >
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem', marginTop: '2rem' }}>
        <button className="btn btn-secondary" onClick={handleGetLocation} disabled={loading}>
          <MapPin size={20} />
          {loading ? '...' : formData.geolocation ? t('locationCaptured') : t('getLocation')}
        </button>
        {formData.geolocation && <p style={{ color: 'var(--primary-color)' }}>{formData.geolocation}</p>}
      </div>
    </StepWrapper>
  );
};

// 2. Cultivation Planned Date
export const PlannedDateStep = () => {
  const { t } = useTranslation();
  const { formData, updateFormData, nextStep } = useFormContext();

  return (
    <StepWrapper 
      title={t('plannedDate')} 
      canProceed={!!formData.plannedDate} 
      onNext={nextStep}
    >
      <input 
        type="date" 
        className="input-base" 
        value={formData.plannedDate} 
        onChange={(e) => updateFormData('plannedDate', e.target.value)} 
      />
    </StepWrapper>
  );
};

// 3. Previous Crop Name
export const PrevCropNameStep = () => {
  const { t } = useTranslation();
  const { formData, updateFormData, nextStep } = useFormContext();

  // Mock exhaustive crop list, but simply using text input with datalist to simulate searchable dropdown
  const crops = ["Rice", "Wheat", "Maize", "Cotton", "Sugarcane"];

  return (
    <StepWrapper 
      title={t('prevCropName')} 
      canProceed={!!formData.prevCropName} 
      onNext={nextStep}
    >
      <input 
        list="crops"
        className="input-base" 
        placeholder={t('selectCrop')}
        value={formData.prevCropName} 
        onChange={(e) => updateFormData('prevCropName', e.target.value)} 
      />
      <datalist id="crops">
        {crops.map(c => <option key={c} value={c} />)}
      </datalist>
    </StepWrapper>
  );
};

// 4. Previous Crop Yield
export const PrevCropYieldStep = () => {
  const { t } = useTranslation();
  const { formData, updateFormData, nextStep } = useFormContext();

  return (
    <StepWrapper 
      title={t('prevCropYield')} 
      canProceed={!!formData.prevCropYieldValue} 
      onNext={nextStep}
    >
      <div style={{ display: 'flex', gap: '1rem' }}>
        <input 
          type="number" 
          className="input-base" 
          style={{ flex: 2 }}
          value={formData.prevCropYieldValue} 
          onChange={(e) => updateFormData('prevCropYieldValue', e.target.value)} 
        />
        <select 
          className="input-base" 
          style={{ flex: 1 }}
          value={formData.prevCropYieldUnit}
          onChange={(e) => updateFormData('prevCropYieldUnit', e.target.value)}
        >
          <option value="kg">{t('kg')}</option>
          <option value="tons">{t('tons')}</option>
        </select>
      </div>
    </StepWrapper>
  );
};

// 5. Previous Fertilizer Name
export const PrevFertilizerStep = () => {
  const { t } = useTranslation();
  const { formData, updateFormData, nextStep } = useFormContext();
  const fertilizers = ["Urea", "DAP", "Compost", "NPK", "None"];

  return (
    <StepWrapper 
      title={t('prevFertilizerName')} 
      canProceed={!!formData.prevFertilizerName} 
      onNext={nextStep}
    >
      <select 
        className="input-base" 
        value={formData.prevFertilizerName} 
        onChange={(e) => updateFormData('prevFertilizerName', e.target.value)}
      >
        <option value="" disabled>{t('selectFertilizer')}</option>
        {fertilizers.map(f => <option key={f} value={f}>{f}</option>)}
      </select>
    </StepWrapper>
  );
};

// 6. Previous Fertilizer Amount
export const PrevFertilizerAmountStep = () => {
  const { t } = useTranslation();
  const { formData, updateFormData, nextStep } = useFormContext();

  return (
    <StepWrapper 
      title={t('prevFertilizerAmount')} 
      canProceed={!!formData.prevFertilizerAmountValue} 
      onNext={nextStep}
    >
      <div style={{ display: 'flex', gap: '1rem' }}>
        <input 
          type="number" 
          className="input-base" 
          style={{ flex: 2 }}
          value={formData.prevFertilizerAmountValue} 
          onChange={(e) => updateFormData('prevFertilizerAmountValue', e.target.value)} 
        />
        <select 
          className="input-base" 
          style={{ flex: 1 }}
          value={formData.prevFertilizerAmountUnit}
          onChange={(e) => updateFormData('prevFertilizerAmountUnit', e.target.value)}
        >
          <option value="kg">{t('kg')}</option>
          <option value="liters">{t('liters')}</option>
        </select>
      </div>
    </StepWrapper>
  );
};

// 7. Harvest Date
export const HarvestDateStep = () => {
  const { t } = useTranslation();
  const { formData, updateFormData, nextStep } = useFormContext();

  return (
    <StepWrapper 
      title={t('harvestDate')} 
      canProceed={!!formData.harvestDate} 
      onNext={nextStep}
    >
      <input 
        type="date" 
        className="input-base" 
        value={formData.harvestDate} 
        onChange={(e) => updateFormData('harvestDate', e.target.value)} 
      />
    </StepWrapper>
  );
};

// 8. Residue Status
export const ResidueStatusStep = () => {
  const { t } = useTranslation();
  const { formData, updateFormData, nextStep } = useFormContext();

  const handleToggle = (val) => {
    updateFormData('residueStatus', val);
  };

  return (
    <StepWrapper 
      title={t('residueStatus')} 
      canProceed={!!formData.residueStatus} 
      onNext={nextStep}
    >
      <div style={{ display: 'flex', gap: '1rem', marginTop: '1rem' }}>
        <button 
          className={`btn ${formData.residueStatus === 'left' ? 'btn-primary' : 'btn-secondary'}`}
          onClick={() => handleToggle('left')}
          style={{ flex: 1 }}
        >
          {t('leftInLand')}
        </button>
        <button 
          className={`btn ${formData.residueStatus === 'disposed' ? 'btn-primary' : 'btn-secondary'}`}
          onClick={() => handleToggle('disposed')}
          style={{ flex: 1 }}
        >
          {t('disposedElsewhere')}
        </button>
      </div>
    </StepWrapper>
  );
};

// 9. Final Review Step
export const ReviewStep = () => {
  const { t } = useTranslation();
  const { formData, jumpToStep, setIsSubmitted } = useFormContext();

  const handleEdit = (stepIdx) => {
    jumpToStep(stepIdx);
  };

  const Row = ({ label, value, stepIdx }) => (
    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.75rem 0', borderBottom: '1px solid #e2e8f0' }}>
      <div>
        <span style={{ fontWeight: 500, color: 'var(--text-muted)', display: 'block', fontSize: '0.85rem' }}>{label}</span>
        <span style={{ fontSize: '1.1rem' }}>{value || '-'}</span>
      </div>
      <button className="btn btn-secondary" style={{ padding: '0.25rem 0.75rem', fontSize: '0.85rem' }} onClick={() => handleEdit(stepIdx)}>
        {t('edit')}
      </button>
    </div>
  );

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', height: '100%' }}>
      <h2 style={{ fontSize: '1.5rem', fontWeight: '600', color: 'var(--primary-hover)' }}>{t('reviewStep')}</h2>
      <p style={{ color: 'var(--text-muted)' }}>{t('reviewDesc')}</p>
      
      <div style={{ flex: 1, overflowY: 'auto', paddingRight: '0.5rem' }}>
        <Row label={t('geolocation')} value={formData.geolocation} stepIdx={0} />
        <Row label={t('plannedDate')} value={formData.plannedDate} stepIdx={1} />
        <Row label={t('prevCropName')} value={formData.prevCropName} stepIdx={2} />
        <Row label={t('prevCropYield')} value={`${formData.prevCropYieldValue} ${t(formData.prevCropYieldUnit)}`} stepIdx={3} />
        <Row label={t('prevFertilizerName')} value={formData.prevFertilizerName} stepIdx={4} />
        <Row label={t('prevFertilizerAmount')} value={`${formData.prevFertilizerAmountValue} ${t(formData.prevFertilizerAmountUnit)}`} stepIdx={5} />
        <Row label={t('harvestDate')} value={formData.harvestDate} stepIdx={6} />
        <Row label={t('residueStatus')} value={t(formData.residueStatus === 'left' ? 'leftInLand' : formData.residueStatus === 'disposed' ? 'disposedElsewhere' : '')} stepIdx={7} />
      </div>

      <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '1rem' }}>
        <button className="btn btn-primary" onClick={() => setIsSubmitted(true)}>
          {t('submit')}
        </button>
      </div>
    </div>
  );
};

export const FormStepsList = [
  GeolocationStep,
  PlannedDateStep,
  PrevCropNameStep,
  PrevCropYieldStep,
  PrevFertilizerStep,
  PrevFertilizerAmountStep,
  HarvestDateStep,
  ResidueStatusStep,
  ReviewStep
];
