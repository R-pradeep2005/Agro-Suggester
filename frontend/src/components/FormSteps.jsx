import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useFormContext } from '../context/FormContext';
import { MapPin } from 'lucide-react';
import AnimatedSelect from './AnimatedSelect';

const StepWrapper = ({ title, children, canProceed, onNext }) => {
  const { t } = useTranslation();
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', height: '100%' }}>
      <h2 style={{ fontSize: '1.5rem', fontWeight: '600', color: 'var(--primary-hover)' }}>{title}</h2>
      <div style={{ flex: 1 }}>{children}</div>
      <div className="step-buttons">
        {onNext && (
          <button 
            className="btn btn-primary" 
            onClick={onNext} 
            disabled={!canProceed}
            style={{ width: window.innerWidth < 640 ? '100%' : 'auto' }}
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

  const [lat, setLat] = useState(formData.geolocation?.split(',')[0]?.trim() || '');
  const [lon, setLon] = useState(formData.geolocation?.split(',')[1]?.trim() || '');

  const handleGetLocation = () => {
    setLoading(true);
    if ("geolocation" in navigator) {
      navigator.geolocation.getCurrentPosition((pos) => {
        const newLat = pos.coords.latitude.toFixed(6);
        const newLon = pos.coords.longitude.toFixed(6);
        setLat(newLat);
        setLon(newLon);
        updateFormData('geolocation', `${newLat}, ${newLon}`);
        setLoading(false);
      }, () => setLoading(false));
    } else {
      setLoading(false);
    }
  };

  const handleManualChange = (field, val) => {
    if (field === 'lat') {
      setLat(val);
      updateFormData('geolocation', `${val}, ${lon}`);
    } else {
      setLon(val);
      updateFormData('geolocation', `${lat}, ${val}`);
    }
  };

  return (
    <StepWrapper 
      title={t('geolocation')} 
      canProceed={!!formData.geolocation && lat && lon} 
      onNext={nextStep}
    >
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1.5rem', marginTop: '1rem' }}>
        <button className="btn btn-secondary" onClick={handleGetLocation} disabled={loading} style={{ width: '100%' }}>
          <MapPin size={20} />
          {loading ? '...' : formData.geolocation ? t('locationCaptured') : t('getLocation')}
        </button>

        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', width: '100%' }}>
          <div style={{ flex: 1, height: '1px', background: '#e2e8f0' }}></div>
          <span style={{ color: 'var(--text-muted)', fontSize: '0.9rem', fontWeight: '500' }}>{t('or')}</span>
          <div style={{ flex: 1, height: '1px', background: '#e2e8f0' }}></div>
        </div>

        <div style={{ width: '100%', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <h3 style={{ fontSize: '1rem', fontWeight: '600', color: 'var(--text-main)' }}>{t('manualInput')}</h3>
          <div style={{ display: 'flex', gap: '1rem' }}>
            <div style={{ flex: 1 }}>
              <label style={{ display: 'block', fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>{t('latitude')}</label>
              <input 
                type="number" 
                step="any"
                className="input-base" 
                placeholder="0.000000"
                value={lat} 
                onChange={(e) => handleManualChange('lat', e.target.value)} 
              />
            </div>
            <div style={{ flex: 1 }}>
              <label style={{ display: 'block', fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>{t('longitude')}</label>
              <input 
                type="number" 
                step="any"
                className="input-base" 
                placeholder="0.000000"
                value={lon} 
                onChange={(e) => handleManualChange('lon', e.target.value)} 
              />
            </div>
          </div>
        </div>
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
      <AnimatedSelect
        value={formData.prevCropName}
        onChange={(val) => updateFormData('prevCropName', val)}
        options={crops.map(c => ({ value: c, label: c }))}
        placeholder={t('selectCrop')}
      />
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
      <div className="flex-responsive">
        <input 
          type="number" 
          className="input-base" 
          style={{ flex: 2 }}
          value={formData.prevCropYieldValue} 
          onChange={(e) => updateFormData('prevCropYieldValue', e.target.value)} 
        />
        <div style={{ flex: 1 }}>
          <AnimatedSelect
            value={formData.prevCropYieldUnit}
            onChange={(val) => updateFormData('prevCropYieldUnit', val)}
            options={[
              { value: 'kg', label: t('kg') },
              { value: 'tons', label: t('tons') }
            ]}
          />
        </div>
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
      <AnimatedSelect
        value={formData.prevFertilizerName}
        onChange={(val) => updateFormData('prevFertilizerName', val)}
        options={fertilizers.map(f => ({ value: f, label: f }))}
        placeholder={t('selectFertilizer')}
      />
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
      <div className="flex-responsive">
        <input 
          type="number" 
          className="input-base" 
          style={{ flex: 2 }}
          value={formData.prevFertilizerAmountValue} 
          onChange={(e) => updateFormData('prevFertilizerAmountValue', e.target.value)} 
        />
        <div style={{ flex: 1 }}>
          <AnimatedSelect
            value={formData.prevFertilizerAmountUnit}
            onChange={(val) => updateFormData('prevFertilizerAmountUnit', val)}
            options={[
              { value: 'kg', label: t('kg') },
              { value: 'liters', label: t('liters') }
            ]}
          />
        </div>
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
      <div className="flex-responsive" style={{ marginTop: '1rem' }}>
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
  const { formData, jumpToStep, setIsSubmitted, setIsLoading, setApiResult } = useFormContext();
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async () => {
    setIsSubmitting(true);
    setIsLoading(true);
    try {
      const lat = parseFloat(formData.geolocation?.split(',')[0]) || 0;
      const lon = parseFloat(formData.geolocation?.split(',')[1]) || 0;
      
      const fertName = formData.prevFertilizerName || '';
      const fertAmt = parseFloat(formData.prevFertilizerAmountValue) || 0;
      
      let n = 0, p = 0, k = 0;
      if (fertName.includes('Urea')) { n = fertAmt * 0.46; }
      else if (fertName.includes('DAP')) { n = fertAmt * 0.18; p = fertAmt * 0.46; }
      else if (fertName.includes('NPK')) { n = fertAmt * 0.19; p = fertAmt * 0.19; k = fertAmt * 0.19; }
      
      const payload = {
        lat: lat,
        lon: lon,
        previous_crop: formData.prevCropName || 'None',
        previous_fertilizer_n: n,
        previous_fertilizer_p: p,
        previous_fertilizer_k: k
      };

      const apiUrl = import.meta.env.VITE_API_GATEWAY || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/input_prep/api/prepare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      
      if (response.ok) {
        const data = await response.json();
        console.log("API response:", data);
        setApiResult(data);
        setIsSubmitted(true);
      } else {
        const errText = await response.text();
        console.error("API error:", errText);
        alert("Prediction failed: " + errText);
      }
    } catch (e) {
      console.error("Submit error:", e);
      alert("Error reaching the backend services. Please ensure they are running.");
    } finally {
      setIsSubmitting(false);
      setIsLoading(false);
    }
  };

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

      <div className="step-buttons">
        <button 
          className="btn btn-primary" 
          onClick={handleSubmit} 
          disabled={isSubmitting}
          style={{ width: window.innerWidth < 640 ? '100%' : 'auto' }}
        >
          {isSubmitting ? '...' : t('submit')}
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
