import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useFormContext } from '../context/FormContext';
import { MapPin, CheckCircle2, Loader2 } from 'lucide-react';

/* ── Shared Step Wrapper ───────────────────────────────── */
const StepWrapper = ({ title, children, canProceed, onNext }) => {
  const { t } = useTranslation();
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <h2 className="step-title">{title}</h2>
      <div style={{ flex: 1 }}>{children}</div>
      <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '2.5rem' }}>
        {onNext && (
          <button
            className="btn btn-primary"
            onClick={onNext}
            disabled={!canProceed}
          >
            {t('next')} →
          </button>
        )}
      </div>
    </div>
  );
};

/* ── 1. Geolocation ────────────────────────────────────── */
export const GeolocationStep = () => {
  const { t } = useTranslation();
  const { formData, updateFormData, nextStep } = useFormContext();
  const [loading, setLoading] = useState(false);

  const handleGetLocation = () => {
    setLoading(true);
    if ('geolocation' in navigator) {
      navigator.geolocation.getCurrentPosition((pos) => {
        updateFormData('geolocation', `${pos.coords.latitude.toFixed(5)}, ${pos.coords.longitude.toFixed(5)}`);
        setLoading(false);
      }, () => setLoading(false));
    } else {
      setLoading(false);
    }
  };

  return (
    <StepWrapper title={t('geolocation')} canProceed={!!formData.geolocation} onNext={nextStep}>
      <div className="location-area">
        <button className="btn btn-secondary" onClick={handleGetLocation} disabled={loading}
          style={{ padding: '1rem 2rem', fontSize: '0.95rem' }}>
          {loading
            ? <><Loader2 size={18} className="spin-icon" /> Detecting…</>
            : <><MapPin size={18} /> {formData.geolocation ? t('locationCaptured') : t('getLocation')}</>
          }
        </button>
        {formData.geolocation && (
          <div className="location-coords">
            <MapPin size={15} />
            {formData.geolocation}
          </div>
        )}
      </div>
    </StepWrapper>
  );
};

/* ── 2. Planned Date ───────────────────────────────────── */
export const PlannedDateStep = () => {
  const { t } = useTranslation();
  const { formData, updateFormData, nextStep } = useFormContext();
  return (
    <StepWrapper title={t('plannedDate')} canProceed={!!formData.plannedDate} onNext={nextStep}>
      <input type="date" className="input-base" value={formData.plannedDate}
        onChange={(e) => updateFormData('plannedDate', e.target.value)} />
    </StepWrapper>
  );
};

/* ── 3. Previous Crop ──────────────────────────────────── */
export const PrevCropNameStep = () => {
  const { t } = useTranslation();
  const { formData, updateFormData, nextStep } = useFormContext();
  const crops = ['Rice', 'Wheat', 'Maize', 'Cotton', 'Sugarcane'];
  return (
    <StepWrapper title={t('prevCropName')} canProceed={!!formData.prevCropName} onNext={nextStep}>
      <input list="crops" className="input-base" placeholder={t('selectCrop')}
        value={formData.prevCropName} onChange={(e) => updateFormData('prevCropName', e.target.value)} />
      <datalist id="crops">
        {crops.map(c => <option key={c} value={c} />)}
      </datalist>
    </StepWrapper>
  );
};

/* ── 4. Crop Yield ─────────────────────────────────────── */
export const PrevCropYieldStep = () => {
  const { t } = useTranslation();
  const { formData, updateFormData, nextStep } = useFormContext();
  return (
    <StepWrapper title={t('prevCropYield')} canProceed={!!formData.prevCropYieldValue} onNext={nextStep}>
      <div style={{ display: 'flex', gap: '1rem' }}>
        <input type="number" className="input-base" style={{ flex: 2 }}
          value={formData.prevCropYieldValue} onChange={(e) => updateFormData('prevCropYieldValue', e.target.value)} />
        <select className="input-base" style={{ flex: 1 }}
          value={formData.prevCropYieldUnit} onChange={(e) => updateFormData('prevCropYieldUnit', e.target.value)}>
          <option value="kg">{t('kg')}</option>
          <option value="tons">{t('tons')}</option>
        </select>
      </div>
    </StepWrapper>
  );
};

/* ── 5. Fertilizer Name ────────────────────────────────── */
export const PrevFertilizerStep = () => {
  const { t } = useTranslation();
  const { formData, updateFormData, nextStep } = useFormContext();
  const fertilizers = ['Urea', 'DAP', 'Compost', 'NPK', 'None'];
  return (
    <StepWrapper title={t('prevFertilizerName')} canProceed={!!formData.prevFertilizerName} onNext={nextStep}>
      <select className="input-base" value={formData.prevFertilizerName}
        onChange={(e) => updateFormData('prevFertilizerName', e.target.value)}>
        <option value="" disabled>{t('selectFertilizer')}</option>
        {fertilizers.map(f => <option key={f} value={f}>{f}</option>)}
      </select>
    </StepWrapper>
  );
};

/* ── 6. Fertilizer Amount ──────────────────────────────── */
export const PrevFertilizerAmountStep = () => {
  const { t } = useTranslation();
  const { formData, updateFormData, nextStep } = useFormContext();
  return (
    <StepWrapper title={t('prevFertilizerAmount')} canProceed={!!formData.prevFertilizerAmountValue} onNext={nextStep}>
      <div style={{ display: 'flex', gap: '1rem' }}>
        <input type="number" className="input-base" style={{ flex: 2 }}
          value={formData.prevFertilizerAmountValue} onChange={(e) => updateFormData('prevFertilizerAmountValue', e.target.value)} />
        <select className="input-base" style={{ flex: 1 }}
          value={formData.prevFertilizerAmountUnit} onChange={(e) => updateFormData('prevFertilizerAmountUnit', e.target.value)}>
          <option value="kg">{t('kg')}</option>
          <option value="liters">{t('liters')}</option>
        </select>
      </div>
    </StepWrapper>
  );
};

/* ── 7. Harvest Date ───────────────────────────────────── */
export const HarvestDateStep = () => {
  const { t } = useTranslation();
  const { formData, updateFormData, nextStep } = useFormContext();
  return (
    <StepWrapper title={t('harvestDate')} canProceed={!!formData.harvestDate} onNext={nextStep}>
      <input type="date" className="input-base" value={formData.harvestDate}
        onChange={(e) => updateFormData('harvestDate', e.target.value)} />
    </StepWrapper>
  );
};

/* ── 8. Residue Status ─────────────────────────────────── */
export const ResidueStatusStep = () => {
  const { t } = useTranslation();
  const { formData, updateFormData, nextStep } = useFormContext();
  return (
    <StepWrapper title={t('residueStatus')} canProceed={!!formData.residueStatus} onNext={nextStep}>
      <div className="toggle-group">
        <button className={`toggle-btn ${formData.residueStatus === 'left' ? 'active' : ''}`}
          onClick={() => updateFormData('residueStatus', 'left')}>
          🌾 {t('leftInLand')}
        </button>
        <button className={`toggle-btn ${formData.residueStatus === 'disposed' ? 'active' : ''}`}
          onClick={() => updateFormData('residueStatus', 'disposed')}>
          ♻️ {t('disposedElsewhere')}
        </button>
      </div>
    </StepWrapper>
  );
};

/* ── 9. Review & Submit ────────────────────────────────── */
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
      if (fertName.includes('Urea'))     { n = fertAmt * 0.46; }
      else if (fertName.includes('DAP')) { n = fertAmt * 0.18; p = fertAmt * 0.46; }
      else if (fertName.includes('NPK')) { n = fertAmt * 0.19; p = fertAmt * 0.19; k = fertAmt * 0.19; }

      const payload = {
        lat, lon,
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
        setApiResult(await response.json());
      } else {
        console.error('API error:', await response.text());
      }
    } catch (e) {
      console.error('Submit error:', e);
    } finally {
      setIsSubmitting(false);
      setIsLoading(false);
      setIsSubmitted(true);
    }
  };

  const Row = ({ label, value, stepIdx }) => (
    <div className="review-row">
      <div>
        <div className="review-label">{label}</div>
        <div className="review-value">{value || '—'}</div>
      </div>
      <button className="btn btn-secondary"
        style={{ padding: '0.4rem 1rem', fontSize: '0.84rem' }}
        onClick={() => jumpToStep(stepIdx)}>
        {t('edit')}
      </button>
    </div>
  );

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <h2 className="step-title">{t('reviewStep')}</h2>
      <p style={{ color: 'var(--text-secondary)', marginBottom: '1.5rem', fontSize: '1rem' }}>{t('reviewDesc')}</p>

      <div style={{ flex: 1, overflowY: 'auto' }}>
        <Row label={t('geolocation')}            value={formData.geolocation}               stepIdx={0} />
        <Row label={t('plannedDate')}             value={formData.plannedDate}               stepIdx={1} />
        <Row label={t('prevCropName')}            value={formData.prevCropName}              stepIdx={2} />
        <Row label={t('prevCropYield')}           value={`${formData.prevCropYieldValue} ${t(formData.prevCropYieldUnit)}`} stepIdx={3} />
        <Row label={t('prevFertilizerName')}      value={formData.prevFertilizerName}        stepIdx={4} />
        <Row label={t('prevFertilizerAmount')}    value={`${formData.prevFertilizerAmountValue} ${t(formData.prevFertilizerAmountUnit)}`} stepIdx={5} />
        <Row label={t('harvestDate')}             value={formData.harvestDate}               stepIdx={6} />
        <Row label={t('residueStatus')}           value={t(formData.residueStatus === 'left' ? 'leftInLand' : formData.residueStatus === 'disposed' ? 'disposedElsewhere' : '')} stepIdx={7} />
      </div>

      <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '2rem' }}>
        <button className="btn btn-primary" onClick={handleSubmit} disabled={isSubmitting}
          style={{ padding: '1rem 2.5rem', fontSize: '1rem' }}>
          {isSubmitting
            ? <><Loader2 size={18} className="spin-icon" /> Processing…</>
            : <><CheckCircle2 size={18} /> {t('submit')}</>
          }
        </button>
      </div>
    </div>
  );
};

export const FormStepsList = [
  GeolocationStep, PlannedDateStep, PrevCropNameStep, PrevCropYieldStep,
  PrevFertilizerStep, PrevFertilizerAmountStep, HarvestDateStep, ResidueStatusStep, ReviewStep,
];
