import React, { useEffect, useRef } from 'react';
import { Leaf, Info, Thermometer, Droplets, Wind, Cloud, CheckCircle2, Award } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { useFormContext } from '../context/FormContext';
import html2canvas from 'html2canvas';
import { jsPDF } from 'jspdf';

const CROP_EMOJI = { corn: '🌽', cotton: '🌿', rice: '🌾', sugarcane: '🎋', tomato: '🍅' };

const scoreColor = (score) => {
  if (score >= 80) return '#10b981';
  if (score >= 60) return '#f59e0b';
  return '#ef4444';
};

const Dashboard = () => {
  const { t } = useTranslation();
  const { resetForm, apiResult } = useFormContext();
  const pdfGeneratedRef = useRef(false);

  // Parsed data from API
  const predictedYield = apiResult?.predicted_yield?.value ?? '—';
  const yieldUnit = apiResult?.predicted_yield?.unit?.replace('_', ' ') ?? 'tons/ha';
  const modelAcc = apiResult?.model_accuracy_percent ?? 92;
  const climate = apiResult?.climate_data ?? {};
  const cropRecs = apiResult?.crop_recommendations ?? [];
  const featureWeights = apiResult?.feature_weights ?? {};

  useEffect(() => {
    if (pdfGeneratedRef.current) return;
    pdfGeneratedRef.current = true;

    const generatePDF = async () => {
      await new Promise(resolve => setTimeout(resolve, 800));
      const element = document.getElementById('step-content');
      if (!element) return;

      try {
        const canvas = await html2canvas(element, { scale: 2, useCORS: true, logging: false });
        const imgData = canvas.toDataURL('image/png');
        const pdf = new jsPDF('p', 'mm', 'a4');
        const pdfWidth = pdf.internal.pageSize.getWidth();
        const pdfHeight = (canvas.height * pdfWidth) / canvas.width;
        pdf.addImage(imgData, 'PNG', 0, 0, pdfWidth, pdfHeight);
        pdf.save('AgroPredict_Results.pdf');
      } catch (err) {
        console.error("PDF creation failed", err);
      }
    };

    generatePDF();
  }, []);

  return (
    <div className="dashboard-container" id="step-content">
      <main className="dash-main">
        {/* Hero Banner */}
        <section className="dash-banner">
          <div className="banner-left">
            <div className="banner-title">
              <span className="banner-icon">↗</span> {t('dash_predYield')}
            </div>
            <div className="banner-value">{predictedYield}</div>
            <div className="banner-subtitle">{yieldUnit}</div>
          </div>
          <div className="banner-right">
            <div className="accuracy-header">
              <span>{t('dash_modelAcc')}</span>
              <Award size={20} />
            </div>
            <div className="accuracy-value">{modelAcc}%</div>
            <div className="progress-bg"><div className="progress-fill" style={{width: `${modelAcc}%`}}></div></div>
            <p className="accuracy-desc">{t('dash_modelDesc')}</p>
          </div>
        </section>

        {/* Climate Data Section */}
        <section className="dash-section">
          <h3 className="section-title"><Cloud size={24} className="title-icon"/> {t('dash_histClimate')}</h3>
          <div className="climate-grid">
            <div className="climate-card bg-orange">
              <Thermometer size={24} />
              <h4>{climate.temperature_celsius ?? '—'}°C</h4>
              <p>{t('dash_curTemp')}</p>
            </div>
            <div className="climate-card bg-blue">
              <Droplets size={24} />
              <h4>{climate.annual_rainfall_mm ?? '—'} mm</h4>
              <p>{t('dash_annRainfall')}</p>
            </div>
            <div className="climate-card bg-teal">
              <Wind size={24} />
              <h4>{climate.annual_humidity_percent ?? '—'}%</h4>
              <p>{t('dash_annHumidity')}</p>
            </div>
            <div className="climate-card bg-purple">
              <Cloud size={24} />
              <h4>{climate.current_condition?.replace('_', ' ') ?? '—'}</h4>
              <p>{t('dash_curCond')}</p>
            </div>
          </div>
          <div className="info-box info-blue">
            <Info size={16} className="info-icon" />
            <p>{t('dash_climateSource')}</p>
          </div>
        </section>

        {/* Crop Recommendations Section */}
        <section className="dash-section">
          <h3 className="section-title"><Leaf size={24} className="title-icon"/> {t('dash_cropRecs')}</h3>
          <p className="section-desc">{t('dash_cropRecsDesc')}</p>

          {cropRecs.length === 0 && (
            <p style={{ color: 'var(--text-muted)', textAlign: 'center', padding: '2rem' }}>
              No crop recommendations available.
            </p>
          )}

          {cropRecs.map((crop) => {
            const emoji = CROP_EMOJI[crop.crop_name] ?? '🌱';
            const color = scoreColor(crop.suitability_score);
            return (
              <div className="crop-card" key={crop.crop_name}>
                <div className="crop-header">
                  <div className="crop-header-left">
                    <span className="crop-emoji">{emoji}</span>
                    <div className="crop-title-col">
                      <div className="crop-title-row">
                        <h4>{t(`crop_${crop.crop_name}`, crop.crop_name.charAt(0).toUpperCase() + crop.crop_name.slice(1))}</h4>
                        {crop.rank === 1 && <span className="badge badge-yellow">{t('dash_bestPick')}</span>}
                        {crop.rank === 2 && <span className="badge badge-cyan">2nd</span>}
                        {crop.rank === 3 && <span className="badge badge-cyan">3rd</span>}
                      </div>
                    </div>
                  </div>
                  <div className="score-circle" style={{ background: color }}>{crop.suitability_score}</div>
                </div>

                <div className="suitability-bar-container">
                  <div className="suitability-labels">
                    <span>{t('dash_suitability')}</span>
                    <span>{crop.suitability_score} / 100</span>
                  </div>
                  <div className="progress-bg">
                    <div className="progress-fill" style={{ width: `${crop.suitability_score}%`, backgroundColor: color }}></div>
                  </div>
                </div>

                <div className="crop-details-grid">
                  <div className="yield-box">
                    <span>{t('dash_estYield')}</span>
                    <strong>{crop.estimated_yield_tons_per_ha} T/Ha</strong>
                  </div>
                </div>

                {crop.why_fits?.length > 0 && (
                  <div className="why-fits">
                    <h5><CheckCircle2 size={16} color="var(--primary-color)" /> {t('dash_whyFits')}</h5>
                    <ul>
                      {crop.why_fits.map((reason, i) => (
                        <li key={i}><CheckCircle2 size={16} color="var(--primary-color)" /> {t(reason, reason)}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            );
          })}
        </section>

        {/* Feature Weights Section */}
        {Object.keys(featureWeights).length > 0 && (
          <section className="dash-section">
            <div className="explainable-card">
              <h3 className="section-title"><Award size={24} className="title-icon"/> {t('dash_expAI')}</h3>
              <p className="section-desc">{t('dash_expAIDesc')}</p>
              <div className="weights-grid">
                {Object.entries(featureWeights).map(([key, val]) => (
                  <div className="weight-item" key={key}>
                    <div className="weight-header">
                      <span className="dot dot-blue"></span> {t(`feat_${key}`, key)} <span className="weight-val">{val}%</span>
                    </div>
                    <div className="progress-bg">
                      <div className="progress-fill" style={{ backgroundColor: '#4f46e5', width: `${val}%` }}></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </section>
        )}

        <div style={{ display: 'flex', justifyContent: 'center', marginTop: '3rem', marginBottom: '2rem' }}>
          <button
            className="btn btn-primary"
            style={{ padding: '1rem 2.5rem', fontSize: '1.25rem', borderRadius: '12px', boxShadow: '0 4px 14px rgba(16, 185, 129, 0.4)' }}
            onClick={resetForm}
          >
            {t('dash_newPrediction')}
          </button>
        </div>
      </main>
    </div>
  );
};

export default Dashboard;
