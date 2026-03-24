import React, { useEffect, useRef } from 'react';
import { Leaf, Info, Thermometer, Droplets, Wind, Cloud, CheckCircle2, Award } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { useFormContext } from '../context/FormContext';
import html2canvas from 'html2canvas';
import { jsPDF } from 'jspdf';

const Dashboard = () => {
  const { t } = useTranslation();
  const { resetForm } = useFormContext();
  const pdfGeneratedRef = useRef(false);

  useEffect(() => {
    if (pdfGeneratedRef.current) return;
    pdfGeneratedRef.current = true;

    const generatePDF = async () => {
      // 800ms buffer to allow heavy icon glyphs and external fonts to paint fully before capture
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
            <div className="banner-value">2.28</div>
            <div className="banner-subtitle">{t('dash_tonsPerHa')}</div>
          </div>
          <div className="banner-right">
            <div className="accuracy-header">
              <span>{t('dash_modelAcc')}</span>
              <Award size={20} />
            </div>
            <div className="accuracy-value">92%</div>
            <div className="progress-bg"><div className="progress-fill" style={{width: '92%'}}></div></div>
            <p className="accuracy-desc">{t('dash_modelDesc')}</p>
          </div>
        </section>

        {/* Climate Data Section */}
        <section className="dash-section">
          <h3 className="section-title"><Cloud size={24} className="title-icon"/> {t('dash_histClimate')}</h3>
          <div className="climate-grid">
            <div className="climate-card bg-orange">
              <Thermometer size={24} />
              <h4>24.3°C</h4>
              <p>{t('dash_curTemp')}</p>
            </div>
            <div className="climate-card bg-blue">
              <Droplets size={24} />
              <h4>824 mm</h4>
              <p>{t('dash_annRainfall')}</p>
            </div>
            <div className="climate-card bg-teal">
              <Wind size={24} />
              <h4>81%</h4>
              <p>{t('dash_annHumidity')}</p>
            </div>
            <div className="climate-card bg-purple">
              <Cloud size={24} />
              <h4>{t('dash_clearSky')}</h4>
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
          
          <div className="crop-card">
            <div className="crop-header">
              <div className="crop-header-left">
                <span className="crop-emoji">🌾</span>
                <div className="crop-title-col">
                  <div className="crop-title-row">
                    <h4>{t('dash_rice')}</h4>
                    <span className="badge badge-yellow">{t('dash_bestPick')}</span>
                  </div>
                  <p>{t('dash_riceDesc')}</p>
                </div>
              </div>
              <div className="score-circle score-green">80</div>
            </div>
            
            <div className="suitability-bar-container">
              <div className="suitability-labels">
                <span>{t('dash_suitability')}</span>
                <span>80 / 100</span>
              </div>
              <div className="progress-bg"><div className="progress-fill bg-green" style={{width: '80%'}}></div></div>
            </div>

            <div className="crop-details-grid">
              <div className="yield-box">
                <span>{t('dash_estYield')}</span>
                <strong>2.3 T/Ha</strong>
              </div>
              <div className="info-box info-purple">
                <Info size={16} className="info-icon" />
                <p>{t('dash_riceInfo')}</p>
              </div>
            </div>
          </div>

          <div className="crop-card">
            <div className="crop-header">
              <div className="crop-header-left">
                <span className="crop-emoji">🧅</span>
                <div className="crop-title-col">
                  <div className="crop-title-row">
                    <h4>{t('dash_onion')}</h4>
                    <span className="badge badge-cyan">{t('dash_5th')}</span>
                  </div>
                  <p>{t('dash_onionDesc')}</p>
                </div>
              </div>
              <div className="score-circle score-orange">60</div>
            </div>
            
            <div className="suitability-bar-container">
              <div className="suitability-labels">
                <span>{t('dash_suitability')}</span>
                <span>60 / 100</span>
              </div>
              <div className="progress-bg"><div className="progress-fill bg-orange" style={{width: '60%'}}></div></div>
            </div>

            <div className="crop-details-grid">
              <div className="yield-box">
                <span>{t('dash_estYield')}</span>
                <strong>5.4 T/Ha</strong>
              </div>
              <div className="info-box info-purple">
                <Info size={16} className="info-icon" />
                <p>{t('dash_onionInfo')}</p>
              </div>
            </div>

            <div className="why-fits">
              <h5><CheckCircle2 size={16} color="var(--primary-color)" /> {t('dash_whyFits')}</h5>
              <ul>
                <li><CheckCircle2 size={16} color="var(--primary-color)" /> {t('dash_fit1')}</li>
                <li><CheckCircle2 size={16} color="var(--primary-color)" /> {t('dash_fit2')}</li>
                <li><CheckCircle2 size={16} color="var(--primary-color)" /> {t('dash_fit3')}</li>
                <li><CheckCircle2 size={16} color="var(--primary-color)" /> {t('dash_fit4')}</li>
                <li><CheckCircle2 size={16} color="var(--primary-color)" /> {t('dash_fit5')}</li>
              </ul>
            </div>
          </div>

        </section>

        {/* Explainable AI Section */}
        <section className="dash-section">
          <div className="explainable-card">
            <h3 className="section-title"><Award size={24} className="title-icon"/> {t('dash_expAI')}</h3>
            <p className="section-desc">{t('dash_expAIDesc')}</p>
            
            {/* Bar Chart Mockup */}
            <div className="bar-chart">
              <div className="y-axis">
                <span>28-</span>
                <span>21-</span>
                <span>14-</span>
                <span>7-</span>
                <span>0-</span>
              </div>
              <div className="chart-grid">
                <div className="bar-col">
                  <div className="bar" style={{height: '80%', backgroundColor: '#4f46e5'}}></div>
                  <span className="bar-label">{t('dash_featRainfall')}</span>
                </div>
                <div className="bar-col">
                  <div className="bar" style={{height: '60%', backgroundColor: '#9333ea'}}></div>
                  <span className="bar-label">{t('dash_featSoil')}</span>
                </div>
                <div className="bar-col">
                  <div className="bar" style={{height: '65%', backgroundColor: '#ff0055'}}></div>
                  <span className="bar-label">{t('dash_featTemp')}</span>
                </div>
                <div className="bar-col">
                  <div className="bar" style={{height: '92%', backgroundColor: '#06b6d4'}}></div>
                  <span className="bar-label">{t('dash_featHumidity')}</span>
                </div>
                <div className="bar-col">
                  <div className="bar" style={{height: '70%', backgroundColor: '#10b981'}}></div>
                  <span className="bar-label">{t('dash_featSeason')}</span>
                </div>
              </div>
            </div>

            {/* Horizontal progress bars for weights */}
            <div className="weights-grid">
              <div className="weight-item">
                <div className="weight-header"><span className="dot dot-blue"></span> {t('dash_featRainfall')} <span className="weight-val">22%</span></div>
                <div className="progress-bg"><div className="progress-fill" style={{backgroundColor: '#4f46e5', width: '22%'}}></div></div>
              </div>
              <div className="weight-item">
                <div className="weight-header"><span className="dot dot-purple"></span> {t('dash_featSoil')} <span className="weight-val">16%</span></div>
                <div className="progress-bg"><div className="progress-fill" style={{backgroundColor: '#9333ea', width: '16%'}}></div></div>
              </div>
              <div className="weight-item">
                <div className="weight-header"><span className="dot dot-red"></span> {t('dash_featTemp')} <span className="weight-val">17%</span></div>
                <div className="progress-bg"><div className="progress-fill" style={{backgroundColor: '#ff0055', width: '17%'}}></div></div>
              </div>
              <div className="weight-item">
                <div className="weight-header"><span className="dot dot-cyan"></span> {t('dash_featHumidity')} <span className="weight-val">26%</span></div>
                <div className="progress-bg"><div className="progress-fill" style={{backgroundColor: '#06b6d4', width: '26%'}}></div></div>
              </div>
              <div className="weight-item">
                <div className="weight-header"><span className="dot dot-green"></span> {t('dash_featSeason')} <span className="weight-val">19%</span></div>
                <div className="progress-bg"><div className="progress-fill" style={{backgroundColor: '#10b981', width: '19%'}}></div></div>
              </div>
            </div>

            {/* Info Box */}
            <div className="info-box info-purple-darker" style={{marginTop: '2rem'}}>
              <Award size={24} className="info-icon" color="#4f46e5" />
              <div>
                <h5 style={{color: '#4f46e5', marginBottom: '0.25rem'}}>{t('dash_whyWeights')}</h5>
                <p>{t('dash_weightsDesc')}</p>
              </div>
            </div>

          </div>
        </section>

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
