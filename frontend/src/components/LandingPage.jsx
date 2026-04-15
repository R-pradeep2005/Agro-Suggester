import React from 'react';
import { motion } from 'framer-motion';
import { Sprout, TrendingUp, Cloud, Lightbulb, MapPin, Brain, BarChart3 } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import Navbar from './Navbar';
import '../landing.css';

const LandingPage = ({ onGetStarted }) => {
  const { t } = useTranslation();

  const features = [
    {
      icon: <Cloud size={32} />,
      title: t('lp_feat1Title'),
      description: t('lp_feat1Desc')
    },
    {
      icon: <TrendingUp size={32} />,
      title: t('lp_feat2Title'),
      description: t('lp_feat2Desc')
    },
    {
      icon: <Sprout size={32} />,
      title: t('lp_feat3Title'),
      description: t('lp_feat3Desc')
    },
    {
      icon: <Lightbulb size={32} />,
      title: t('lp_feat4Title'),
      description: t('lp_feat4Desc')
    }
  ];

  const steps = [
    {
      icon: <MapPin size={24} />,
      title: t('lp_step1Title'),
      description: t('lp_step1Desc')
    },
    {
      icon: <Cloud size={24} />,
      title: t('lp_step2Title'),
      description: t('lp_step2Desc')
    },
    {
      icon: <Brain size={24} />,
      title: t('lp_step3Title'),
      description: t('lp_step3Desc')
    }
  ];

  const stats = [
    { icon: <TrendingUp size={32} />, value: '4.3T', label: t('lp_statYield'), gradient: 'lp-stat-green' },
    { icon: <Sprout size={32} />, value: 'Top 3', label: t('lp_statRecs'), gradient: 'lp-stat-emerald' },
    { icon: <Brain size={32} />, value: '92%', label: t('lp_statAcc'), gradient: 'lp-stat-teal' },
  ];

  return (
    <div className="lp-root">
      {/* Shared Navbar (matches form & dashboard pages) */}
      <Navbar currentInstructionText={t('lp_heroDesc')} />


      <div id="step-content">
        {/* Hero Section */}
        <section className="lp-hero">
          <div className="lp-hero-inner">
            <motion.div
              initial={{ opacity: 1, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
            >
              <div className="lp-badge">{t('lp_badge')}</div>
              <h1 className="lp-hero-title">
                {t('lp_heroTitle1')}
                <br />
                {t('lp_heroTitle2')}
              </h1>
              <p className="lp-hero-desc">
                {t('lp_heroDesc')}
              </p>
              <div className="lp-hero-btns">
                <motion.button
                  className="lp-btn-primary"
                  onClick={onGetStarted}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <Sprout size={20} />
                  {t('lp_btnStart')}
                </motion.button>
                <motion.button
                  className="lp-btn-outline"
                  onClick={onGetStarted}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <BarChart3 size={20} />
                  {t('lp_btnViewDash')}
                </motion.button>
              </div>
            </motion.div>

            {/* Hero Stats */}
            <motion.div
              className="lp-stats-wrap"
              initial={{ opacity: 1, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.8, delay: 0.2 }}
            >
              <div className="lp-stats-glow"></div>
              <div className="lp-stats-card">
                <div className="lp-stats-grid">
                  {stats.map((stat, i) => (
                    <motion.div
                      key={i}
                      className={`lp-stat-item ${stat.gradient}`}
                      whileHover={{ scale: 1.03, y: -4 }}
                      transition={{ duration: 0.2 }}
                    >
                      <div className="lp-stat-icon">{stat.icon}</div>
                      <div className="lp-stat-value">{stat.value}</div>
                      <div className="lp-stat-label">{stat.label}</div>
                    </motion.div>
                  ))}
                </div>
              </div>
            </motion.div>
          </div>
        </section>

        {/* Features Section */}
        <section className="lp-features">
          <div className="lp-section-inner">
            <motion.div
              className="lp-section-header"
              initial={{ opacity: 1, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6 }}
            >
              <h2 className="lp-section-title">{t('lp_featuresTitle')}</h2>
              <p className="lp-section-desc">{t('lp_featuresDesc')}</p>
            </motion.div>
            <div className="lp-features-grid">
              {features.map((feat, i) => (
                <motion.div
                  key={i}
                  className="lp-feature-card"
                  initial={{ opacity: 1, y: 30 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.6, delay: i * 0.1 }}
                  whileHover={{ y: -6 }}
                >
                  <div className="lp-feature-icon">{feat.icon}</div>
                  <h3 className="lp-feature-title">{feat.title}</h3>
                  <p className="lp-feature-desc">{feat.description}</p>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        {/* How It Works */}
        <section className="lp-steps">
          <div className="lp-section-inner">
            <motion.div
              className="lp-section-header"
              initial={{ opacity: 1, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6 }}
            >
              <h2 className="lp-section-title">{t('lp_stepsTitle')}</h2>
              <p className="lp-section-desc">{t('lp_stepsDesc')}</p>
            </motion.div>
            <div className="lp-steps-grid">
              {steps.map((step, i) => (
                <motion.div
                  key={i}
                  className="lp-step-card"
                  initial={{ opacity: 1, x: -30 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.6, delay: i * 0.2 }}
                >
                  <div className="lp-step-icon">{step.icon}</div>
                  <div className="lp-step-num">0{i + 1}</div>
                  <h3 className="lp-step-title">{step.title}</h3>
                  <p className="lp-step-desc">{step.description}</p>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        {/* CTA Section */}
        <section className="lp-cta-section">
          <div className="lp-section-inner">
            <motion.div
              className="lp-cta-card"
              initial={{ opacity: 1, scale: 0.9 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6 }}
            >
              <div className="lp-cta-glow lp-cta-glow-tr"></div>
              <div className="lp-cta-glow lp-cta-glow-bl"></div>
              <div className="lp-cta-content">
                <h2 className="lp-cta-title">{t('lp_ctaTitle')}</h2>
                <p className="lp-cta-desc">{t('lp_ctaDesc')}</p>
                <motion.button
                  className="lp-cta-btn"
                  onClick={onGetStarted}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  {t('lp_ctaBtn')}
                </motion.button>
              </div>
            </motion.div>
          </div>
        </section>

        {/* Footer */}
        <footer className="lp-footer">
          <div className="lp-footer-inner">
            <div className="lp-footer-logo">
              <Sprout size={20} color="#10b981" />
              <span>AgroPredict AI</span>
            </div>
            <p className="lp-footer-copy">
              {t('lp_footerCopy')}
            </p>
          </div>
        </footer>
      </div>
    </div>
  );
};

export default LandingPage;
