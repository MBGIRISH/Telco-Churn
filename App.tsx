import React from 'react';
import { HashRouter, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './components/Dashboard';
import PredictionForm from './components/PredictionForm';
import DataView from './components/DataView';

const App: React.FC = () => {
  return (
    <HashRouter>
      <Layout>
        <Routes>
          <Route path="/" element={
              <div>
                  <header className="mb-8">
                    <h1 className="text-2xl font-bold text-slate-800">Dashboard</h1>
                    <p className="text-slate-500">Overview of customer churn metrics and data analysis.</p>
                  </header>
                  <Dashboard />
              </div>
          } />
          <Route path="/predict" element={
              <div>
                  <header className="mb-8">
                    <h1 className="text-2xl font-bold text-slate-800">Churn Prediction</h1>
                    <p className="text-slate-500">AI-powered analysis to identify at-risk customers.</p>
                  </header>
                  <PredictionForm />
              </div>
          } />
          <Route path="/data" element={
              <div>
                  <header className="mb-8">
                    <h1 className="text-2xl font-bold text-slate-800">Data Management</h1>
                    <p className="text-slate-500">Inspect raw data and generate SQL migration scripts.</p>
                  </header>
                  <DataView />
              </div>
          } />
        </Routes>
      </Layout>
    </HashRouter>
  );
};

export default App;