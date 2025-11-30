import React, { useState } from 'react';
import { predictChurn, PredictionResult } from '../services/gemini';
import { Loader2, AlertCircle, CheckCircle2 } from 'lucide-react';

const PredictionForm: React.FC = () => {
  const [formData, setFormData] = useState({
    gender: 'Male',
    SeniorCitizen: 0,
    Partner: 'No',
    Dependents: 'No',
    tenure: 12,
    PhoneService: 'Yes',
    MultipleLines: 'No',
    InternetService: 'Fiber optic',
    OnlineSecurity: 'No',
    OnlineBackup: 'No',
    DeviceProtection: 'No',
    TechSupport: 'No',
    StreamingTV: 'Yes',
    StreamingMovies: 'Yes',
    Contract: 'Month-to-month',
    PaperlessBilling: 'Yes',
    PaymentMethod: 'Electronic check',
    MonthlyCharges: 70.5
  });

  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: name === 'tenure' || name === 'MonthlyCharges' || name === 'SeniorCitizen' ? Number(value) : value
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);
    
    const prediction = await predictChurn(formData);
    
    setResult(prediction);
    setLoading(false);
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
      {/* Form Section */}
      <div className="lg:col-span-2 bg-white p-8 rounded-xl shadow-sm border border-slate-100">
        <h2 className="text-xl font-semibold text-slate-800 mb-6">Customer Details</h2>
        <form onSubmit={handleSubmit} className="grid grid-cols-1 md:grid-cols-2 gap-6">
          
          {/* Demographics */}
          <div className="space-y-4">
            <h3 className="font-medium text-slate-500 border-b pb-2">Demographics</h3>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Gender</label>
              <select name="gender" value={formData.gender} onChange={handleChange} className="w-full p-2 border rounded-lg bg-slate-50">
                <option value="Male">Male</option>
                <option value="Female">Female</option>
              </select>
            </div>
            <div className="flex items-center gap-4">
                <div className="flex-1">
                    <label className="block text-sm font-medium text-slate-700 mb-1">Senior Citizen</label>
                    <select name="SeniorCitizen" value={formData.SeniorCitizen} onChange={handleChange} className="w-full p-2 border rounded-lg bg-slate-50">
                        <option value={0}>No</option>
                        <option value={1}>Yes</option>
                    </select>
                </div>
                <div className="flex-1">
                    <label className="block text-sm font-medium text-slate-700 mb-1">Partner</label>
                    <select name="Partner" value={formData.Partner} onChange={handleChange} className="w-full p-2 border rounded-lg bg-slate-50">
                        <option value="No">No</option>
                        <option value="Yes">Yes</option>
                    </select>
                </div>
            </div>
             <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Dependents</label>
              <select name="Dependents" value={formData.Dependents} onChange={handleChange} className="w-full p-2 border rounded-lg bg-slate-50">
                <option value="No">No</option>
                <option value="Yes">Yes</option>
              </select>
            </div>
          </div>

          {/* Account Info */}
          <div className="space-y-4">
             <h3 className="font-medium text-slate-500 border-b pb-2">Account Info</h3>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Tenure (Months)</label>
              <input type="number" name="tenure" value={formData.tenure} onChange={handleChange} className="w-full p-2 border rounded-lg bg-slate-50" min="0" max="100" />
            </div>
             <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Contract</label>
              <select name="Contract" value={formData.Contract} onChange={handleChange} className="w-full p-2 border rounded-lg bg-slate-50">
                <option value="Month-to-month">Month-to-month</option>
                <option value="One year">One year</option>
                <option value="Two year">Two year</option>
              </select>
            </div>
             <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Monthly Charges ($)</label>
              <input type="number" name="MonthlyCharges" value={formData.MonthlyCharges} onChange={handleChange} className="w-full p-2 border rounded-lg bg-slate-50" />
            </div>
          </div>

          {/* Services */}
          <div className="space-y-4 md:col-span-2">
            <h3 className="font-medium text-slate-500 border-b pb-2">Services</h3>
             <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                    <label className="block text-sm font-medium text-slate-700 mb-1">Internet Service</label>
                    <select name="InternetService" value={formData.InternetService} onChange={handleChange} className="w-full p-2 border rounded-lg bg-slate-50">
                        <option value="DSL">DSL</option>
                        <option value="Fiber optic">Fiber optic</option>
                        <option value="No">No</option>
                    </select>
                </div>
                <div>
                    <label className="block text-sm font-medium text-slate-700 mb-1">Tech Support</label>
                    <select name="TechSupport" value={formData.TechSupport} onChange={handleChange} className="w-full p-2 border rounded-lg bg-slate-50">
                        <option value="No">No</option>
                        <option value="Yes">Yes</option>
                        <option value="No internet service">No internet service</option>
                    </select>
                </div>
                 <div>
                    <label className="block text-sm font-medium text-slate-700 mb-1">Online Security</label>
                    <select name="OnlineSecurity" value={formData.OnlineSecurity} onChange={handleChange} className="w-full p-2 border rounded-lg bg-slate-50">
                        <option value="No">No</option>
                        <option value="Yes">Yes</option>
                         <option value="No internet service">No internet service</option>
                    </select>
                </div>
            </div>
          </div>

          <div className="md:col-span-2 mt-4">
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-3 px-6 rounded-lg transition-colors flex items-center justify-center gap-2 disabled:opacity-70"
            >
              {loading ? <Loader2 className="animate-spin h-5 w-5" /> : 'Predict Churn Risk'}
            </button>
          </div>
        </form>
      </div>

      {/* Prediction Result */}
      <div className="lg:col-span-1">
        {result ? (
          <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-100 h-full">
            <div className="mb-6 text-center">
               <span className={`inline-flex items-center justify-center w-16 h-16 rounded-full mb-4 ${
                   result.riskLevel === 'High' ? 'bg-red-100 text-red-600' :
                   result.riskLevel === 'Medium' ? 'bg-amber-100 text-amber-600' :
                   'bg-emerald-100 text-emerald-600'
               }`}>
                   {result.churnProbability}%
               </span>
               <h3 className="text-xl font-bold text-slate-800">{result.riskLevel} Risk</h3>
               <p className="text-slate-500 text-sm mt-1">Churn Probability</p>
            </div>
            
            <div className="space-y-6">
                <div>
                    <h4 className="font-semibold text-slate-800 mb-2 flex items-center gap-2">
                        <AlertCircle className="w-4 h-4 text-slate-400" /> Reasoning
                    </h4>
                    <p className="text-slate-600 text-sm leading-relaxed">
                        {result.reasoning}
                    </p>
                </div>

                <div>
                    <h4 className="font-semibold text-slate-800 mb-2 flex items-center gap-2">
                        <CheckCircle2 className="w-4 h-4 text-slate-400" /> Key Factors
                    </h4>
                    <ul className="space-y-2">
                        {result.keyFactors.map((factor, idx) => (
                            <li key={idx} className="text-sm text-slate-600 flex items-start gap-2">
                                <span className="w-1.5 h-1.5 rounded-full bg-indigo-500 mt-1.5 flex-shrink-0" />
                                {factor}
                            </li>
                        ))}
                    </ul>
                </div>
            </div>
          </div>
        ) : (
          <div className="bg-slate-50 border border-dashed border-slate-200 rounded-xl h-full p-8 flex flex-col items-center justify-center text-center">
             <div className="w-16 h-16 bg-indigo-50 rounded-full flex items-center justify-center mb-4">
                <span className="text-2xl">🤖</span>
             </div>
             <h3 className="text-slate-800 font-medium">Ready to Predict</h3>
             <p className="text-slate-500 text-sm mt-2">
                 Fill out the form and let Gemini analyze the customer profile for churn risk.
             </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default PredictionForm;