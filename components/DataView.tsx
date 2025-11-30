import React, { useState } from 'react';
import { sampleData } from '../utils/csvData';
import { generateSqlScript } from '../services/gemini';
import { Database, Code, Download, Search } from 'lucide-react';

const DataView: React.FC = () => {
  const [showSql, setShowSql] = useState(false);
  const [sqlScript, setSqlScript] = useState('');

  const handleGenerateSql = async () => {
      const script = await generateSqlScript();
      setSqlScript(script);
      setShowSql(true);
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
          <h2 className="text-xl font-semibold text-slate-800">Customer Data (Sample)</h2>
          <div className="flex gap-3">
             <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
                <input 
                    type="text" 
                    placeholder="Search customers..." 
                    className="pl-10 pr-4 py-2 border border-slate-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
             </div>
             <button 
                onClick={handleGenerateSql}
                className="flex items-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg text-sm font-medium transition-colors"
             >
                <Code className="w-4 h-4" /> Generate SQL
             </button>
          </div>
      </div>

      {/* Table */}
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden overflow-x-auto">
        <table className="w-full text-sm text-left">
          <thead className="bg-slate-50 text-slate-500 font-medium border-b border-slate-200">
            <tr>
              <th className="px-6 py-4">Customer ID</th>
              <th className="px-6 py-4">Gender</th>
              <th className="px-6 py-4">Tenure</th>
              <th className="px-6 py-4">Contract</th>
              <th className="px-6 py-4">Payment Method</th>
              <th className="px-6 py-4">Monthly Charges</th>
              <th className="px-6 py-4">Churn</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {sampleData.map((customer) => (
              <tr key={customer.customerID} className="hover:bg-slate-50 transition-colors">
                <td className="px-6 py-4 font-medium text-slate-900">{customer.customerID}</td>
                <td className="px-6 py-4 text-slate-600">{customer.gender}</td>
                <td className="px-6 py-4 text-slate-600">{customer.tenure} mos</td>
                <td className="px-6 py-4 text-slate-600">{customer.Contract}</td>
                <td className="px-6 py-4 text-slate-600">{customer.PaymentMethod}</td>
                <td className="px-6 py-4 text-slate-600 font-medium">${customer.MonthlyCharges}</td>
                <td className="px-6 py-4">
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                        customer.Churn === 'Yes' 
                        ? 'bg-red-100 text-red-700' 
                        : 'bg-green-100 text-green-700'
                    }`}>
                        {customer.Churn}
                    </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* SQL Modal/Section */}
      {showSql && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
              <div className="bg-white rounded-xl shadow-2xl max-w-2xl w-full overflow-hidden flex flex-col max-h-[80vh]">
                  <div className="p-4 border-b flex justify-between items-center bg-slate-50">
                      <h3 className="font-semibold text-slate-800 flex items-center gap-2">
                          <Database className="w-4 h-4 text-indigo-600" /> SQL Integration
                      </h3>
                      <button onClick={() => setShowSql(false)} className="text-slate-400 hover:text-slate-600">
                          &times;
                      </button>
                  </div>
                  <div className="p-0 overflow-auto bg-slate-900 text-slate-50 font-mono text-sm">
                      <pre className="p-4">{sqlScript}</pre>
                  </div>
                  <div className="p-4 border-t bg-slate-50 flex justify-end">
                      <button 
                        className="flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700"
                        onClick={() => setShowSql(false)}
                      >
                          <Download className="w-4 h-4" /> Download Script
                      </button>
                  </div>
              </div>
          </div>
      )}
    </div>
  );
};

export default DataView;