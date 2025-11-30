import React from 'react';
import { sampleData, getStats } from '../utils/csvData';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  ScatterChart,
  Scatter,
  ZAxis
} from 'recharts';

const Dashboard: React.FC = () => {
  const stats = getStats();
  
  // Data processing for charts
  const churnData = [
    { name: 'Churn', value: stats.churned },
    { name: 'Retained', value: stats.total - stats.churned },
  ];
  
  const internetServiceData = sampleData.reduce((acc: any[], curr) => {
    const existing = acc.find(item => item.name === curr.InternetService);
    if (existing) {
      existing.total++;
      if (curr.Churn === 'Yes') existing.churned++;
    } else {
      acc.push({ name: curr.InternetService, total: 1, churned: curr.Churn === 'Yes' ? 1 : 0 });
    }
    return acc;
  }, []);

  const COLORS = ['#ef4444', '#10b981'];

  return (
    <div className="space-y-6">
      {/* KPIS */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-100">
          <h3 className="text-slate-500 text-sm font-medium">Total Customers</h3>
          <p className="text-2xl font-bold text-slate-800">{stats.total}</p>
        </div>
        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-100">
          <h3 className="text-slate-500 text-sm font-medium">Churn Rate</h3>
          <p className="text-2xl font-bold text-red-500">{stats.churnRate.toFixed(1)}%</p>
        </div>
        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-100">
          <h3 className="text-slate-500 text-sm font-medium">Avg. Monthly Charges</h3>
          <p className="text-2xl font-bold text-slate-800">${stats.avgMonthly.toFixed(2)}</p>
        </div>
         <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-100">
          <h3 className="text-slate-500 text-sm font-medium">Data Source</h3>
          <p className="text-sm font-bold text-indigo-600 mt-2">Telco Customer CSV</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Churn Distribution */}
        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-100">
          <h3 className="text-lg font-semibold text-slate-800 mb-4">Overall Churn Distribution</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={churnData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  fill="#8884d8"
                  paddingAngle={5}
                  dataKey="value"
                >
                  {churnData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Churn by Internet Service */}
        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-100">
          <h3 className="text-lg font-semibold text-slate-800 mb-4">Churn by Internet Service</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={internetServiceData}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="name" axisLine={false} tickLine={false} />
                <YAxis axisLine={false} tickLine={false} />
                <Tooltip cursor={{fill: 'transparent'}} />
                <Legend />
                <Bar dataKey="total" name="Total Customers" fill="#e2e8f0" radius={[4, 4, 0, 0]} />
                <Bar dataKey="churned" name="Churned Customers" fill="#ef4444" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

         {/* Tenure vs Monthly Charges */}
         <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-100 col-span-1 lg:col-span-2">
          <h3 className="text-lg font-semibold text-slate-800 mb-4">Tenure vs Monthly Charges (Churn Analysis)</h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                <CartesianGrid />
                <XAxis type="number" dataKey="tenure" name="Tenure (months)" unit="mo" />
                <YAxis type="number" dataKey="MonthlyCharges" name="Monthly Charges" unit="$" />
                <ZAxis type="category" dataKey="Churn" name="Churn" />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                <Legend />
                <Scatter name="Customers" data={sampleData} fill="#4f46e5" />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;