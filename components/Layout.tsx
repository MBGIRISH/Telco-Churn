import React, { ReactNode } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { LayoutDashboard, BrainCircuit, Database, UserCircle } from 'lucide-react';

interface LayoutProps {
  children: ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const location = useLocation();

  const navItems = [
    { name: 'Dashboard', path: '/', icon: <LayoutDashboard className="w-5 h-5" /> },
    { name: 'Predict Churn', path: '/predict', icon: <BrainCircuit className="w-5 h-5" /> },
    { name: 'Data View', path: '/data', icon: <Database className="w-5 h-5" /> },
  ];

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col md:flex-row">
      {/* Sidebar */}
      <aside className="w-full md:w-64 bg-white border-r border-slate-200 md:h-screen fixed z-10 top-0 left-0 bottom-0 overflow-y-auto">
        <div className="p-6 border-b border-slate-100">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-indigo-600 rounded-lg flex items-center justify-center text-white shadow-lg shadow-indigo-200">
                <BrainCircuit className="w-6 h-6" />
            </div>
            <div>
                <h1 className="font-bold text-slate-800 text-lg leading-tight">Telco<br/>Churn AI</h1>
            </div>
          </div>
        </div>

        <nav className="p-4 space-y-1">
          {navItems.map((item) => (
            <Link
              key={item.path}
              to={item.path}
              className={`flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all ${
                location.pathname === item.path
                  ? 'bg-indigo-50 text-indigo-600 shadow-sm'
                  : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900'
              }`}
            >
              {item.icon}
              {item.name}
            </Link>
          ))}
        </nav>

        <div className="absolute bottom-0 w-full p-4 border-t border-slate-100">
            <div className="flex items-center gap-3 px-2 py-2">
                <UserCircle className="w-8 h-8 text-slate-400" />
                <div>
                    <p className="text-sm font-medium text-slate-700">Admin User</p>
                    <p className="text-xs text-slate-500">ML Engineer</p>
                </div>
            </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 md:ml-64 p-6 md:p-8">
        <div className="max-w-6xl mx-auto">
             {children}
        </div>
      </main>
    </div>
  );
};

export default Layout;