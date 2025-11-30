export interface Customer {
  customerID: string;
  gender: string;
  SeniorCitizen: number;
  Partner: string;
  Dependents: string;
  tenure: number;
  PhoneService: string;
  MultipleLines: string;
  InternetService: string;
  OnlineSecurity: string;
  OnlineBackup: string;
  DeviceProtection: string;
  TechSupport: string;
  StreamingTV: string;
  StreamingMovies: string;
  Contract: string;
  PaperlessBilling: string;
  PaymentMethod: string;
  MonthlyCharges: number;
  TotalCharges: string;
  Churn: string;
}

// A representative subset of the provided data for visualization
export const sampleData: Customer[] = [
  { customerID: "7590-VHVEG", gender: "Female", SeniorCitizen: 0, Partner: "Yes", Dependents: "No", tenure: 1, PhoneService: "No", MultipleLines: "No phone service", InternetService: "DSL", OnlineSecurity: "No", OnlineBackup: "Yes", DeviceProtection: "No", TechSupport: "No", StreamingTV: "No", StreamingMovies: "No", Contract: "Month-to-month", PaperlessBilling: "Yes", PaymentMethod: "Electronic check", MonthlyCharges: 29.85, TotalCharges: "29.85", Churn: "No" },
  { customerID: "5575-GNVDE", gender: "Male", SeniorCitizen: 0, Partner: "No", Dependents: "No", tenure: 34, PhoneService: "Yes", MultipleLines: "No", InternetService: "DSL", OnlineSecurity: "Yes", OnlineBackup: "No", DeviceProtection: "Yes", TechSupport: "No", StreamingTV: "No", StreamingMovies: "No", Contract: "One year", PaperlessBilling: "No", PaymentMethod: "Mailed check", MonthlyCharges: 56.95, TotalCharges: "1889.5", Churn: "No" },
  { customerID: "3668-QPYBK", gender: "Male", SeniorCitizen: 0, Partner: "No", Dependents: "No", tenure: 2, PhoneService: "Yes", MultipleLines: "No", InternetService: "DSL", OnlineSecurity: "Yes", OnlineBackup: "Yes", DeviceProtection: "No", TechSupport: "No", StreamingTV: "No", StreamingMovies: "No", Contract: "Month-to-month", PaperlessBilling: "Yes", PaymentMethod: "Mailed check", MonthlyCharges: 53.85, TotalCharges: "108.15", Churn: "Yes" },
  { customerID: "7795-CFOCW", gender: "Male", SeniorCitizen: 0, Partner: "No", Dependents: "No", tenure: 45, PhoneService: "No", MultipleLines: "No phone service", InternetService: "DSL", OnlineSecurity: "Yes", OnlineBackup: "No", DeviceProtection: "Yes", TechSupport: "Yes", StreamingTV: "No", StreamingMovies: "No", Contract: "One year", PaperlessBilling: "No", PaymentMethod: "Bank transfer (automatic)", MonthlyCharges: 42.3, TotalCharges: "1840.75", Churn: "No" },
  { customerID: "9237-HQITU", gender: "Female", SeniorCitizen: 0, Partner: "No", Dependents: "No", tenure: 2, PhoneService: "Yes", MultipleLines: "No", InternetService: "Fiber optic", OnlineSecurity: "No", OnlineBackup: "No", DeviceProtection: "No", TechSupport: "No", StreamingTV: "No", StreamingMovies: "No", Contract: "Month-to-month", PaperlessBilling: "Yes", PaymentMethod: "Electronic check", MonthlyCharges: 70.7, TotalCharges: "151.65", Churn: "Yes" },
  { customerID: "9305-CDSKC", gender: "Female", SeniorCitizen: 0, Partner: "No", Dependents: "No", tenure: 8, PhoneService: "Yes", MultipleLines: "Yes", InternetService: "Fiber optic", OnlineSecurity: "No", OnlineBackup: "No", DeviceProtection: "Yes", TechSupport: "No", StreamingTV: "Yes", StreamingMovies: "Yes", Contract: "Month-to-month", PaperlessBilling: "Yes", PaymentMethod: "Electronic check", MonthlyCharges: 99.65, TotalCharges: "820.5", Churn: "Yes" },
  { customerID: "1452-KIOVK", gender: "Male", SeniorCitizen: 0, Partner: "No", Dependents: "Yes", tenure: 22, PhoneService: "Yes", MultipleLines: "Yes", InternetService: "Fiber optic", OnlineSecurity: "No", OnlineBackup: "Yes", DeviceProtection: "No", TechSupport: "No", StreamingTV: "Yes", StreamingMovies: "No", Contract: "Month-to-month", PaperlessBilling: "Yes", PaymentMethod: "Credit card (automatic)", MonthlyCharges: 89.1, TotalCharges: "1949.4", Churn: "No" },
  { customerID: "6713-OKOMC", gender: "Female", SeniorCitizen: 0, Partner: "No", Dependents: "No", tenure: 10, PhoneService: "No", MultipleLines: "No phone service", InternetService: "DSL", OnlineSecurity: "Yes", OnlineBackup: "No", DeviceProtection: "No", TechSupport: "No", StreamingTV: "No", StreamingMovies: "No", Contract: "Month-to-month", PaperlessBilling: "No", PaymentMethod: "Mailed check", MonthlyCharges: 29.75, TotalCharges: "301.9", Churn: "No" },
  { customerID: "7892-POOKP", gender: "Female", SeniorCitizen: 0, Partner: "Yes", Dependents: "No", tenure: 28, PhoneService: "Yes", MultipleLines: "Yes", InternetService: "Fiber optic", OnlineSecurity: "No", OnlineBackup: "No", DeviceProtection: "Yes", TechSupport: "Yes", StreamingTV: "Yes", StreamingMovies: "Yes", Contract: "Month-to-month", PaperlessBilling: "Yes", PaymentMethod: "Electronic check", MonthlyCharges: 104.8, TotalCharges: "3046.05", Churn: "Yes" },
  { customerID: "6388-TABGU", gender: "Male", SeniorCitizen: 0, Partner: "No", Dependents: "Yes", tenure: 62, PhoneService: "Yes", MultipleLines: "No", InternetService: "DSL", OnlineSecurity: "Yes", OnlineBackup: "Yes", DeviceProtection: "No", TechSupport: "No", StreamingTV: "No", StreamingMovies: "No", Contract: "One year", PaperlessBilling: "No", PaymentMethod: "Bank transfer (automatic)", MonthlyCharges: 56.15, TotalCharges: "3487.95", Churn: "No" },
  { customerID: "9763-GRSKD", gender: "Male", SeniorCitizen: 0, Partner: "Yes", Dependents: "Yes", tenure: 13, PhoneService: "Yes", MultipleLines: "No", InternetService: "DSL", OnlineSecurity: "Yes", OnlineBackup: "No", DeviceProtection: "No", TechSupport: "No", StreamingTV: "No", StreamingMovies: "No", Contract: "Month-to-month", PaperlessBilling: "Yes", PaymentMethod: "Mailed check", MonthlyCharges: 49.95, TotalCharges: "587.45", Churn: "No" },
  { customerID: "7469-LKBCI", gender: "Male", SeniorCitizen: 0, Partner: "No", Dependents: "No", tenure: 16, PhoneService: "Yes", MultipleLines: "No", InternetService: "No", OnlineSecurity: "No internet service", OnlineBackup: "No internet service", DeviceProtection: "No internet service", TechSupport: "No internet service", StreamingTV: "No internet service", StreamingMovies: "No internet service", Contract: "Two year", PaperlessBilling: "No", PaymentMethod: "Credit card (automatic)", MonthlyCharges: 18.95, TotalCharges: "326.8", Churn: "No" },
  { customerID: "8091-TTVAX", gender: "Male", SeniorCitizen: 0, Partner: "Yes", Dependents: "No", tenure: 58, PhoneService: "Yes", MultipleLines: "Yes", InternetService: "Fiber optic", OnlineSecurity: "No", OnlineBackup: "No", DeviceProtection: "Yes", TechSupport: "No", StreamingTV: "Yes", StreamingMovies: "Yes", Contract: "One year", PaperlessBilling: "No", PaymentMethod: "Credit card (automatic)", MonthlyCharges: 100.35, TotalCharges: "5681.1", Churn: "No" },
  { customerID: "0280-XJGEX", gender: "Male", SeniorCitizen: 0, Partner: "No", Dependents: "No", tenure: 49, PhoneService: "Yes", MultipleLines: "Yes", InternetService: "Fiber optic", OnlineSecurity: "No", OnlineBackup: "Yes", DeviceProtection: "Yes", TechSupport: "No", StreamingTV: "Yes", StreamingMovies: "Yes", Contract: "Month-to-month", PaperlessBilling: "Yes", PaymentMethod: "Bank transfer (automatic)", MonthlyCharges: 103.7, TotalCharges: "5036.3", Churn: "Yes" },
  { customerID: "5129-JLPIS", gender: "Male", SeniorCitizen: 0, Partner: "No", Dependents: "No", tenure: 25, PhoneService: "Yes", MultipleLines: "No", InternetService: "Fiber optic", OnlineSecurity: "Yes", OnlineBackup: "No", DeviceProtection: "Yes", TechSupport: "Yes", StreamingTV: "Yes", StreamingMovies: "Yes", Contract: "Month-to-month", PaperlessBilling: "Yes", PaymentMethod: "Electronic check", MonthlyCharges: 105.5, TotalCharges: "2686.05", Churn: "No" },
  { customerID: "3655-SNQYZ", gender: "Female", SeniorCitizen: 0, Partner: "Yes", Dependents: "Yes", tenure: 69, PhoneService: "Yes", MultipleLines: "Yes", InternetService: "Fiber optic", OnlineSecurity: "Yes", OnlineBackup: "Yes", DeviceProtection: "Yes", TechSupport: "Yes", StreamingTV: "Yes", StreamingMovies: "Yes", Contract: "Two year", PaperlessBilling: "No", PaymentMethod: "Credit card (automatic)", MonthlyCharges: 113.25, TotalCharges: "7895.15", Churn: "No" },
  { customerID: "8191-XWSZG", gender: "Female", SeniorCitizen: 0, Partner: "No", Dependents: "No", tenure: 52, PhoneService: "Yes", MultipleLines: "No", InternetService: "No", OnlineSecurity: "No internet service", OnlineBackup: "No internet service", DeviceProtection: "No internet service", TechSupport: "No internet service", StreamingTV: "No internet service", StreamingMovies: "No internet service", Contract: "One year", PaperlessBilling: "No", PaymentMethod: "Mailed check", MonthlyCharges: 20.65, TotalCharges: "1022.95", Churn: "No" },
  { customerID: "9959-WOFKT", gender: "Male", SeniorCitizen: 0, Partner: "No", Dependents: "Yes", tenure: 71, PhoneService: "Yes", MultipleLines: "Yes", InternetService: "Fiber optic", OnlineSecurity: "Yes", OnlineBackup: "No", DeviceProtection: "Yes", TechSupport: "No", StreamingTV: "Yes", StreamingMovies: "Yes", Contract: "Two year", PaperlessBilling: "No", PaymentMethod: "Bank transfer (automatic)", MonthlyCharges: 106.7, TotalCharges: "7382.25", Churn: "No" },
  { customerID: "4190-MFLUW", gender: "Female", SeniorCitizen: 0, Partner: "Yes", Dependents: "Yes", tenure: 10, PhoneService: "Yes", MultipleLines: "No", InternetService: "DSL", OnlineSecurity: "No", OnlineBackup: "No", DeviceProtection: "Yes", TechSupport: "Yes", StreamingTV: "No", StreamingMovies: "No", Contract: "Month-to-month", PaperlessBilling: "No", PaymentMethod: "Credit card (automatic)", MonthlyCharges: 55.2, TotalCharges: "528.35", Churn: "Yes" },
  { customerID: "4183-MYFRB", gender: "Female", SeniorCitizen: 0, Partner: "No", Dependents: "No", tenure: 21, PhoneService: "Yes", MultipleLines: "No", InternetService: "Fiber optic", OnlineSecurity: "No", OnlineBackup: "Yes", DeviceProtection: "Yes", TechSupport: "No", StreamingTV: "No", StreamingMovies: "Yes", Contract: "Month-to-month", PaperlessBilling: "Yes", PaymentMethod: "Electronic check", MonthlyCharges: 90.05, TotalCharges: "1862.9", Churn: "No" }
];

// Helper to get stats
export const getStats = () => {
  const total = sampleData.length;
  const churned = sampleData.filter(c => c.Churn === "Yes").length;
  const churnRate = (churned / total) * 100;
  const avgMonthly = sampleData.reduce((acc, curr) => acc + curr.MonthlyCharges, 0) / total;

  return {
    total,
    churned,
    churnRate,
    avgMonthly
  };
};