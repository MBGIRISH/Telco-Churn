import { GoogleGenAI, Type } from "@google/genai";

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

export interface PredictionResult {
  churnProbability: number;
  riskLevel: 'Low' | 'Medium' | 'High';
  reasoning: string;
  keyFactors: string[];
}

export const predictChurn = async (customerData: any): Promise<PredictionResult> => {
  try {
    const model = "gemini-2.5-flash";
    const prompt = `
      You are an expert Churn Prediction ML model for a Telco company. 
      Analyze the following customer data and predict the probability of churn.
      
      Customer Data:
      ${JSON.stringify(customerData, null, 2)}

      Provide a response in JSON format with the following schema:
      {
        "churnProbability": number (0-100),
        "riskLevel": "Low" | "Medium" | "High",
        "reasoning": "A concise explanation of why this customer might churn or stay.",
        "keyFactors": ["Factor 1", "Factor 2", "Factor 3"]
      }
      
      Do not include markdown code blocks in the response, just the raw JSON.
    `;

    const response = await ai.models.generateContent({
      model: model,
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
            type: Type.OBJECT,
            properties: {
                churnProbability: { type: Type.NUMBER },
                riskLevel: { type: Type.STRING, enum: ["Low", "Medium", "High"] },
                reasoning: { type: Type.STRING },
                keyFactors: { type: Type.ARRAY, items: { type: Type.STRING } }
            }
        }
      }
    });

    const text = response.text;
    if (!text) throw new Error("No response from Gemini");

    return JSON.parse(text) as PredictionResult;

  } catch (error) {
    console.error("Gemini Prediction Error:", error);
    // Fallback mock response if API fails or key is missing
    return {
      churnProbability: 45.5,
      riskLevel: "Medium",
      reasoning: "This is a fallback prediction. Please check your API key configuration. The customer has a month-to-month contract which is a high risk factor, but tenure is moderate.",
      keyFactors: ["Contract Type", "Monthly Charges"]
    };
  }
};

export const generateSqlScript = async (): Promise<string> => {
    // Mock generator for the SQL requirement
    return `-- SQL Script to create Customers Table
CREATE TABLE customers (
    customerID VARCHAR(20) PRIMARY KEY,
    gender VARCHAR(10),
    SeniorCitizen INT,
    Partner VARCHAR(5),
    Dependents VARCHAR(5),
    tenure INT,
    PhoneService VARCHAR(5),
    MultipleLines VARCHAR(20),
    InternetService VARCHAR(20),
    OnlineSecurity VARCHAR(20),
    OnlineBackup VARCHAR(20),
    DeviceProtection VARCHAR(20),
    TechSupport VARCHAR(20),
    StreamingTV VARCHAR(20),
    StreamingMovies VARCHAR(20),
    Contract VARCHAR(20),
    PaperlessBilling VARCHAR(5),
    PaymentMethod VARCHAR(30),
    MonthlyCharges DECIMAL(10, 2),
    TotalCharges DECIMAL(10, 2),
    Churn VARCHAR(5)
);

-- Example Select Query
SELECT customerID, tenure, MonthlyCharges, Churn 
FROM customers 
WHERE tenure < 12 AND MonthlyCharges > 50;
`;
}