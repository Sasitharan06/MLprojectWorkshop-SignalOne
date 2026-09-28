import pandas as pd
import numpy as np

df = pd.read_csv('telco_customer_churn_cleaned.csv')

print('='*60)
print('CHURN DISTRIBUTION')
print('='*60)
print(df['Churn'].value_counts())
print(f'Overall churn rate: {df["Churn"].mean()*100:.2f}%')

print()
print('='*60)
print('CUSTOMER STATUS DISTRIBUTION')
print('='*60)
print(df['Customer Status'].value_counts())

print()
print('='*60)
print('CHURN vs CUSTOMER STATUS CROSS-TAB')
print('='*60)
print(pd.crosstab(df['Customer Status'], df['Churn']))

print()
print('='*60)
print('CHURN CATEGORY DISTRIBUTION')
print('='*60)
print(df['Churn Category'].value_counts())

print()
print('='*60)
print('CHURN SCORE SUMMARY')
print('='*60)
print(df['Churn Score'].describe())

print()
print('='*60)
print('SATISFACTION SCORE DISTRIBUTION')
print('='*60)
print(df['Satisfaction Score'].value_counts().sort_index())

print()
print('='*60)
print('CLTV SUMMARY')
print('='*60)
print(df['CLTV'].describe())

print()
print('='*60)
print('TENURE SUMMARY')
print('='*60)
print(df['Tenure in Months'].describe())

print()
print('='*60)
print('MONTHLY CHARGE SUMMARY')
print('='*60)
print(df['Monthly Charge'].describe())

print()
print('='*60)
print('TOTAL CHARGES SUMMARY')
print('='*60)
print(df['Total Charges'].describe())

print()
print('='*60)
print('CORRELATION: Total Charges vs Monthly x Tenure')
print('='*60)
df['derived_total'] = df['Monthly Charge'] * df['Tenure in Months']
corr = df['Total Charges'].corr(df['derived_total'])
print(f'Correlation(Total Charges, Monthly Charge x Tenure): {corr:.4f}')

print()
print('='*60)
print('CORRELATION MATRIX - Financial columns')
print('='*60)
fin_cols = ['Monthly Charge','Total Charges','Total Revenue','Total Long Distance Charges','Total Extra Data Charges','CLTV','Tenure in Months']
print(df[fin_cols].corr().round(3).to_string())

print()
print('='*60)
print('CONTRACT DISTRIBUTION')
print('='*60)
print(df['Contract'].value_counts())

print()
print('='*60)
print('INTERNET TYPE DISTRIBUTION')
print('='*60)
print(df['Internet Type'].value_counts(dropna=False))

print()
print('='*60)
print('PAYMENT METHOD DISTRIBUTION')
print('='*60)
print(df['Payment Method'].value_counts())

print()
print('='*60)
print('OFFER DISTRIBUTION (with NaN)')
print('='*60)
print(df['Offer'].value_counts(dropna=False))

print()
print('='*60)
print('SINGLE-VALUE COLUMNS (to drop)')
print('='*60)
for col in df.columns:
    if df[col].nunique() == 1:
        print(f'  {col} = {df[col].iloc[0]}')

print()
print('='*60)
print('NEW/JOINED CUSTOMERS ANALYSIS')
print('='*60)
joined = df[df['Customer Status'] == 'Joined']
print(f'Joined customers: {len(joined)}')
print(f'Avg tenure (Joined): {joined["Tenure in Months"].mean():.2f}')
print(f'Avg tenure (others): {df[df["Customer Status"] != "Joined"]["Tenure in Months"].mean():.2f}')

print()
print('='*60)
print('AGE DISTRIBUTION')
print('='*60)
print(df['Age'].describe())
print(f'\nSenior Citizen: {df["Senior Citizen"].sum()} ({df["Senior Citizen"].mean()*100:.1f}%)')
print(f'Under 30: {df["Under 30"].sum()} ({df["Under 30"].mean()*100:.1f}%)')
