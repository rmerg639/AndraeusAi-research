# Stripe Payment Setup Guide

## Setting Up Stripe for License Payments

### Step 1: Create Stripe Account

1. Go to: https://stripe.com/au
2. Click "Start now"
3. Enter email: andraeusbeats@gmail.com
4. Create password
5. Verify email

---

### Step 2: Business Information

**Business Details:**
```
Business Type: Sole Proprietor (or Company if you have Pty Ltd)
Business Name: Andraeus AI
Industry: Software / Technology
Website: https://github.com/rmerg639/AndraeusAi-research
```

**Personal Details:**
```
Full Name: Rocco Andraeus Sergi
Date of Birth: [YOUR DOB]
Address: [YOUR ADDRESS], NSW, Australia
Phone: [YOUR PHONE]
```

**Bank Account:**
```
Bank: [YOUR BANK]
BSB: [YOUR BSB]
Account Number: [YOUR ACCOUNT]
```

---

### Step 3: Verification

Stripe will require:
- Government ID (driver's license or passport)
- Proof of address (utility bill or bank statement)
- ABN (once registered)

---

### Step 4: Create Payment Links

For each license tier, create payment links:

**Tier 7 - Enterprise Minimum ($25,000 AUD/year)**
```
Product: Andraeus AI Enterprise License (Annual Minimum)
Price: $25,000 AUD
Type: One-time or Recurring
Description: Enterprise commercial license for Andraeus AI technology.
             Annual minimum fee. Additional fees based on 3.5% net profits.
```

**Tier 8-14 - Premium Industries**
```
Product: Andraeus AI Premium Industry License
Price: Custom quote
Type: Invoice
Description: License for mining, military, gambling, surveillance,
             predatory finance, fossil fuel, or tobacco/alcohol industries.
             10% of net profits. Contact for quote.
```

---

### Step 5: Create Invoice Templates

**Enterprise Invoice Template:**
```
INVOICE

From: Rocco Andraeus Sergi
      Andraeus AI
      andraeusbeats@gmail.com
      ABN: [YOUR ABN]

To: [CLIENT NAME]
    [CLIENT ADDRESS]

Invoice Number: ANDRAI-2025-001
Date: [DATE]
Due Date: [DATE + 30 DAYS]

Description                              Amount (AUD)
---------------------------------------------------------
Andraeus AI Enterprise License
Q[X] 2025 - 3.5% Net Profits            $[AMOUNT]
(Based on reported profits of $X)

Subtotal:                                $[AMOUNT]
GST (10%):                               $[AMOUNT]
---------------------------------------------------------
TOTAL DUE:                               $[AMOUNT]

Payment Methods:
- Bank Transfer: [BSB] / [ACCOUNT]
- Stripe: [PAYMENT LINK]
- PayPal: andraeusbeats@gmail.com

Payment Terms: Net 30 days
Late Fee: 2% per month on overdue balance
```

---

### Step 6: Set Up Recurring Billing

For quarterly payments:
1. Stripe Dashboard → Products
2. Create product for each tier
3. Set billing interval to "Every 3 months"
4. Send subscription link to clients

---

### Step 7: Tax Settings

**Australian GST:**
- Register for GST if revenue > $75,000/year
- Add 10% GST to all invoices
- Configure Stripe Tax for automatic GST

---

### Step 8: Stripe Dashboard

Bookmark these pages:
- Dashboard: https://dashboard.stripe.com
- Payments: https://dashboard.stripe.com/payments
- Invoices: https://dashboard.stripe.com/invoices
- Customers: https://dashboard.stripe.com/customers

---

## Alternative: PayPal Business

If you prefer PayPal:

1. Go to: https://www.paypal.com/au/business
2. Create business account
3. Link to andraeusbeats@gmail.com
4. Verify bank account
5. Create invoices from PayPal dashboard

---

## Cost Summary

| Service | Cost |
|---------|------|
| Stripe account | FREE |
| Stripe transaction fee | 1.75% + $0.30 per transaction |
| PayPal transaction fee | 2.6% + $0.30 per transaction |
| Bank transfer | FREE |

---

## Recommended Setup

1. **Primary:** Stripe (for credit cards, professional invoices)
2. **Secondary:** Bank transfer (for large payments, no fees)
3. **Backup:** PayPal (for international, convenience)
