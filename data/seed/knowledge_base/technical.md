# NovaTech Solutions - Technical Support

## Account Setup

### How do I create an account?

Click "Sign Up" in the top right corner. Enter your email, create a password (minimum 8 characters), and verify your email. You'll be guided through initial setup including company name and preferences. Account creation takes less than 2 minutes.

### Can I use my account on multiple devices?

Yes! Log in from any device using your email and password. Your data syncs automatically across all devices. You can have up to 5 active sessions simultaneously. Manage active sessions in Settings > Security > Active Sessions.

### How do I add team members?

Go to Settings > Team > Invite Members. Enter email addresses (one per line) and select their role (Admin, Member, or Viewer). They'll receive an invitation email with a setup link. Invitations expire after 7 days.

### What are the different user roles?

- **Admin**: Full access including billing, team management, and all features
- **Member**: Can create and edit content, view analytics, no access to billing/team settings
- **Viewer**: Read-only access to shared dashboards and reports

## Password and Security

### I forgot my password, how do I reset it?

Click "Forgot Password" on the login page. Enter your email and click "Send Reset Link." Check your email (and spam folder) for a password reset link. The link expires after 1 hour. Create a new password (minimum 8 characters).

### How do I change my password?

Log in and go to Settings > Security > Change Password. Enter your current password, then your new password twice. Click "Update Password." You'll be logged out of all devices and need to log in again with your new password.

### Do you support two-factor authentication (2FA)?

Yes! We strongly recommend enabling 2FA. Go to Settings > Security > Two-Factor Authentication. Choose SMS or authenticator app (we recommend Google Authenticator or Authy). Follow the setup wizard. Save your backup codes in a safe place.

### What should I do if my account is compromised?

Immediately change your password at Settings > Security > Change Password. Enable 2FA if not already active. Review Settings > Security > Active Sessions and revoke any unfamiliar sessions. Contact security@novatech.com if you notice unauthorized activity.

## API Access

### How do I get API access?

API access is available on Professional and Enterprise plans. Go to Settings > API > Generate API Key. Give your key a descriptive name, select permissions, and click "Create." Store your key securely - it's only shown once!

### Where is the API documentation?

Full API documentation is at https://api.novatech.com/docs. It includes authentication, endpoints, request/response examples, rate limits, and SDKs for Python, JavaScript, and Ruby.

### What are the API rate limits?

- **Professional**: 1,000 requests per hour
- **Enterprise**: 10,000 requests per hour

If you consistently hit limits, contact sales@novatech.com to discuss custom plans. Rate limit headers are included in every API response.

### Can I regenerate my API key?

Yes, in Settings > API > Manage Keys. Click "Regenerate" next to your key. This immediately invalidates the old key and generates a new one. Update your applications with the new key to avoid disruptions.

## Troubleshooting

### The application is running slowly, what can I do?

First, try these steps:
1. Clear your browser cache (Ctrl/Cmd + Shift + Delete)
2. Disable browser extensions temporarily
3. Try a different browser (we recommend Chrome or Firefox)
4. Check your internet connection speed

If issues persist, contact support@novatech.com with your browser version and a description of the slowness.

### I'm getting a "Session Expired" error

This happens if you've been inactive for 2 hours or logged in from another device. Click "Log In Again" and re-enter your credentials. To stay logged in longer, check "Keep me signed in" on the login page.

### Features are missing after an update

Clear your browser cache and hard refresh (Ctrl/Cmd + Shift + R). If features are still missing, check if you need to re-grant permissions in Settings > Permissions. Contact support if issues continue.

### My data isn't syncing across devices

Check your internet connection on both devices. Verify you're logged in with the same account. Force a sync by clicking your profile icon > Sync Now. Syncing usually takes 5-10 seconds. Contact support if data doesn't sync within 5 minutes.

## Data Import and Export

### Can I import data from other platforms?

Yes! We support CSV, Excel, and JSON imports. Go to Settings > Import/Export > Import Data. Select your file format, map columns to our fields, and click "Start Import." Large imports are processed in the background with email notification on completion.

### How do I export my data?

Go to Settings > Import/Export > Export Data. Choose your format (CSV, Excel, JSON) and date range. Click "Export." Small exports download immediately; large exports are emailed as a download link within 30 minutes.

### What data can I export?

You can export all your data: contacts, invoices, inventory items, CRM notes, analytics, and settings. Exports preserve all custom fields and metadata. Enterprise plans include scheduled exports (daily, weekly, monthly).

## Mobile App

### Is there a mobile app?

Yes! Download "NovaTech" from the App Store (iOS 14+) or Google Play (Android 8+). Log in with your existing credentials. Mobile and web data sync automatically.

### Why isn't the mobile app syncing?

Check your internet connection and enable mobile data for the app. Go to Settings > Sync and toggle "Auto-Sync" on. Manually sync by pulling down on the main screen. If issues persist, log out and log back in.

### What features are available on mobile?

Most features are available on mobile: invoicing, inventory management, CRM, dashboards. Advanced features like bulk operations and detailed analytics work best on desktop. We're continuously improving mobile feature parity.

## Integrations

### What integrations do you offer?

We integrate with 100+ platforms including QuickBooks, Shopify, Stripe, Slack, Zapier, Gmail, and Salesforce. Browse all integrations at Settings > Integrations or visit https://novatech.com/integrations.

### How do I connect an integration?

Go to Settings > Integrations, find your platform, and click "Connect." You'll be redirected to authorize NovaTech. Grant the requested permissions and you'll be redirected back. Most integrations sync within 5 minutes.

### Can I disconnect an integration?

Yes, at Settings > Integrations > Active Integrations. Click the three dots next to the integration and select "Disconnect." This stops all data syncing immediately. Your data in NovaTech is retained.
